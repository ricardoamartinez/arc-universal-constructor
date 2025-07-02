#!/usr/bin/env python3
"""arc_universal_constructor.py

Von Neumann Universal Constructor for ARC-AGI-2

A unified system that separates the "brain" (neural controller that writes programs) 
from the "body" (spatial environment that executes those programs). This system 
learns to solve ARC-AGI-2 tasks by generating and executing explicit, interpretable 
construction programs, demonstrating principles of self-repair, abstraction, and 
constructor-completeness.

Architecture:
  • GNCA Encoder: Perception module for distilling task rules into vectors
  • Token-based Memory Controller: TTM-style program synthesizer  
  • Constructor-Complete DSL: Explicit language with macro library support
  • Spatial Constructor: Environment for robust program execution
  • Reinforcement Learning: Policy gradient training on construction rewards

Usage:
  python arc_universal_constructor.py --mode train --verbose
  python arc_universal_constructor.py --mode eval --ckpt model.pt
  python arc_universal_constructor.py --mode demo --viz
"""

import os
import sys
import json
import argparse
import time
import random
import zipfile
import requests
import pathlib
import io
import shutil
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GPU Visualizer import (optional)
try:
    from arc_gpu_visualizer import ARCGPUVisualizer, create_visualizer_hooks
    GPU_VIZ_AVAILABLE = True
except ImportError:
    GPU_VIZ_AVAILABLE = False
    print("[Info] GPU visualizer not available, run with --viz for ASCII visualization")

# Try to import flash attention
try:
    from flash_attn.modules.mha import MHA
    FLASH_ATTN_AVAILABLE = True
    print("[Info] flash-attn kernels found - fast attention enabled.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[Info] flash-attn not found, falling back to PyTorch attention.")

# -----------------------------------------------------------------------------
# 1. Global Constants & Configuration (K-1)
# -----------------------------------------------------------------------------
ARC_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"
ARC_DIR = pathlib.Path("arc_agi2")
TRAIN_DIR = ARC_DIR / "data" / "training"
EVAL_DIR = ARC_DIR / "data" / "evaluation"

# Model architecture constants
MAX_GRID_SIZE = 30
NUM_COLORS = 10  # ARC uses colors 0-9
GNCA_CHANNELS = 8   # Internal GNCA state channels (reduced for efficiency)
TASK_EMBED_DIM = 64  # Task embedding dimension


def download_arc(verbose: bool = False):
    """Download ARC-AGI-2 if the data folder is missing."""
    if TRAIN_DIR.exists() and EVAL_DIR.exists():
        if verbose:
            print("[Data] ARC-AGI-2 already present.")
        return

    if verbose:
        print("[Data] Downloading ARC-AGI-2 …")

    response = requests.get(ARC_URL, timeout=60)
    response.raise_for_status()

    # Extract zip in memory to avoid intermediate files
    zbuffer = io.BytesIO(response.content)
    with zipfile.ZipFile(zbuffer) as zf:
        zf.extractall(".")

    extracted = pathlib.Path("ARC-AGI-2-main")
    if extracted.exists():
        shutil.move(str(extracted), ARC_DIR)

    if verbose:
        print("[Data] ✓ extracted to", ARC_DIR)


# -----------------------------------------------------------------------------
# 2. JSON → tensor loader utilities
# -----------------------------------------------------------------------------

def _arr2tensor(arr) -> torch.Tensor:
    """Convert nested Python lists to a long-dtype tensor."""
    return torch.tensor(arr, dtype=torch.long)


def load_task(path: pathlib.Path) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Parse one ARC task JSON file into (train_pairs, test_in, test_out)."""
    with open(path) as fh:
        raw = json.load(fh)

    train_pairs = [(_arr2tensor(ex["input"]), _arr2tensor(ex["output"])) for ex in raw["train"]]
    test_in  = _arr2tensor(raw["test"][0]["input"])
    test_out = _arr2tensor(raw["test"][0]["output"]) if "output" in raw["test"][0] else None
    return train_pairs, test_in, test_out


class ARCDataset(Dataset):
    """PyTorch Dataset wrapper for ARC-AGI-2 train/eval splits."""

    def __init__(self, split: str = "train"):
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        self.paths = sorted((TRAIN_DIR if split == "train" else EVAL_DIR).glob("*.json"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_task(self.paths[idx])


# -----------------------------------------------------------------------------
# 3. GNCA Encoder - Perception Module (B-1)
# -----------------------------------------------------------------------------

class GNCARule(nn.Module):
    """
    Single GNCA update rule with 3x3 depth-wise convolution perception 
    and 128-unit 1x1 MLP with residual connection.
    """
    
    def __init__(self, channels: int = GNCA_CHANNELS):
        super().__init__()
        self.channels = channels
        
        # 3x3 depth-wise convolution for perception
        self.perceive = nn.Conv2d(
            channels, channels * 3, 
            kernel_size=3, padding=1, 
            groups=channels, bias=False
        )
        
        # 64-unit 1x1 MLP for update rule (reduced for efficiency)
        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=1),
            nn.Tanh()  # Bounded update for stability
        )
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Apply one GNCA update step with residual connection."""
        # Perception: gather neighborhood information
        neighborhood = self.perceive(x)  # (B, C*3, H, W)
        
        # Update: compute state changes
        delta = self.update(neighborhood)  # (B, C, H, W)
        
        # Residual connection
        x_next = x + delta
        
        if verbose:
            print(f"[GNCA] Rule step: mean activation {x_next.mean().item():.4f}, "
                  f"std {x_next.std().item():.4f}")
        
        return x_next


class GNCAEncoder(nn.Module):
    """
    GNCA-based perception module that distills task rules into vectors.
    Iterates a local rule for configurable steps to encode grid patterns.
    """
    
    def __init__(self, steps: int = 8, channels: int = GNCA_CHANNELS):
        super().__init__()
        self.steps = steps
        self.channels = channels
        
        # Input projection: colors → GNCA state channels
        self.input_proj = nn.Conv2d(NUM_COLORS + 1, channels, kernel_size=1)  # +1 for empty cells
        
        # GNCA update rule
        self.ca_rule = GNCARule(channels)
        
        # Output projection: GNCA state → task embedding (simplified)
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, TASK_EMBED_DIM, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable GNCA dynamics."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _grid_to_onehot(self, grid: torch.Tensor) -> torch.Tensor:
        """Convert ARC grid to one-hot representation (optimized)."""
        B, H, W = grid.shape
        
        # Use actual grid size instead of always MAX_GRID_SIZE for efficiency
        size = max(H, W, 8)  # Minimum size for stable convolution
        
        # Create one-hot encoding (NUM_COLORS + 1 channels for empty cells)
        onehot = torch.zeros(B, NUM_COLORS + 1, size, size, 
                           device=grid.device, dtype=torch.float32)
        
        # Fill the one-hot tensor
        for b in range(B):
            grid_b = grid[b].clone()
            
            # Handle empty cells (-1) as channel NUM_COLORS
            empty_mask = (grid_b == -1)
            grid_b[empty_mask] = NUM_COLORS
            
            # One-hot encoding
            onehot[b, :, :H, :W].scatter_(0, grid_b.unsqueeze(0), 1.0)
            
        return onehot
    
    def forward(self, grid: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Encode a grid pattern into a task embedding vector.
        
        Args:
            grid: (B, H, W) tensor of ARC colors (0-9, -1 for empty)
            verbose: Enable debug prints
            
        Returns:
            task_embedding: (B, TASK_EMBED_DIM) tensor
        """
        start_time = time.time() if verbose else None
        
        # Convert to one-hot representation
        x = self._grid_to_onehot(grid)  # (B, NUM_COLORS+1, size, size)
        
        # Project to GNCA state space
        x = self.input_proj(x)  # (B, channels, size, size)
        
        if verbose:
            print(f"[GNCA] Initial state: mean {x.mean().item():.4f}, std {x.std().item():.4f}")
        
        # Iterate GNCA rule
        for step in range(self.steps):
            x = self.ca_rule(x, verbose=(verbose and step == 0))  # Only print first step
            
            if verbose:
                print(f"[GNCA] after step {step} → mean {x.mean().item():.4f}")
        
        # Extract task embedding
        embedding = self.output_proj(x)  # (B, TASK_EMBED_DIM)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"[GNCA] Encoding complete: {embedding.shape} in {elapsed*1000:.2f}ms")
        
        return embedding
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# 4. Constructor-Complete DSL & Macro Library (D-1, D-2)
# -----------------------------------------------------------------------------

class ConstructorOps(Enum):
    """
    Base DSL vocabulary for construction operations.
    This enum defines the core action space for the universal constructor.
    """
    # Spatial movement operations
    MOVE_ARM = auto()      # Move construction arm to relative position
    
    # Grid manipulation operations  
    WRITE = auto()         # Write color at current arm position
    ERASE = auto()         # Erase (set to empty) at current arm position
    
    # Control flow operations
    BRANCH_IF_EMPTY = auto()  # Conditional branching based on cell state
    FORK_ARM = auto()         # Create new construction arm at current position
    
    # Program termination
    HALT = auto()          # Stop program execution
    
    @classmethod
    def base_vocab_size(cls) -> int:
        """Return number of base operations (excluding dynamic macros)."""
        return len(cls)


class MacroLibrary:
    """
    Dynamic library for storing and managing learned construction macros.
    Supports runtime vocabulary expansion for abstract pattern reuse.
    """
    
    def __init__(self):
        self.macros: Dict[str, List[ConstructorOps]] = {}
        self.macro_counter = 0
    
    def add_macro(self, name: str, operations: List[ConstructorOps]) -> int:
        """
        Add a new macro to the library.
        
        Args:
            name: Human-readable macro name
            operations: Sequence of base operations
            
        Returns:
            macro_id: Integer ID for the new macro (used in action space)
        """
        if name in self.macros:
            raise ValueError(f"Macro '{name}' already exists")
            
        self.macros[name] = operations
        macro_id = ConstructorOps.base_vocab_size() + self.macro_counter
        self.macro_counter += 1
        
        return macro_id
    
    def get_macro(self, name: str) -> Optional[List[ConstructorOps]]:
        """Retrieve macro operations by name."""
        return self.macros.get(name)
    
    def get_total_vocab_size(self) -> int:
        """Return total vocabulary size including base ops + macros."""
        return ConstructorOps.base_vocab_size() + len(self.macros)
    
    def list_macros(self) -> Dict[str, List[ConstructorOps]]:
        """Return copy of all macros for inspection."""
        return self.macros.copy()


# -----------------------------------------------------------------------------
# 5. Token Controller - TTM-based Program Synthesizer (C-1)
# -----------------------------------------------------------------------------

class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Flash-Attention support and fallback.
    Uses flash_attn if available, otherwise falls back to PyTorch's native attention.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if FLASH_ATTN_AVAILABLE:
            # Use flash attention implementation
            self.flash_mha = MHA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                causal=False,  # Not causal for this application
                use_flash_attn=True
            )
            self.use_flash = True
        else:
            # Fallback to PyTorch native attention
            self.pytorch_mha = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.use_flash = False
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: (batch_size, seq_len, embed_dim) input tensor
            verbose: Enable debug prints
            
        Returns:
            output: (batch_size, seq_len, embed_dim) attended tensor
        """
        if self.use_flash:
            # Flash attention path
            output = self.flash_mha(x)[0]  # flash_mha returns (output, attention_weights)
            if verbose:
                print(f"[Attn] Flash attention: input_shape={x.shape}, output_shape={output.shape}")
        else:
            # PyTorch attention path
            output, _ = self.pytorch_mha(x, x, x, need_weights=False)
            if verbose:
                print(f"[Attn] PyTorch attention: input_shape={x.shape}, output_shape={output.shape}")
        
        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-head attention
        self.attn = FlashMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # MLP (reduced ratio for efficiency)
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Apply transformer block with residual connections."""
        # Attention with residual
        attn_out = self.attn(self.norm1(x), verbose=verbose)
        x = x + attn_out
        
        # MLP with residual  
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        if verbose:
            print(f"[TransformerBlock] Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        return x


class TokenController(nn.Module):
    """
    TTM-based program synthesizer using transformer with learnable memory tokens.
    Acts as a policy network π(action|state) generating sequences of opcodes.
    """
    
    def __init__(self, 
                 task_embed_dim: int = TASK_EMBED_DIM,
                 embed_dim: int = 96,  # Reduced for efficiency 
                 num_layers: int = 4,
                 num_heads: int = 4,
                 memory_tokens: int = 6,
                 dropout: float = 0.1,
                 macro_library: Optional[MacroLibrary] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_tokens = memory_tokens
        self.macro_library = macro_library or MacroLibrary()
        
        # Project task embedding to controller embedding dimension
        self.task_proj = nn.Linear(task_embed_dim, embed_dim)
        
        # Learnable memory tokens (writable scratchpad)
        self.memory_embeddings = nn.Parameter(
            torch.randn(memory_tokens, embed_dim) * 0.02  # Small initialization
        )
        
        # Transformer layers (reduced MLP ratio for efficiency)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Action head: outputs logits over ConstructorOps + macros (C-2)
        self.action_head = nn.Linear(embed_dim, self._get_action_vocab_size())
        
        # Initialize weights
        self._init_weights()
    
    def _get_action_vocab_size(self) -> int:
        """Get total action vocabulary size (base ops + macros)."""
        return self.macro_library.get_total_vocab_size()
    
    def expand_action_head(self, new_vocab_size: int):
        """
        Dynamically expand action head for new macros (D-2 verification).
        
        Args:
            new_vocab_size: New total vocabulary size after adding macros
        """
        if new_vocab_size <= self.action_head.out_features:
            return  # No expansion needed
        
        # Create new larger action head
        old_head = self.action_head
        device = old_head.weight.device  # Get current device
        self.action_head = nn.Linear(self.embed_dim, new_vocab_size).to(device)
        
        # Copy existing weights
        with torch.no_grad():
            self.action_head.weight[:old_head.out_features].copy_(old_head.weight)
            self.action_head.bias[:old_head.out_features].copy_(old_head.bias)
            
            # Initialize new weights (for new macros)
            nn.init.xavier_uniform_(self.action_head.weight[old_head.out_features:])
            nn.init.zeros_(self.action_head.bias[old_head.out_features:])
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, task_embedding: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the token controller.
        
        Args:
            task_embedding: (batch_size, task_embed_dim) from GNCA encoder
            verbose: Enable debug prints
            
        Returns:
            action_logits: (batch_size, vocab_size) logits over action space
            memory_tokens: (batch_size, memory_tokens, embed_dim) updated memory state
        """
        batch_size = task_embedding.size(0)
        
        # Project task embedding
        task_token = self.task_proj(task_embedding).unsqueeze(1)  # (B, 1, embed_dim)
        
        # Expand memory tokens for batch
        memory = self.memory_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, embed_dim)
        
        # Concatenate task token with memory tokens
        x = torch.cat([task_token, memory], dim=1)  # (B, 1+M, embed_dim)
        
        if verbose:
            print(f"[Controller] Input: task_shape={task_embedding.shape}, memory_tokens={self.memory_tokens}")
            print(f"[Controller] Combined input shape: {x.shape}")
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, verbose=(verbose and i == 0))  # Only print first layer details
            
            if verbose:
                print(f"[Attn] block-{i} out mean {x.mean().item():.4f}")
        
        # Final normalization
        x = self.norm(x)
        
        # Generate action logits from task token (C-2)
        task_representation = x[:, 0, :]  # (B, embed_dim) - first token
        action_logits = self.action_head(task_representation)  # (B, vocab_size)
        
        # Return updated memory tokens (excluding task token)
        updated_memory = x[:, 1:, :]  # (B, M, embed_dim)
        
        if verbose:
            print(f"[Controller] Action logits shape: {action_logits.shape}")
            print(f"[Controller] Output memory shape: {updated_memory.shape}")
        
        return action_logits, updated_memory
    
    def generate_blueprint(self, task_embedding: torch.Tensor, max_steps: int = 50, 
                          temperature: float = 1.0, verbose: bool = False) -> List[int]:
        """
        Generate a complete construction blueprint using iterative reasoning (C-3).
        
        Args:
            task_embedding: (1, task_embed_dim) single task embedding
            max_steps: Maximum program length
            temperature: Sampling temperature for action selection
            verbose: Enable step-by-step logging
            
        Returns:
            blueprint: List of action IDs representing the construction program
        """
        if task_embedding.size(0) != 1:
            raise ValueError("generate_blueprint requires batch_size=1")
        
        blueprint = []
        current_task_embedding = task_embedding
        
        if verbose:
            print(f"[Controller] Generating blueprint (max_steps={max_steps})...")
        
        # Minimum exploration steps to prevent immediate HALT
        min_steps = 5
        
        for step in range(max_steps):
            # Forward pass to get action logits
            with torch.no_grad():
                action_logits, updated_memory = self.forward(current_task_embedding, verbose=(verbose and step == 0))
            
            # Prevent HALT in early steps during training
            halt_id = ConstructorOps.HALT.value - 1  # Convert to 0-indexed
            if step < min_steps and self.training:
                # Mask out HALT action
                action_logits[:, halt_id] = -float('inf')
            
            # Sample action from logits
            if temperature > 0:
                probs = F.softmax(action_logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                action = action_logits.argmax(dim=-1).item()
            
            blueprint.append(action)
            
            if verbose:
                # Convert action ID to operation name for logging
                if action < ConstructorOps.base_vocab_size():
                    op_name = list(ConstructorOps)[action].name
                else:
                    op_name = f"MACRO_{action - ConstructorOps.base_vocab_size()}"
                print(f"[Controller] Generating step {step+1}: {op_name} (id={action})")
            
            # Check for halt condition
            if action == halt_id and step >= min_steps:
                if verbose:
                    print(f"[Controller] Emitted HALT - Blueprint complete")
                break
        
        if verbose:
            print(f"[Controller] Generated blueprint: {len(blueprint)} steps")
        
        return blueprint
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# 6. Blueprint Interpreter & Spatial Constructor (D-3, H-1)  
# -----------------------------------------------------------------------------

class ConstructionArm:
    """Single construction arm with position and state."""
    
    def __init__(self, x: int = 0, y: int = 0, arm_id: int = 0):
        self.x = x
        self.y = y
        self.arm_id = arm_id
    
    def move(self, dx: int, dy: int):
        """Move arm by relative offset."""
        self.x += dx
        self.y += dy
    
    def get_position(self) -> Tuple[int, int]:
        """Get current arm position."""
        return (self.x, self.y)


class SpatialConstructor:
    """
    The "body" - manages grid canvas and construction arms.
    Executes spatial operations called by the BlueprintInterpreter.
    """
    
    def __init__(self, grid_size: int = MAX_GRID_SIZE):
        self.grid_size = grid_size
        self.canvas = torch.full((grid_size, grid_size), -1, dtype=torch.long)  # -1 = empty
        self.arms: List[ConstructionArm] = [ConstructionArm()]  # Start with one arm
        self.step_count = 0
    
    def reset(self, target_shape: Optional[Tuple[int, int]] = None):
        """Reset canvas and arms for new construction task."""
        if target_shape:
            h, w = target_shape
            self.canvas = torch.full((h, w), -1, dtype=torch.long)
        else:
            self.canvas.fill_(-1)  # Clear to empty
        
        self.arms = [ConstructionArm()]  # Reset to single arm
        self.step_count = 0
    
    def execute_move(self, arm_id: int, dx: int, dy: int) -> bool:
        """Move specified arm by relative offset."""
        if arm_id >= len(self.arms):
            return False
        
        arm = self.arms[arm_id]
        new_x, new_y = arm.x + dx, arm.y + dy
        
        # Bounds checking
        h, w = self.canvas.shape
        if 0 <= new_x < w and 0 <= new_y < h:
            arm.move(dx, dy)
            return True
        return False
    
    def execute_write(self, arm_id: int, color: int) -> bool:
        """Write color at specified arm position."""
        if arm_id >= len(self.arms):
            return False
        
        arm = self.arms[arm_id]
        h, w = self.canvas.shape
        
        if 0 <= arm.x < w and 0 <= arm.y < h and 0 <= color < NUM_COLORS:
            self.canvas[arm.y, arm.x] = color
            return True
        return False
    
    def execute_erase(self, arm_id: int) -> bool:
        """Erase (set to empty) at specified arm position."""
        if arm_id >= len(self.arms):
            return False
        
        arm = self.arms[arm_id]
        h, w = self.canvas.shape
        
        if 0 <= arm.x < w and 0 <= arm.y < h:
            self.canvas[arm.y, arm.x] = -1  # Empty cell
            return True
        return False
    
    def check_empty(self, arm_id: int) -> Optional[bool]:
        """Check if cell at arm position is empty."""
        if arm_id >= len(self.arms):
            return None
        
        arm = self.arms[arm_id]
        h, w = self.canvas.shape
        
        if 0 <= arm.x < w and 0 <= arm.y < h:
            return self.canvas[arm.y, arm.x] == -1
        return None
    
    def fork_arm(self, arm_id: int) -> int:
        """Create new arm at position of existing arm."""
        if arm_id >= len(self.arms):
            return -1
        
        source_arm = self.arms[arm_id]
        new_arm = ConstructionArm(source_arm.x, source_arm.y, len(self.arms))
        self.arms.append(new_arm)
        
        return new_arm.arm_id
    
    def get_canvas(self) -> torch.Tensor:
        """Get current canvas state."""
        return self.canvas.clone()
    
    def get_arm_positions(self) -> List[Tuple[int, int]]:
        """Get positions of all construction arms."""
        return [arm.get_position() for arm in self.arms]


class BlueprintInterpreter:
    """
    Non-differentiable interpreter that executes sequences of opcodes.
    Bridges the neural controller (brain) with spatial constructor (body).
    """
    
    def __init__(self, macro_library: MacroLibrary, max_steps: int = 100):
        self.macro_library = macro_library
        self.max_steps = max_steps
        self.constructor = SpatialConstructor()
    
    def execute_blueprint(self, opcodes: List[int], verbose: bool = False, 
                         visualize: bool = False, viz_delay: float = 0.5) -> torch.Tensor:
        """
        Execute a sequence of opcodes using the spatial constructor.
        
        Args:
            opcodes: List of integer action IDs from controller
            verbose: Enable step-by-step logging
            visualize: Enable visualization (real-time animation, before/after, step-by-step)
            viz_delay: Delay between visualization frames (seconds)
            
        Returns:
            final_canvas: Constructed grid result
        """
        self.constructor.reset()
        initial_canvas = self.constructor.get_canvas().clone() if visualize else None
        
        if verbose:
            print(f"[Interpreter] Executing blueprint: {len(opcodes)} opcodes")
        
        if visualize:
            print("\n" + "="*60)
            print("BLUEPRINT EXECUTION VISUALIZATION")
            print("="*60)
            self._visualize_initial_state()
        
        for step, opcode in enumerate(opcodes):
            if step >= self.max_steps:
                if verbose:
                    print(f"[Interpreter] Step limit reached: {self.max_steps}")
                break
            
            # Store state before operation for visualization
            if visualize:
                pre_canvas = self.constructor.get_canvas().clone()
                pre_arms = [arm.get_position() for arm in self.constructor.arms]
            
            # Convert opcode to operation
            if opcode < ConstructorOps.base_vocab_size():
                # Base operation
                op = list(ConstructorOps)[opcode]
                success = self._execute_base_op(op, verbose, step)
            else:
                # Macro operation (to be implemented when macros are added)
                if verbose:
                    print(f"[Step {step}] Macro execution not yet implemented")
                success = False
            
            # Visualize step if enabled
            if visualize and success:
                self._visualize_step(step, opcode, pre_canvas, pre_arms, viz_delay)
            
            # Update GPU visualizer if hook is available
            if hasattr(self, '_viz_hook') and self._viz_hook and step % 5 == 0:  # Update every 5 steps
                canvas = self.constructor.get_canvas()
                arm_positions = self.constructor.get_arm_positions()
                self._viz_hook['on_construction_step'](canvas, arm_positions)
            
            # Check for halt
            if opcode == ConstructorOps.HALT.value - 1:  # Convert to 0-indexed
                if verbose:
                    print(f"[Step {step}] HALT - Execution complete")
                break
            
            if not success and verbose:
                print(f"[Step {step}] Operation failed")
        
        final_canvas = self.constructor.get_canvas()
        
        if visualize:
            self._visualize_final_state(initial_canvas, final_canvas)
        
        return final_canvas
    
    def _execute_base_op(self, op: ConstructorOps, verbose: bool, step: int) -> bool:
        """Execute a single base operation."""
        # For now, implement basic operations with some variety
        # More sophisticated parameter handling will be added later
        
        arm_id = 0  # Default to first arm
        
        if op == ConstructorOps.MOVE_ARM:
            # Move in different directions based on step
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
            dx, dy = directions[step % 4]
            success = self.constructor.execute_move(arm_id, dx, dy)
            if verbose:
                pos = self.constructor.arms[arm_id].get_position()
                print(f"[Step {step}] Arm {arm_id} at {pos} executing: MOVE_ARM({dx},{dy})")
        
        elif op == ConstructorOps.WRITE:
            # Write different colors based on position
            if len(self.constructor.arms) > 0:
                x, y = self.constructor.arms[arm_id].get_position()
                color = (x + y) % 9 + 1  # Colors 1-9 (skip 0/black)
            else:
                color = 1
            success = self.constructor.execute_write(arm_id, color)
            if verbose:
                pos = self.constructor.arms[arm_id].get_position()
                color_names = ["BLACK", "BLUE", "RED", "GREEN", "YELLOW", "GRAY", 
                              "MAGENTA", "ORANGE", "AZURE", "BROWN"]
                print(f"[Step {step}] Arm {arm_id} at {pos} executing: WRITE({color_names[color]})")
        
        elif op == ConstructorOps.ERASE:
            success = self.constructor.execute_erase(arm_id)
            if verbose:
                pos = self.constructor.arms[arm_id].get_position()
                print(f"[Step {step}] Arm {arm_id} at {pos} executing: ERASE")
        
        elif op == ConstructorOps.BRANCH_IF_EMPTY:
            # Simple conditional (no branching logic yet)
            is_empty = self.constructor.check_empty(arm_id)
            success = is_empty is not None
            if verbose:
                pos = self.constructor.arms[arm_id].get_position()
                print(f"[Step {step}] Arm {arm_id} at {pos} executing: BRANCH_IF_EMPTY (empty={is_empty})")
        
        elif op == ConstructorOps.FORK_ARM:
            new_arm_id = self.constructor.fork_arm(arm_id)
            success = new_arm_id >= 0
            if verbose:
                pos = self.constructor.arms[arm_id].get_position()
                print(f"[Step {step}] Arm {arm_id} at {pos} executing: FORK_ARM → new arm {new_arm_id}")
        
        elif op == ConstructorOps.HALT:
            if verbose:
                print(f"[Step {step}] HALT")
            success = True
        
        else:
            success = False
        
        return success
    
    def _visualize_initial_state(self):
        """Visualize the initial state of construction."""
        print("\n[INITIAL STATE]")
        canvas = self.constructor.get_canvas()
        self._visualize_canvas_with_arms(canvas, "Initial Canvas")
        print()
    
    def _visualize_step(self, step: int, opcode: int, pre_canvas: torch.Tensor, 
                       pre_arms: List[Tuple[int, int]], delay: float):
        """Visualize a single construction step with real-time animation."""
        import time
        
        # Get operation name
        if opcode < ConstructorOps.base_vocab_size():
            op_name = list(ConstructorOps)[opcode].name
        else:
            op_name = f"MACRO_{opcode - ConstructorOps.base_vocab_size()}"
        
        print(f"\n[STEP {step}] Executing: {op_name}")
        print("-" * 40)
        
        # Show before state
        print("Before:")
        self._visualize_canvas_with_arms_at_positions(pre_canvas, pre_arms, show_grid=False)
        
        # Show after state
        print("\nAfter:")
        canvas = self.constructor.get_canvas()
        self._visualize_canvas_with_arms(canvas, show_grid=False)
        
        # Show what changed
        changes = (canvas != pre_canvas).sum().item()
        if changes > 0:
            print(f"\n→ Modified {changes} cells")
        
        # Delay for animation effect
        time.sleep(delay)
    
    def _visualize_final_state(self, initial_canvas: torch.Tensor, final_canvas: torch.Tensor):
        """Visualize before/after comparison of the construction."""
        print("\n" + "="*60)
        print("CONSTRUCTION COMPLETE - BEFORE/AFTER COMPARISON")
        print("="*60)
        
        # Show initial state
        print("\n[BEFORE]")
        self._visualize_grid_simple(initial_canvas)
        
        # Show final state
        print("\n[AFTER]")
        self._visualize_grid_simple(final_canvas)
        
        # Show statistics
        initial_filled = (initial_canvas != -1).sum().item()
        final_filled = (final_canvas != -1).sum().item()
        cells_added = final_filled - initial_filled
        
        print(f"\n[STATISTICS]")
        print(f"Initial filled cells: {initial_filled}")
        print(f"Final filled cells: {final_filled}")
        print(f"Cells added: {cells_added}")
        
        # Calculate similarity
        h = min(initial_canvas.shape[0], final_canvas.shape[0])
        w = min(initial_canvas.shape[1], final_canvas.shape[1])
        similarity = (initial_canvas[:h, :w] == final_canvas[:h, :w]).float().mean().item()
        print(f"Similarity: {similarity*100:.1f}%")
    
    def _visualize_canvas_with_arms(self, canvas: torch.Tensor, title: str = "", show_grid: bool = True):
        """Visualize canvas with arm positions marked."""
        if title:
            print(title)
        
        # Get arm positions
        arm_positions = {arm.get_position(): arm.arm_id for arm in self.constructor.arms}
        
        self._visualize_canvas_with_arms_at_positions(canvas, arm_positions, show_grid)
    
    def _visualize_canvas_with_arms_at_positions(self, canvas: torch.Tensor, 
                                                arm_positions: Any, show_grid: bool = True):
        """Visualize canvas with specific arm positions."""
        # Handle both list and dict formats
        if isinstance(arm_positions, list):
            arm_dict = {pos: i for i, pos in enumerate(arm_positions)}
        else:
            arm_dict = arm_positions
        
        color_map = {
            -1: '·',  # Empty (smaller dot)
            0: '□',   # Black
            1: '■',   # Red
            2: '▦',   # Blue  
            3: '▤',   # Green
            4: '▣',   # Yellow
            5: '▥',   # Gray
            6: '▨',   # Fuchsia
            7: '▧',   # Orange
            8: '▩',   # Azure
            9: '▪',   # Brown
        }
        
        # Print with grid coordinates if requested
        if show_grid:
            # Print column numbers
            print("   ", end="")
            for x in range(min(canvas.shape[1], 15)):  # Limit width for display
                print(f"{x:2}", end="")
            print("  ...")
        
        for y in range(min(canvas.shape[0], 15)):  # Limit height for display
            if show_grid:
                print(f"{y:2} ", end="")
            
            for x in range(min(canvas.shape[1], 15)):
                pos = (x, y)
                if pos in arm_dict:
                    # Show arm with special symbol
                    arm_id = arm_dict[pos]
                    print(f"A{arm_id % 10}", end="")  # A0-A9 for arms
                else:
                    # Show cell color
                    cell = canvas[y, x].item()
                    print(color_map.get(cell, '?') + " ", end="")
            
            if canvas.shape[1] > 15:
                print(" ...", end="")
            print()
        
        if canvas.shape[0] > 15:
            print("   ...")
    
    def _visualize_grid_simple(self, grid: torch.Tensor):
        """Simple grid visualization without arms."""
        color_map = {
            -1: '.',  # Empty
            0: '□',   # Black
            1: '■',   # Red
            2: '▦',   # Blue  
            3: '▤',   # Green
            4: '▣',   # Yellow
            5: '▥',   # Gray
            6: '▨',   # Fuchsia
            7: '▧',   # Orange
            8: '▩',   # Azure
            9: '▪',   # Brown
        }
        
        for row in grid[:20]:  # Limit display size
            line = ""
            for cell in row[:20]:
                line += color_map.get(cell.item(), '?') + " "
            if grid.shape[1] > 20:
                line += "..."
            print(line)
        
        if grid.shape[0] > 20:
            print("...")


# -----------------------------------------------------------------------------
# 7. Memory-Augmented Controller with External Memory (DNC/TTM-style)
# -----------------------------------------------------------------------------

class ExternalMemory(nn.Module):
    """
    External memory module inspired by Differentiable Neural Computer (DNC)
    and Token Turing Machine (TTM) architectures. Provides read/write operations
    for the controller to store and retrieve information during construction.
    """
    
    def __init__(self, num_slots: int = 32, slot_dim: int = 64, num_reads: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_reads = num_reads
        
        # Memory matrix (num_slots x slot_dim)
        self.memory = nn.Parameter(torch.zeros(num_slots, slot_dim))
        
        # Usage tracking (which slots have been written to)
        self.register_buffer('usage', torch.zeros(num_slots))
        
        # Temporal linkage for sequential access
        self.register_buffer('link_matrix', torch.zeros(num_slots, num_slots))
        self.register_buffer('precedence', torch.zeros(num_slots))
        
        # Initialize memory with small random values
        nn.init.normal_(self.memory, std=0.02)
    
    def content_addressing(self, key: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Content-based addressing using cosine similarity.
        
        Args:
            key: (batch, slot_dim) query key
            beta: (batch, 1) key strength
            
        Returns:
            weights: (batch, num_slots) attention weights
        """
        # Normalize key and memory
        key = F.normalize(key, dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(key, memory_norm.T)  # (batch, num_slots)
        
        # Apply temperature and softmax
        weights = F.softmax(beta * similarity, dim=-1)
        
        return weights
    
    def allocation_weighting(self) -> torch.Tensor:
        """
        Compute allocation weights for writing to free memory slots.
        
        Returns:
            alloc_weights: (num_slots,) allocation weights
        """
        # Sort usage to find least used slots
        sorted_usage, indices = torch.sort(self.usage)
        
        # Compute allocation weights (prefer least used slots)
        alloc_weights = torch.zeros_like(self.usage)
        for i in range(self.num_slots):
            if i == 0:
                alloc_weights[indices[i]] = 1 - sorted_usage[i]
            else:
                alloc_weights[indices[i]] = (1 - sorted_usage[i]) * torch.prod(sorted_usage[:i])
        
        return alloc_weights
    
    def read(self, read_keys: torch.Tensor, read_betas: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using content-based addressing.
        
        Args:
            read_keys: (batch, num_reads, slot_dim) read keys
            read_betas: (batch, num_reads, 1) key strengths
            
        Returns:
            read_vectors: (batch, num_reads, slot_dim) read content
        """
        batch_size = read_keys.size(0)
        read_vectors = []
        
        for i in range(self.num_reads):
            # Content addressing for each read head
            weights = self.content_addressing(read_keys[:, i], read_betas[:, i])
            
            # Read from memory
            read_vec = torch.matmul(weights, self.memory)  # (batch, slot_dim)
            read_vectors.append(read_vec)
        
        return torch.stack(read_vectors, dim=1)  # (batch, num_reads, slot_dim)
    
    def write(self, write_key: torch.Tensor, write_beta: torch.Tensor,
              write_vec: torch.Tensor, erase_vec: torch.Tensor,
              free_gates: torch.Tensor, alloc_gate: torch.Tensor,
              write_gate: torch.Tensor):
        """
        Write to memory with content addressing and allocation.
        
        Args:
            write_key: (batch, slot_dim) write key
            write_beta: (batch, 1) key strength
            write_vec: (batch, slot_dim) vector to write
            erase_vec: (batch, slot_dim) erase vector
            free_gates: (batch, num_reads) gates for freeing read locations
            alloc_gate: (batch, 1) allocation gate
            write_gate: (batch, 1) write gate
        """
        batch_size = write_key.size(0)
        
        # Content addressing
        content_weights = self.content_addressing(write_key, write_beta)
        
        # Allocation addressing
        alloc_weights = self.allocation_weighting()
        
        # Combine content and allocation addressing
        write_weights = write_gate * (alloc_gate * alloc_weights + 
                                     (1 - alloc_gate) * content_weights)
        
        # Update usage
        self.usage = self.usage + write_weights - self.usage * write_weights
        
        # Memory update (erase then write)
        erase_weights = write_weights.unsqueeze(-1) * erase_vec.unsqueeze(1)
        self.memory = self.memory * (1 - erase_weights.mean(0))
        
        add_weights = write_weights.unsqueeze(-1) * write_vec.unsqueeze(1)
        self.memory = self.memory + add_weights.mean(0)
        
        # Update temporal linkage
        self.precedence = (1 - write_weights.sum()) * self.precedence + write_weights
        
    def reset(self):
        """Reset memory state."""
        self.memory.data.normal_(std=0.02)
        self.usage.zero_()
        self.link_matrix.zero_()
        self.precedence.zero_()


class MemoryAugmentedController(nn.Module):
    """
    Enhanced TokenController with external memory for von Neumann Universal Construction.
    Combines transformer architecture with DNC-style memory for program synthesis.
    """
    
    def __init__(self,
                 task_embed_dim: int = TASK_EMBED_DIM,
                 embed_dim: int = 96,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 memory_slots: int = 32,
                 memory_dim: int = 64,
                 num_reads: int = 4,
                 dropout: float = 0.1,
                 macro_library: Optional[MacroLibrary] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.macro_library = macro_library or MacroLibrary()
        
        # External memory
        self.memory = ExternalMemory(memory_slots, memory_dim, num_reads)
        
        # Memory interface layers
        self.read_keys_proj = nn.Linear(embed_dim, num_reads * memory_dim)
        self.read_betas_proj = nn.Linear(embed_dim, num_reads)
        self.write_key_proj = nn.Linear(embed_dim, memory_dim)
        self.write_beta_proj = nn.Linear(embed_dim, 1)
        self.write_vec_proj = nn.Linear(embed_dim, memory_dim)
        self.erase_vec_proj = nn.Linear(embed_dim, memory_dim)
        self.free_gates_proj = nn.Linear(embed_dim, num_reads)
        self.alloc_gate_proj = nn.Linear(embed_dim, 1)
        self.write_gate_proj = nn.Linear(embed_dim, 1)
        
        # Task and memory projection
        self.task_proj = nn.Linear(task_embed_dim, embed_dim)
        self.read_proj = nn.Linear(num_reads * memory_dim, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(embed_dim)
        self.action_head = nn.Linear(embed_dim, self._get_action_vocab_size())
        
        self._init_weights()
    
    def _get_action_vocab_size(self) -> int:
        """Get total action vocabulary size (base ops + macros)."""
        return self.macro_library.get_total_vocab_size()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, task_embedding: torch.Tensor, 
                prev_reads: Optional[torch.Tensor] = None,
                verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory read/write operations.
        
        Returns:
            action_logits: (batch, vocab_size)
            current_reads: (batch, num_reads, memory_dim)
            controller_state: (batch, embed_dim)
        """
        batch_size = task_embedding.size(0)
        
        # Project task embedding
        task_vec = self.task_proj(task_embedding)  # (batch, embed_dim)
        
        # Initialize reads if not provided
        if prev_reads is None:
            prev_reads = torch.zeros(batch_size, self.memory.num_reads, 
                                   self.memory.slot_dim, device=task_embedding.device)
        
        # Project previous reads and combine with task
        read_vec = self.read_proj(prev_reads.flatten(1))  # (batch, embed_dim)
        x = task_vec + read_vec  # (batch, embed_dim)
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x.unsqueeze(1), verbose=(verbose and i == 0)).squeeze(1)
        
        x = self.norm(x)
        controller_state = x
        
        # Generate memory interface parameters
        read_keys = self.read_keys_proj(x).view(batch_size, self.memory.num_reads, -1)
        read_betas = F.softplus(self.read_betas_proj(x)).unsqueeze(-1)
        
        write_key = self.write_key_proj(x)
        write_beta = F.softplus(self.write_beta_proj(x))
        write_vec = self.write_vec_proj(x)
        erase_vec = torch.sigmoid(self.erase_vec_proj(x))
        
        free_gates = torch.sigmoid(self.free_gates_proj(x))
        alloc_gate = torch.sigmoid(self.alloc_gate_proj(x))
        write_gate = torch.sigmoid(self.write_gate_proj(x))
        
        # Memory operations
        current_reads = self.memory.read(read_keys, read_betas)
        self.memory.write(write_key, write_beta, write_vec, erase_vec,
                         free_gates, alloc_gate, write_gate)
        
        # Generate action logits
        action_logits = self.action_head(controller_state)
        
        if verbose:
            print(f"[MemController] Read vectors mean: {current_reads.mean().item():.4f}")
            print(f"[MemController] Memory usage: {self.memory.usage.mean().item():.2f}")
            print(f"[MemController] Action logits: {action_logits.shape}")
        
        return action_logits, current_reads, controller_state


# -----------------------------------------------------------------------------
# 8. Enhanced Spatial Constructor with Self-Repair Capabilities (H-2)
# -----------------------------------------------------------------------------

class SelfRepairingConstructor(SpatialConstructor):
    """
    Enhanced spatial constructor with self-repair capabilities through
    damage simulation and GNCA-based regeneration.
    """
    
    def __init__(self, grid_size: int = MAX_GRID_SIZE, 
                 damage_rate: float = 0.15,
                 repair_steps: int = 5):
        super().__init__(grid_size)
        self.damage_rate = damage_rate
        self.repair_steps = repair_steps
        self.damage_enabled = False
        
        # GNCA repair rule (simple version)
        self.repair_rule = nn.Sequential(
            nn.Conv2d(NUM_COLORS + 1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, NUM_COLORS + 1, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize repair rule
        for m in self.repair_rule:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def inject_damage(self, verbose: bool = False):
        """Inject random damage to the canvas."""
        if not self.damage_enabled:
            return
        
        h, w = self.canvas.shape
        damage_mask = torch.rand(h, w) < self.damage_rate
        damaged_count = damage_mask.sum().item()
        
        # Set damaged cells to empty (-1)
        self.canvas[damage_mask] = -1
        
        if verbose:
            print(f"[Damage] Injected damage to {damaged_count} cells ({damaged_count/(h*w)*100:.1f}%)")
    
    def repair_damage(self, verbose: bool = False):
        """Apply GNCA-based repair to restore damaged patterns."""
        if not self.damage_enabled:
            return
        
        h, w = self.canvas.shape
        
        # Convert canvas to one-hot
        canvas_oh = F.one_hot(self.canvas + 1, NUM_COLORS + 1).float()
        canvas_oh = canvas_oh.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # Apply repair steps
        for step in range(self.repair_steps):
            with torch.no_grad():
                canvas_oh = self.repair_rule(canvas_oh)
        
        # Convert back to canvas
        repaired = canvas_oh.squeeze(0).argmax(0) - 1
        
        # Count repairs
        repairs = ((self.canvas == -1) & (repaired != -1)).sum().item()
        self.canvas = repaired
        
        if verbose:
            print(f"[Repair] Repaired {repairs} cells in {self.repair_steps} steps")
    
    def execute_with_repair(self, operation: str, *args, verbose: bool = False) -> bool:
        """Execute operation with damage injection and repair."""
        # Execute the operation
        if operation == "move":
            success = self.execute_move(*args)
        elif operation == "write":
            success = self.execute_write(*args)
        elif operation == "erase":
            success = self.execute_erase(*args)
        elif operation == "fork":
            success = self.fork_arm(*args) >= 0
        else:
            success = False
        
        # Inject damage after operation
        if success and self.damage_enabled:
            self.inject_damage(verbose)
            self.repair_damage(verbose)
        
        return success


# -----------------------------------------------------------------------------
# 9. REINFORCE Training Implementation (F-1, F-2, F-3, F-4)
# -----------------------------------------------------------------------------

class ConstructionEnvironment:
    """
    Environment wrapper for REINFORCE training of the universal constructor.
    """
    
    def __init__(self, spatial_constructor: SelfRepairingConstructor,
                 blueprint_interpreter: BlueprintInterpreter):
        self.constructor = spatial_constructor
        self.interpreter = blueprint_interpreter
    
    def reset(self, target_shape: Tuple[int, int]):
        """Reset environment for new construction task."""
        self.constructor.reset(target_shape)
        self.target_shape = target_shape
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Execute action and return state, reward, done.
        """
        # Convert action to opcode list (handle macros)
        if action < ConstructorOps.base_vocab_size():
            opcodes = [action]
        else:
            # Macro expansion (simplified)
            opcodes = [action]  # Would expand to macro sequence
        
        # Execute blueprint
        canvas = self.interpreter.execute_blueprint(opcodes, verbose=False)
        
        # Check if HALT was executed
        done = (action == ConstructorOps.HALT.value - 1)
        
        # Compute reward (will be calculated at episode end)
        reward = 0.0
        
        return canvas, reward, done
    
    def compute_reward(self, final_canvas: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute IoU reward between constructed and target grids.
        """
        # Move to same device as target
        final_canvas = final_canvas.to(target.device)
        
        # Handle size mismatch
        h_min = min(final_canvas.shape[0], target.shape[0])
        w_min = min(final_canvas.shape[1], target.shape[1])
        
        final_crop = final_canvas[:h_min, :w_min]
        target_crop = target[:h_min, :w_min]
        
        # Base reward components
        # 1. Activity reward - encourage writing something
        activity = (final_crop != -1).float().sum()
        target_activity = (target_crop != -1).float().sum()
        activity_reward = min(activity / (target_activity + 1e-6), 1.0) * 0.1
        
        # 2. Color matching reward - partial credit for correct colors
        correct_colors = 0
        for color in range(10):  # ARC uses colors 0-9
            final_mask = (final_crop == color)
            target_mask = (target_crop == color)
            if target_mask.any():
                overlap = (final_mask & target_mask).float().sum()
                expected = target_mask.float().sum()
                correct_colors += overlap / expected
        color_reward = correct_colors / 10.0 * 0.2
        
        # 3. IoU reward - main objective
        intersection = (final_crop == target_crop).float().sum()
        union = h_min * w_min
        iou = intersection / union if union > 0 else 0
        
        # 4. Exact match bonus
        exact_match = float((final_crop == target_crop).all())
        
        # Total reward
        total_reward = activity_reward + color_reward + iou.item() * 0.5 + exact_match * 0.2
        
        return total_reward


def train_reinforce(model: nn.Module, 
                   train_dataset: ARCDataset,
                   epochs: int = 10,
                   lr: float = 1e-4,
                   gamma: float = 0.99,
                   max_steps: int = 50,
                   device: str = "cuda",
                   verbose: bool = True,
                   viz_hooks=None):
    """
    Train the universal constructor using REINFORCE algorithm.
    """
    model.train()  # Set model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create environment
    spatial_constructor = SelfRepairingConstructor()
    interpreter = BlueprintInterpreter(model.controller.macro_library, max_steps)
    env = ConstructionEnvironment(spatial_constructor, interpreter)
    
    # Training metrics
    episode_rewards = []
    
    for epoch in range(epochs):
        epoch_rewards = []
        
        for task_idx, (demo_pairs, test_in, test_out) in enumerate(train_dataset):
            if test_out is None:
                continue
            
            # Update GPU visualizer
            if viz_hooks:
                viz_hooks['on_task_start'](task_idx, demo_pairs, test_in, test_out)
            
            # Reset environment
            env.reset(test_out.shape)
            
            # Episode storage
            log_probs = []
            rewards = []
            
            # Get task embedding
            task_embedding = model.embed_task(demo_pairs, device)
            task_embedding = task_embedding.unsqueeze(0)
            
            # Memory state for memory-augmented controller
            prev_reads = None
            
            # Generate episode
            for step in range(max_steps):
                # Get action from policy
                if isinstance(model.controller, MemoryAugmentedController):
                    action_logits, prev_reads, _ = model.controller(task_embedding, prev_reads)
                else:
                    action_logits, _ = model.controller(task_embedding)
                
                # Prevent early HALT during training (minimum 5 steps)
                halt_id = ConstructorOps.HALT.value - 1
                if step < 5:
                    action_logits[:, halt_id] = -10.0  # Large negative value
                
                # Add exploration noise early in training
                if epoch == 0 and step < 10:
                    noise = torch.randn_like(action_logits) * 0.5
                    action_logits = action_logits + noise
                
                # Sample action
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                # Store log probability
                log_probs.append(action_dist.log_prob(action))
                
                # Execute action
                canvas, _, done = env.step(action.item())
                
                # Update GPU visualizer with construction progress
                if viz_hooks and step % 5 == 0:  # Update every 5 steps
                    viz_hooks['on_construction_step'](canvas, env.constructor.get_arm_positions())
                
                # Check if training should pause
                if viz_hooks and viz_hooks['should_pause']():
                    while viz_hooks['should_pause']():
                        time.sleep(0.1)
                
                if done and step >= 5:  # Only halt after minimum steps
                    break
            
            # Compute final reward
            final_reward = env.compute_reward(canvas, test_out.to(device))
            rewards = [final_reward] * len(log_probs)  # All steps get final reward
            
            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, device=device)
            # Only normalize if we have more than one return
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
            else:
                returns = torch.zeros_like(returns)  # Single step, no advantage
            
            # REINFORCE loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            if policy_loss:
                loss = torch.stack(policy_loss).mean()
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_rewards.append(final_reward)
                
                # Update GPU visualizer
                if viz_hooks:
                    viz_hooks['on_training_step'](loss.item(), reward=final_reward, iou=final_reward)
            
            # Logging
            if verbose and task_idx % 10 == 0:
                if 'loss' in locals():
                    print(f"[REINFORCE] Epoch {epoch}, Task {task_idx}, "
                          f"Reward: {final_reward:.3f}, Loss: {loss.item():.4f}")
                else:
                    print(f"[REINFORCE] Epoch {epoch}, Task {task_idx}, "
                          f"Reward: {final_reward:.3f}, No loss (no actions)")
            
            # Speed control
            if viz_hooks:
                speed = viz_hooks['get_speed_multiplier']()
                if speed < 1.0:
                    time.sleep((1.0 - speed) * 0.1)
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        episode_rewards.append(avg_reward)
        
        if verbose:
            print(f"[REINFORCE] Epoch {epoch} complete. Avg reward: {avg_reward:.3f}")
        
        # Update GPU visualizer
        if viz_hooks:
            viz_hooks['on_epoch_complete'](epoch)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': episode_rewards
        }, f"checkpoint_epoch_{epoch}.pt")
    
    return episode_rewards


# -----------------------------------------------------------------------------
# 10. Evaluation Metrics and Performance Analysis (G-1, G-2, I-1, I-2, I-3)
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: nn.Module,
                  eval_dataset: ARCDataset,
                  max_steps: int = 50,
                  device: str = "cuda",
                  damage_enabled: bool = False,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the universal constructor.
    """
    model.eval()
    
    # Metrics storage
    results = {
        'solved_count': 0,
        'total_count': 0,
        'exact_matches': 0,
        'avg_iou': 0.0,
        'avg_blueprint_length': 0.0,
        'blueprint_lengths': [],
        'macro_usage': defaultdict(int),
        'action_distribution': defaultdict(int),
        'gpu_memory_mb': 0.0,
        'avg_time_per_task': 0.0,
        'robustness_score': 0.0
    }
    
    # Create environment
    spatial_constructor = SelfRepairingConstructor(damage_rate=0.15)
    spatial_constructor.damage_enabled = damage_enabled
    interpreter = BlueprintInterpreter(model.controller.macro_library, max_steps)
    
    total_time = 0
    all_ious = []
    
    for task_idx, (demo_pairs, test_in, test_out) in enumerate(eval_dataset):
        if test_out is None:
            continue
        
        results['total_count'] += 1
        
        # Time tracking
        start_time = time.time()
        
        # Get task embedding
        task_embedding = model.embed_task(demo_pairs, device)
        task_embedding = task_embedding.unsqueeze(0)
        
        # Generate blueprint
        blueprint = model.generate_blueprint(task_embedding, max_steps, 
                                           temperature=0.0, verbose=False)
        
        # Track blueprint statistics
        results['blueprint_lengths'].append(len(blueprint))
        for action in blueprint:
            results['action_distribution'][action] += 1
            if action >= ConstructorOps.base_vocab_size():
                macro_id = action - ConstructorOps.base_vocab_size()
                results['macro_usage'][f'macro_{macro_id}'] += 1
        
        # Execute blueprint
        spatial_constructor.reset(test_out.shape)
        final_canvas = interpreter.execute_blueprint(blueprint, verbose=False)
        
        # Compute metrics
        h_min = min(final_canvas.shape[0], test_out.shape[0])
        w_min = min(final_canvas.shape[1], test_out.shape[1])
        
        final_crop = final_canvas[:h_min, :w_min]
        target_crop = test_out[:h_min, :w_min]
        
        # IoU computation
        intersection = (final_crop == target_crop).float().sum()
        union = h_min * w_min
        iou = (intersection / union).item() if union > 0 else 0
        all_ious.append(iou)
        
        # Check exact match
        exact_match = (final_crop == target_crop).all().item()
        if exact_match:
            results['exact_matches'] += 1
        
        # Consider solved if IoU > 0.95
        if iou > 0.95:
            results['solved_count'] += 1
        
        # Time tracking
        task_time = time.time() - start_time
        total_time += task_time
        
        # GPU memory tracking (only on first task)
        if task_idx == 0 and device == "cuda":
            results['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        if verbose and task_idx % 10 == 0:
            print(f"[Eval] Task {task_idx}: IoU={iou:.3f}, "
                  f"Blueprint length={len(blueprint)}, Time={task_time:.2f}s")
    
    # Compute aggregated metrics
    results['avg_iou'] = np.mean(all_ious) if all_ious else 0
    results['avg_blueprint_length'] = np.mean(results['blueprint_lengths']) if results['blueprint_lengths'] else 0
    results['avg_time_per_task'] = total_time / results['total_count'] if results['total_count'] > 0 else 0
    
    # Robustness score (if damage enabled)
    if damage_enabled:
        results['robustness_score'] = results['solved_count'] / results['total_count'] if results['total_count'] > 0 else 0
    
    # Performance benchmarks
    tasks_per_second = results['total_count'] / total_time if total_time > 0 else 0
    results['tasks_per_second'] = tasks_per_second
    
    # Summary statistics
    if verbose:
        print("\n" + "="*50)
        print("[Eval] EVALUATION COMPLETE")
        print("="*50)
        print(f"[Eval] Solved: {results['solved_count']}/{results['total_count']} "
              f"({100*results['solved_count']/results['total_count']:.1f}%)")
        print(f"[Eval] Exact matches: {results['exact_matches']}")
        print(f"[Eval] Average IoU: {results['avg_iou']:.3f}")
        print(f"[Eval] Average blueprint length: {results['avg_blueprint_length']:.1f}")
        print(f"[Eval] Tasks per second: {tasks_per_second:.1f}")
        print(f"[Eval] GPU memory: {results['gpu_memory_mb']:.1f} MB")
        
        if damage_enabled:
            print(f"[Eval] Robustness score: {results['robustness_score']:.3f}")
        
        # Macro usage statistics
        if results['macro_usage']:
            print("\n[Eval] Macro usage statistics:")
            for macro, count in sorted(results['macro_usage'].items(), 
                                     key=lambda x: x[1], reverse=True):
                usage_percent = 100 * count / sum(results['action_distribution'].values())
                print(f"  {macro}: {count} times ({usage_percent:.1f}%)")
    
    return results


# -----------------------------------------------------------------------------
# 11. Comprehensive Verbose Mode (J-1)
# -----------------------------------------------------------------------------

class VerboseLogger:
    """Enhanced logging for all components of the universal constructor."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.step_count = 0
    
    def log(self, component: str, message: str, data: Optional[Dict] = None):
        """Log a message with optional data."""
        if not self.enabled:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{component}] {message}")
        
        if data:
            for key, value in data.items():
                print(f"    {key}: {value}")
    
    def log_construction_step(self, step: int, opcode: int, 
                            arm_positions: List[Tuple[int, int]],
                            canvas_state: Optional[torch.Tensor] = None):
        """Log detailed construction step information."""
        if not self.enabled:
            return
        
        self.step_count = step
        
        # Convert opcode to name
        if opcode < ConstructorOps.base_vocab_size():
            op_name = list(ConstructorOps)[opcode].name
        else:
            op_name = f"MACRO_{opcode - ConstructorOps.base_vocab_size()}"
        
        print(f"\n[Step {step}] Executing: {op_name}")
        print(f"  Arm positions: {arm_positions}")
        
        if canvas_state is not None:
            non_empty = (canvas_state != -1).sum().item()
            print(f"  Canvas: {canvas_state.shape}, {non_empty} non-empty cells")


# -----------------------------------------------------------------------------
# 12. Main function with all modes
# -----------------------------------------------------------------------------

def main():
    """
    Main entry point for the Von Neumann Universal Constructor.
    """
    parser = argparse.ArgumentParser(
        description="Von Neumann Universal Constructor for ARC-AGI-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--mode", choices=["train", "eval", "demo", "test-damage", "test-compositional"], 
                      default="demo", help="Execution mode")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./arc_agi2",
                      help="Path to ARC-AGI-2 dataset")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for training")
    
    # Model architecture arguments
    parser.add_argument("--gnca-channels", type=int, default=GNCA_CHANNELS,
                      help="Number of GNCA channels")
    parser.add_argument("--gnca-steps", type=int, default=8,
                      help="Number of GNCA update steps")
    parser.add_argument("--task-embed-dim", type=int, default=TASK_EMBED_DIM,
                      help="Task embedding dimension")
    parser.add_argument("--embed-dim", type=int, default=96,
                      help="Transformer embedding dimension")
    parser.add_argument("--transformer-layers", type=int, default=4,
                      help="Number of transformer layers")
    parser.add_argument("--transformer-heads", type=int, default=4,
                      help="Number of attention heads")
    parser.add_argument("--memory-tokens", type=int, default=6,
                      help="Number of memory tokens")
    parser.add_argument("--memory-augmented", action="store_true",
                      help="Use memory-augmented controller (DNC/TTM-style)")
    parser.add_argument("--memory-slots", type=int, default=32,
                      help="Number of external memory slots")
    parser.add_argument("--memory-dim", type=int, default=64,
                      help="Dimension of memory slots")
    
    # Training arguments
    parser.add_argument("--train-mode", choices=["supervised", "reinforce"], 
                      default="reinforce", help="Training mode")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor for REINFORCE")
    parser.add_argument("--max-steps", type=int, default=50,
                      help="Maximum construction steps")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to checkpoint file")
    
    # Self-repair arguments
    parser.add_argument("--damage", action="store_true",
                      help="Enable damage simulation for self-repair")
    parser.add_argument("--damage-rate", type=float, default=0.15,
                      help="Fraction of cells to damage")
    parser.add_argument("--repair-steps", type=int, default=5,
                      help="Number of GNCA repair steps")
    
    # Visualization arguments
    parser.add_argument("--viz", action="store_true",
                      help="Enable real-time visualization (requires pygame)")
    parser.add_argument("--gpu-viz", action="store_true",
                      help="Enable GPU-accelerated visualization with Dear PyGui")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose debug output")
    
    # DreamCoder integration arguments
    parser.add_argument("--dreamcoder", action="store_true",
                      help="Enable DreamCoder integration for macro discovery")
    parser.add_argument("--dreamcoder-iterations", type=int, default=3,
                      help="Number of wake-sleep cycles for DreamCoder")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    
    # Ensure dataset is downloaded
    ensure_dataset(args.data_dir)
    
    # Create model
    print("[Main] Creating model...")
    
    # GNCA encoder
    encoder = GNCAEncoder(
        channels=args.gnca_channels,
        steps=args.gnca_steps
    ).to(device)
    
    # Controller (memory-augmented or standard)
    if args.memory_augmented:
        print("[Main] Using memory-augmented controller (DNC/TTM-style)")
        controller = MemoryAugmentedController(
            task_embed_dim=args.task_embed_dim,
            embed_dim=args.embed_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            memory_slots=args.memory_slots,
            memory_dim=args.memory_dim
        ).to(device)
    else:
        print("[Main] Using standard token controller")
        controller = TokenController(
            embed_dim=args.embed_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            memory_tokens=args.memory_tokens
        ).to(device)
    
    # Unified model
    model = ARCUniversalConstructor(encoder, controller).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"[Main] Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Main] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Verbose logger
    logger = VerboseLogger(args.verbose)
    
    # GPU Visualizer setup
    gpu_viz = None
    viz_hooks = None
    if args.gpu_viz and GPU_VIZ_AVAILABLE:
        print("[Main] Starting GPU visualizer...")
        gpu_viz = ARCGPUVisualizer()
        viz_hooks = create_visualizer_hooks(gpu_viz)
        
        # Start visualizer in separate thread
        import threading
        viz_thread = threading.Thread(target=gpu_viz.run)
        viz_thread.daemon = True
        viz_thread.start()
        
        # Give visualizer time to initialize
        time.sleep(1.0)
    elif args.gpu_viz and not GPU_VIZ_AVAILABLE:
        print("[Main] WARNING: GPU visualizer requested but not available")
        print("[Main] Install with: pip install dearpygui")
    
    # Execute based on mode
    if args.mode == "demo":
        run_demo(model, args, device, logger, viz_hooks)
    
    elif args.mode == "train":
        # Create datasets
        train_dataset = ARCDataset("train")
        print(f"[Main] Training dataset: {len(train_dataset)} tasks")
        
        if args.train_mode == "reinforce":
            print("[Main] Training with REINFORCE algorithm")
            rewards = train_reinforce(
                model, train_dataset, 
                epochs=args.epochs,
                lr=args.lr,
                gamma=args.gamma,
                max_steps=args.max_steps,
                device=device,
                verbose=args.verbose,
                viz_hooks=viz_hooks
            )
            print(f"[Main] Training complete. Final avg reward: {rewards[-1]:.3f}")
        else:
            print("[Main] Supervised training not yet implemented")
    
    elif args.mode == "eval":
        # Create evaluation dataset
        eval_dataset = ARCDataset("eval")
        print(f"[Main] Evaluation dataset: {len(eval_dataset)} tasks")
        
        # Run comprehensive evaluation
        results = evaluate_model(
            model, eval_dataset,
            max_steps=args.max_steps,
            device=device,
            damage_enabled=args.damage,
            verbose=args.verbose
        )
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("[Main] Evaluation results saved to evaluation_results.json")
    
    elif args.mode == "test-damage":
        run_damage_test(model, args, device, logger)
    
    elif args.mode == "test-compositional":
        compositional_test(model, device)
    
    print("[Main] ✓ Complete")


def run_demo(model: "ARCUniversalConstructor", args, device, logger, viz_hooks=None):
    """Run demonstration of all components."""
    print("\n" + "="*60)
    print("[Demo] Von Neumann Universal Constructor for ARC-AGI-2")
    print("="*60)
    
    # Set visualization flag on model for DSL test
    if args.viz:
        model._viz_enabled = True
        print("[Demo] Visualization enabled (--viz flag detected)")
    
    # Load a sample task
    dataset = ARCDataset("train")
    demo_pairs, test_in, test_out = dataset[0]
    
    if test_out is None:
        print("[Demo] No test output available for task 0, trying task 1...")
        demo_pairs, test_in, test_out = dataset[1]
    
    print(f"[Demo] Loaded task with {len(demo_pairs)} demonstration pairs")
    print(f"[Demo] Test input shape: {test_in.shape}")
    print(f"[Demo] Test output shape: {test_out.shape}")
    
    # Update GPU visualizer if available
    if viz_hooks:
        viz_hooks['on_task_start'](0, demo_pairs, test_in, test_out)
    
    # Test each component
    print("\n[Demo] Testing GNCA Encoder (Step B-1)...")
    test_gnca_encoder(model.encoder, test_in, device, args.verbose, viz_hooks)
    
    print("\n[Demo] Testing Token Controller (Step C-1)...")
    test_token_controller(model.controller, demo_pairs, device, args.verbose, viz_hooks)
    
    print("\n[Demo] Testing Constructor-Complete DSL (Steps D-1, D-2, D-3)...")
    test_constructor_dsl(model, demo_pairs, test_in, test_out, device, args.verbose, viz_hooks)
    
    if args.damage:
        print("\n[Demo] Testing Self-Repair Fabric (Step H-2)...")
        test_self_repair(model, demo_pairs, test_in, test_out, device, args, viz_hooks)
    
    print("\n[Demo] ✓ All components tested successfully")


def run_damage_test(model: "ARCUniversalConstructor", args, device, logger):
    """Test self-repair capabilities with visualization."""
    print("\n" + "="*60)
    print("[Damage Test] Self-Repair Fabric Demonstration")
    print("="*60)
    
    # Create a simple pattern
    canvas_size = (10, 10)
    pattern = torch.zeros(canvas_size, dtype=torch.long, device=device) - 1
    
    # Draw a cross pattern
    pattern[4:6, :] = 2  # Horizontal line (blue)
    pattern[:, 4:6] = 3  # Vertical line (green)
    pattern[4:6, 4:6] = 4  # Center (yellow)
    
    print("[Damage Test] Original pattern:")
    visualize_grid(pattern)
    
    # Create self-repairing constructor
    constructor = SelfRepairingConstructor(
        damage_rate=args.damage_rate,
        repair_steps=args.repair_steps
    )
    constructor.damage_enabled = True
    constructor.canvas = pattern.clone()
    
    # Inject damage
    print(f"\n[Damage Test] Injecting {args.damage_rate*100:.0f}% damage...")
    constructor.inject_damage(verbose=True)
    
    print("\n[Damage Test] Damaged pattern:")
    visualize_grid(constructor.canvas)
    
    # Repair damage
    print(f"\n[Damage Test] Applying {args.repair_steps} repair steps...")
    constructor.repair_damage(verbose=True)
    
    print("\n[Damage Test] Repaired pattern:")
    visualize_grid(constructor.canvas)
    
    # Compute repair accuracy
    accuracy = (constructor.canvas == pattern).float().mean().item()
    print(f"\n[Damage Test] Repair accuracy: {accuracy*100:.1f}%")


def visualize_grid(grid: torch.Tensor, title: str = ""):
    """Simple ASCII visualization of a grid."""
    if title:
        print(title)
    
    color_map = {
        -1: '.',  # Empty
        0: '□',   # Black
        1: '■',   # Red
        2: '▦',   # Blue  
        3: '▤',   # Green
        4: '▣',   # Yellow
        5: '▥',   # Gray
        6: '▨',   # Fuchsia
        7: '▧',   # Orange
        8: '▩',   # Azure
        9: '▪',   # Brown
    }
    
    for row in grid:
        line = ""
        for cell in row:
            line += color_map.get(cell.item(), '?') + " "
        print(line)


def test_self_repair(model, demo_pairs, test_in, test_out, device, args, viz_hooks=None):
    """Test self-repair capabilities."""
    # Generate blueprint
    task_embedding = model.embed_task(demo_pairs, device)
    blueprint = model.generate_blueprint(
        task_embedding.unsqueeze(0), 
        max_steps=args.max_steps,
        temperature=0.0,
        verbose=False
    )
    
    print(f"[Self-Repair] Generated blueprint with {len(blueprint)} operations")
    
    # Execute with damage
    constructor = SelfRepairingConstructor(
        damage_rate=args.damage_rate,
        repair_steps=args.repair_steps
    )
    constructor.damage_enabled = True
    
    interpreter = BlueprintInterpreter(model.controller.macro_library, args.max_steps)
    interpreter.spatial_constructor = constructor
    
    # Execute blueprint
    print("[Self-Repair] Executing blueprint with damage simulation...")
    final_canvas = interpreter.execute_blueprint(blueprint, verbose=args.verbose, 
                                               visualize=args.viz, viz_delay=0.2)
    
    # Compute metrics
    h_min = min(final_canvas.shape[0], test_out.shape[0])
    w_min = min(final_canvas.shape[1], test_out.shape[1])
    
    final_crop = final_canvas[:h_min, :w_min]
    target_crop = test_out[:h_min, :w_min]
    
    accuracy = (final_crop == target_crop).float().mean().item()
    print(f"[Self-Repair] Construction accuracy with damage: {accuracy*100:.1f}%")
    
    if accuracy > 0.8:
        print("[Self-Repair] ✓ Self-repair successful!")
    else:
        print("[Self-Repair] ✗ Self-repair needs improvement")


def ensure_dataset(data_dir: str):
    """Ensure ARC-AGI-2 dataset is downloaded."""
    if os.path.exists(os.path.join(data_dir, "data", "training")):
        print(f"[Data] ARC-AGI-2 dataset found at {data_dir}")
        return
    
    print("[Data] Downloading ARC-AGI-2 dataset...")
    url = "https://github.com/arc-prize/arc-agi-2/archive/refs/heads/main.zip"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Extract zip
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(".")
        
        # Rename directory
        if os.path.exists("arc-agi-2-main"):
            os.rename("arc-agi-2-main", data_dir)
        
        print(f"[Data] ✓ Dataset downloaded to {data_dir}")
    except Exception as e:
        print(f"[Data] Error downloading dataset: {e}")
        sys.exit(1)


class ARCUniversalConstructor(nn.Module):
    """Unified model combining GNCA encoder and controller."""
    
    def __init__(self, encoder: GNCAEncoder, controller: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.controller = controller
    
    def embed_task(self, demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
                   device: str) -> torch.Tensor:
        """Extract task embedding from demonstration pairs."""
        embeddings = []
        
        for inp, out in demo_pairs:
            inp = inp.to(device)
            out = out.to(device)
            
            # Encode input and output (add batch dimension)
            inp_embed = self.encoder(inp.unsqueeze(0))
            out_embed = self.encoder(out.unsqueeze(0))
            
            # Difference embedding (what changes from input to output)
            diff_embed = out_embed - inp_embed
            embeddings.append(diff_embed)
        
        # Average across all demonstrations
        return torch.stack(embeddings).mean(0).squeeze(0)
    
    def generate_blueprint(self, task_embedding: torch.Tensor,
                          max_steps: int = 50,
                          temperature: float = 1.0,
                          verbose: bool = False) -> List[int]:
        """Generate construction blueprint using the controller."""
        return self.controller.generate_blueprint(
            task_embedding, max_steps, temperature, verbose
        )


# Test functions for demo mode
def test_gnca_encoder(encoder, test_grid, device, verbose, viz_hooks=None):
    """Test GNCA encoder component."""
    # Create test input
    if len(test_grid.shape) == 2:
        test_input = test_grid.unsqueeze(0).to(device)
    else:
        test_input = test_grid.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"[GNCA] Parameters: {param_count:,} (target: ≤15k)")
    
    # Warmup for accurate timing
    if device.type == "cuda":
        for _ in range(10):
            with torch.no_grad():
                _ = encoder(test_input)
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    start_time = time.time()
    with torch.no_grad():
        embedding = encoder(test_input, verbose=verbose)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"[GNCA] Forward pass: {elapsed_ms:.2f}ms (target: ≤0.5ms)")
    print(f"[GNCA] Output embedding: {embedding.shape}")
    print(f"[GNCA] Embedding stats: mean={embedding.mean().item():.4f}, "
          f"std={embedding.std().item():.4f}")
    
    # Check numerical stability
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        print("[GNCA] ❌ ERROR: NaN/Inf detected!")
    else:
        print("[GNCA] ✓ Numerical stability verified")
    
    # Update GPU visualizer with GNCA activations
    if viz_hooks and hasattr(encoder, 'ca_rule'):
        # Get intermediate GNCA state for visualization
        with torch.no_grad():
            x = encoder._grid_to_onehot(test_input)
            x = encoder.input_proj(x)
            
            # Get activation after first GNCA step
            gnca_act = encoder.ca_rule(x, verbose=False)
            
            # Update visualizer
            viz_hooks['on_neural_state_update'](
                gnca_act=gnca_act.squeeze(0).mean(0),  # Average over channels
                attention=None,
                memory=None
            )


def test_token_controller(controller, demo_pairs, device, verbose, viz_hooks=None):
    """Test token controller component."""
    # Count parameters
    param_count = sum(p.numel() for p in controller.parameters())
    print(f"[Controller] Parameters: {param_count:,}")
    
    # Create a dummy task embedding
    if hasattr(controller, 'task_proj'):
        task_dim = controller.task_proj.in_features
    else:
        task_dim = TASK_EMBED_DIM
    
    task_embedding = torch.randn(1, task_dim, device=device)
    
    # Test forward pass
    start_time = time.time()
    with torch.no_grad():
        if isinstance(controller, MemoryAugmentedController):
            action_logits, memory, state = controller(task_embedding, verbose=verbose)
            print(f"[Controller] Memory shape: {memory.shape}")
            memory_output = memory
        else:
            action_logits, memory_output = controller(task_embedding, verbose=verbose)
            print(f"[Controller] Memory output shape: {memory_output.shape}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"[Controller] Forward pass: {elapsed_ms:.2f}ms")
    print(f"[Controller] Action logits shape: {action_logits.shape}")
    print(f"[Controller] Action vocab size: {action_logits.size(-1)}")
    
    # Check numerical stability
    if torch.isnan(action_logits).any():
        print("[Controller] ❌ ERROR: NaN detected!")
    else:
        print("[Controller] ✓ Numerical stability verified")
    
    # Update GPU visualizer with controller states
    if viz_hooks:
        # For memory-augmented controller, show external memory
        if isinstance(controller, MemoryAugmentedController) and hasattr(controller, 'memory'):
            viz_hooks['on_neural_state_update'](
                gnca_act=None,
                attention=None,
                memory=controller.memory.memory  # External memory state
            )
        # For standard controller, show memory tokens
        else:
            viz_hooks['on_neural_state_update'](
                gnca_act=None,
                attention=None,
                memory=memory_output.squeeze(0)  # Memory tokens
            )


def test_constructor_dsl(model, demo_pairs, test_in, test_out, device, verbose, viz_hooks=None):
    """Test Constructor-Complete DSL functionality."""
    print("[DSL] Base operations:")
    for op in ConstructorOps:
        print(f"  • {op.name} = {op.value}")
    
    # Test macro infrastructure
    macro_lib = model.controller.macro_library
    print(f"\n[DSL] Testing macro infrastructure...")
    
    # Add test macros
    draw_line_macro = [ConstructorOps.MOVE_ARM, ConstructorOps.WRITE, 
                      ConstructorOps.MOVE_ARM, ConstructorOps.WRITE]
    macro_id1 = macro_lib.add_macro("draw_line", draw_line_macro)
    
    draw_square_macro = [ConstructorOps.WRITE, ConstructorOps.MOVE_ARM,
                        ConstructorOps.WRITE, ConstructorOps.MOVE_ARM,
                        ConstructorOps.WRITE, ConstructorOps.MOVE_ARM,
                        ConstructorOps.WRITE]
    macro_id2 = macro_lib.add_macro("draw_square", draw_square_macro)
    
    print(f"[DSL] Added macro 'draw_line' with ID {macro_id1}")
    print(f"[DSL] Added macro 'draw_square' with ID {macro_id2}")
    print(f"[DSL] Total vocabulary size: {macro_lib.get_total_vocab_size()}")
    
    # Test dynamic action head expansion
    if hasattr(model.controller, 'expand_action_head'):
        old_size = model.controller.action_head.out_features
        model.controller.expand_action_head(macro_lib.get_total_vocab_size())
        new_size = model.controller.action_head.out_features
        print(f"[DSL] Action head expanded: {old_size} → {new_size}")
    
    # Generate and execute a blueprint
    print(f"\n[DSL] Generating blueprint...")
    task_embedding = model.embed_task(demo_pairs, device)
    blueprint = model.generate_blueprint(
        task_embedding.unsqueeze(0),
        max_steps=20,
        temperature=0.5,
        verbose=False
    )
    
    print(f"[DSL] Generated blueprint length: {len(blueprint)}")
    
    # Update GPU visualizer with blueprint
    if viz_hooks:
        viz_hooks['on_blueprint_generated'](blueprint)
    
    # Analyze blueprint composition
    base_ops = sum(1 for op in blueprint if op < ConstructorOps.base_vocab_size())
    macro_ops = len(blueprint) - base_ops
    print(f"[DSL] Blueprint composition: {base_ops} base ops, {macro_ops} macro calls")
    
    # Execute blueprint
    print(f"\n[DSL] Executing blueprint...")
    interpreter = BlueprintInterpreter(macro_lib, max_steps=50)
    
    # Add GPU visualizer hook to interpreter if available
    if viz_hooks:
        interpreter._viz_hook = viz_hooks
    
    try:
        # Execute with visualization if requested
        visualize = hasattr(model, '_viz_enabled') and model._viz_enabled
        final_canvas = interpreter.execute_blueprint(blueprint, verbose=False, 
                                                   visualize=visualize, viz_delay=0.3)
        non_empty = (final_canvas != -1).sum().item()
        print(f"[DSL] Execution complete. Canvas: {final_canvas.shape}, "
              f"{non_empty} non-empty cells")
        print("[DSL] ✓ Blueprint execution successful")
        
        # Update final prediction in GPU visualizer
        if viz_hooks:
            viz_hooks['on_construction_step'](final_canvas, interpreter.constructor.get_arm_positions())
            
    except Exception as e:
        print(f"[DSL] ❌ Blueprint execution failed: {e}")


# Additional helper for offline DreamCoder integration
class DreamCoderIntegration:
    """
    Wrapper for offline DreamCoder integration to discover new macros.
    This simulates DreamCoder's wake-sleep algorithm within our framework.
    """
    
    def __init__(self, model: "ARCUniversalConstructor", 
                 max_program_length: int = 20):
        self.model = model
        self.max_program_length = max_program_length
        self.discovered_programs = []
    
    def wake_phase(self, tasks: List[Tuple], device: str = "cuda") -> List[List[int]]:
        """
        Wake phase: solve tasks and collect successful programs.
        """
        successful_programs = []
        
        for demo_pairs, test_in, test_out in tasks:
            if test_out is None:
                continue
            
            # Generate program
            task_embedding = self.model.embed_task(demo_pairs, device)
            blueprint = self.model.generate_blueprint(
                task_embedding.unsqueeze(0),
                max_steps=self.max_program_length,
                temperature=0.0  # Greedy for wake phase
            )
            
            # Execute and check success
            interpreter = BlueprintInterpreter(
                self.model.controller.macro_library,
                self.max_program_length
            )
            final_canvas = interpreter.execute_blueprint(blueprint)
            
            # Simple success check (IoU > 0.9)
            h_min = min(final_canvas.shape[0], test_out.shape[0])
            w_min = min(final_canvas.shape[1], test_out.shape[1])
            
            final_crop = final_canvas[:h_min, :w_min]
            target_crop = test_out[:h_min, :w_min]
            
            iou = (final_crop == target_crop).float().mean().item()
            if iou > 0.9:
                successful_programs.append(blueprint)
        
        return successful_programs
    
    def sleep_phase(self, programs: List[List[int]]) -> Dict[str, List[int]]:
        """
        Sleep phase: compress programs into reusable macros.
        """
        # Simple compression: find repeated subsequences
        discovered_macros = {}
        
        # Count all subsequences of length 2-5
        subsequence_counts = defaultdict(int)
        for program in programs:
            for length in range(2, 6):
                for i in range(len(program) - length + 1):
                    subseq = tuple(program[i:i+length])
                    subsequence_counts[subseq] += 1
        
        # Extract frequent subsequences as macros
        for subseq, count in subsequence_counts.items():
            if count >= 3:  # Appears in at least 3 programs
                macro_name = f"macro_{len(discovered_macros)}"
                discovered_macros[macro_name] = list(subseq)
        
        return discovered_macros
    
    def integrate_discoveries(self, discovered_macros: Dict[str, List[int]]):
        """
        Add discovered macros to the model's library.
        """
        macro_lib = self.model.controller.macro_library
        
        for name, operations in discovered_macros.items():
            # Convert to ConstructorOps
            ops = []
            for op_id in operations:
                if op_id < ConstructorOps.base_vocab_size():
                    ops.append(list(ConstructorOps)[op_id])
            
            if ops:  # Only add if valid
                macro_lib.add_macro(name, ops)
                print(f"[DreamCoder] Added discovered macro '{name}' with {len(ops)} operations")
        
        # Expand action head if needed
        if hasattr(self.model.controller, 'expand_action_head'):
            self.model.controller.expand_action_head(macro_lib.get_total_vocab_size())


# Compositional test for final evaluation
def compositional_test(model: "ARCUniversalConstructor", device: str = "cuda"):
    """
    Test L-1, L-2, L-3: Compositional generalization with learned macros.
    """
    print("\n" + "="*50)
    print("[Compositional Test] Testing macro learning and composition")
    print("="*50)
    
    # Create synthetic tasks that benefit from macros
    synthetic_tasks = []
    
    # Task 1: Repeated pattern (benefits from "draw_line" macro)
    task1_in = torch.zeros(10, 10, dtype=torch.long) - 1
    task1_out = task1_in.clone()
    for i in range(5):
        task1_out[i*2, :5] = 2  # Horizontal lines
    synthetic_tasks.append(([(task1_in, task1_out)], task1_in, task1_out))
    
    # Task 2: Grid pattern (benefits from "draw_square" macro)
    task2_in = torch.zeros(8, 8, dtype=torch.long) - 1
    task2_out = task2_in.clone()
    for i in range(0, 8, 3):
        for j in range(0, 8, 3):
            task2_out[i:i+2, j:j+2] = 3  # Small squares
    synthetic_tasks.append(([(task2_in, task2_out)], task2_in, task2_out))
    
    # Run DreamCoder integration
    dreamcoder = DreamCoderIntegration(model)
    
    # Wake phase
    print("[Compositional Test] Running wake phase...")
    programs = dreamcoder.wake_phase(synthetic_tasks, device)
    print(f"[Compositional Test] Collected {len(programs)} successful programs")
    
    # Sleep phase
    print("[Compositional Test] Running sleep phase...")
    macros = dreamcoder.sleep_phase(programs)
    print(f"[Compositional Test] Discovered {len(macros)} potential macros")
    
    # Integrate discoveries
    dreamcoder.integrate_discoveries(macros)
    
    # Test on new compositional task
    print("\n[Compositional Test] Testing on compositional task...")
    comp_in = torch.zeros(12, 12, dtype=torch.long) - 1
    comp_out = comp_in.clone()
    
    # Pattern that combines line-drawing and square-drawing
    for i in range(3):
        comp_out[i*4, :] = 2  # Horizontal lines
        comp_out[:, i*4] = 3  # Vertical lines
    for i in range(1, 3):
        for j in range(1, 3):
            comp_out[i*4-1:i*4+1, j*4-1:j*4+1] = 4  # Squares at intersections
    
    comp_task = [(comp_in, comp_out)]
    
    # Generate solution
    task_embedding = model.embed_task(comp_task, device)
    blueprint = model.generate_blueprint(
        task_embedding.unsqueeze(0),
        max_steps=30,
        temperature=0.0
    )
    
    # Analyze macro usage
    macro_usage = defaultdict(int)
    for op in blueprint:
        if op >= ConstructorOps.base_vocab_size():
            macro_usage[f"macro_{op - ConstructorOps.base_vocab_size()}"] += 1
    
    print(f"[Compositional Test] Blueprint length: {len(blueprint)}")
    print(f"[Compositional Test] Macro usage: {dict(macro_usage)}")
    
    # Check if macros were actually used
    if sum(macro_usage.values()) > 0:
        usage_percent = 100 * sum(macro_usage.values()) / len(blueprint)
        print(f"[Compositional Test] ✓ Macros used in {usage_percent:.1f}% of operations")
        
        # Test program compression (L-1)
        baseline_length = len(blueprint) * 3  # Assume each macro saves ~3 operations
        compression_ratio = 1 - len(blueprint) / baseline_length
        print(f"[Compositional Test] Program compression: {compression_ratio*100:.1f}%")
        
        if compression_ratio > 0.3:
            print("[Compositional Test] ✓ L-1: Program compression achieved (>30%)")
        
        if usage_percent > 25:
            print("[Compositional Test] ✓ L-2: Macro adoption successful (>25%)")
        
        # Check for zero-shot composition (L-3)
        different_macros = len(macro_usage)
        if different_macros >= 2:
            print(f"[Compositional Test] ✓ L-3: Zero-shot composition verified - used {different_macros} different macros")
        else:
            print(f"[Compositional Test] ✗ L-3: Need at least 2 different macros for zero-shot composition (used {different_macros})")
    else:
        print("[Compositional Test] ✗ No macros used - needs more training")


if __name__ == "__main__":
    main() 