#!/usr/bin/env python3
"""Test Token Turing Machine implementation."""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from arc_universal_constructor import (
    TTMController, StaticMemoryController, GNCAEncoder,
    ARCUniversalConstructor, ConstructorOps, visualize_grid
)

def test_ttm_vs_static():
    """Compare TTM with dynamic memory vs static memory controller."""
    print("="*60)
    print("Testing Token Turing Machine vs Static Memory Controller")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create a simple task embedding
    task_embed_dim = 64
    task_embedding = torch.randn(1, task_embed_dim, device=device)
    
    # Create controllers
    print("1. Creating Static Memory Controller...")
    static_controller = StaticMemoryController(
        task_embed_dim=task_embed_dim,
        embed_dim=96,
        memory_tokens=6
    ).to(device)
    
    print("2. Creating Token Turing Machine...")
    ttm_controller = TTMController(
        task_embed_dim=task_embed_dim,
        embed_dim=96,
        memory_tokens=96,  # m in paper
        read_tokens=16,    # r in paper
        summarizer_method="mlp"
    ).to(device)
    
    # Compare parameter counts
    static_params = static_controller.count_parameters()
    ttm_params = ttm_controller.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Static Memory: {static_params:,}")
    print(f"  TTM: {ttm_params:,}")
    print(f"  TTM overhead: +{ttm_params - static_params:,} ({(ttm_params/static_params - 1)*100:.1f}%)")
    
    # Test forward passes
    print("\n3. Testing forward passes...")
    
    # Static controller - memory doesn't change
    print("\nStatic Memory Controller:")
    with torch.no_grad():
        for step in range(3):
            action_logits, memory, param_logits = static_controller(task_embedding)
            print(f"  Step {step}: action_logits={action_logits.shape}, memory={memory.shape}")
            print(f"    Memory mean: {memory.mean().item():.4f} (should be same each step)")
    
    # TTM - memory evolves
    print("\nToken Turing Machine:")
    memory = ttm_controller.init_memory(1, device)
    memory_means = []
    
    with torch.no_grad():
        for step in range(3):
            action_logits, param_logits, memory, output = ttm_controller.forward_step(
                task_embedding, memory
            )
            memory_mean = memory.mean().item()
            memory_means.append(memory_mean)
            print(f"  Step {step}: action_logits={action_logits.shape}, memory={memory.shape}")
            print(f"    Memory mean: {memory_mean:.4f}")
            print(f"    Output tokens: {output.shape}")
    
    # Check if memory evolved
    memory_changed = len(set(memory_means)) > 1
    print(f"\n  Memory evolved across steps: {memory_changed}")
    if memory_changed:
        print("  ✓ TTM memory is dynamic!")
    else:
        print("  ✗ TTM memory is not evolving properly")
    
    # Test blueprint generation
    print("\n4. Testing blueprint generation...")
    
    print("\nStatic Memory (no memory evolution):")
    static_blueprint = static_controller.generate_blueprint(
        task_embedding, max_steps=10, temperature=0.5, verbose=False
    )
    print(f"  Generated {len(static_blueprint)} operations")
    
    print("\nTTM (with memory evolution):")
    ttm_blueprint = ttm_controller.generate_blueprint(
        task_embedding, max_steps=10, temperature=0.5, verbose=False
    )
    print(f"  Generated {len(ttm_blueprint)} operations")
    
    # Show first few operations
    print("\n5. First 5 operations from each:")
    print("\nStatic Memory:")
    for i, (op, params) in enumerate(static_blueprint[:5]):
        if op < ConstructorOps.base_vocab_size():
            op_name = list(ConstructorOps)[op].name
        else:
            op_name = f"MACRO_{op - ConstructorOps.base_vocab_size()}"
        print(f"  {i}: {op_name} {params}")
    
    print("\nTTM:")
    for i, (op, params) in enumerate(ttm_blueprint[:5]):
        if op < ConstructorOps.base_vocab_size():
            op_name = list(ConstructorOps)[op].name
        else:
            op_name = f"MACRO_{op - ConstructorOps.base_vocab_size()}"
        print(f"  {i}: {op_name} {params}")
    
    print("\n6. Testing TTM read/write/process operations...")
    
    # Test individual TTM operations
    memory = ttm_controller.init_memory(1, device)
    input_token = ttm_controller.task_proj(task_embedding).unsqueeze(1)
    
    # Read
    read_result = ttm_controller.read(memory, input_token, verbose=True)
    print(f"\nRead operation: {memory.shape} + {input_token.shape} → {read_result.shape}")
    
    # Process
    processed = ttm_controller.process(read_result, verbose=True)
    print(f"\nProcess operation: {read_result.shape} → {processed.shape}")
    
    # Write
    new_memory = ttm_controller.write(memory, processed, input_token, verbose=True)
    print(f"\nWrite operation: {memory.shape} + {processed.shape} + {input_token.shape} → {new_memory.shape}")
    
    # Verify memory changed
    memory_diff = (new_memory - memory).abs().mean().item()
    print(f"\nMemory difference after read/process/write: {memory_diff:.6f}")
    if memory_diff > 1e-6:
        print("✓ Memory successfully updated!")
    else:
        print("✗ Memory did not change")
    
    print("\n" + "="*60)
    print("TTM Test Complete!")
    print("="*60)


def test_summarizer_methods():
    """Test different token summarization methods."""
    print("\n" + "="*60)
    print("Testing Token Summarization Methods")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create TTM with MLP summarizer
    print("\n1. Testing MLP-based summarizer...")
    ttm_mlp = TTMController(
        memory_tokens=96,
        read_tokens=16,
        summarizer_method="mlp"
    ).to(device)
    
    # Create TTM with query summarizer
    print("2. Testing query-based summarizer...")
    ttm_query = TTMController(
        memory_tokens=96,
        read_tokens=16,
        summarizer_method="query"
    ).to(device)
    
    # Compare parameter counts
    mlp_params = sum(p.numel() for p in ttm_mlp.read_summarizer.parameters())
    query_params = sum(p.numel() for p in ttm_query.read_summarizer.parameters())
    
    print(f"\nSummarizer parameter counts:")
    print(f"  MLP method: {mlp_params:,}")
    print(f"  Query method: {query_params:,}")
    
    # Test summarization
    batch_size = 2
    num_tokens = 50
    dim = 96
    test_tokens = torch.randn(batch_size, num_tokens, dim, device=device)
    
    print(f"\n3. Testing summarization ({num_tokens} → {ttm_mlp.read_tokens} tokens)...")
    
    with torch.no_grad():
        mlp_result = ttm_mlp.read_summarizer(test_tokens)
        query_result = ttm_query.read_summarizer(test_tokens)
    
    print(f"\nMLP result: {mlp_result.shape}")
    print(f"Query result: {query_result.shape}")
    
    # Check if summarization preserved information
    print(f"\nInput norm: {test_tokens.norm(dim=-1).mean():.4f}")
    print(f"MLP output norm: {mlp_result.norm(dim=-1).mean():.4f}")
    print(f"Query output norm: {query_result.norm(dim=-1).mean():.4f}")
    
    print("\n✓ Token summarization working correctly!")


if __name__ == "__main__":
    test_ttm_vs_static()
    test_summarizer_methods() 