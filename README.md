# ARC Universal Constructor

A Von Neumann Universal Constructor implementation for solving ARC-AGI-2 tasks through explicit program synthesis and spatial construction.

## Overview

This project implements a learnable Universal Constructor that separates the "brain" (neural controller) from the "body" (spatial environment). Instead of directly predicting output grids, the system learns to write construction programs that are then executed step-by-step on a spatial canvas.

### Key Features

- **GNCA Encoder**: Perception module using Growing Neural Cellular Automata to extract task patterns
- **Program Synthesis**: Neural controller generates explicit construction blueprints
- **Spatial Execution**: Non-differentiable interpreter executes programs with construction arms
- **Macro Learning**: DreamCoder-inspired discovery of reusable subroutines
- **Self-Repair**: GNCA-based damage recovery for robust constructions
- **Memory-Augmented Control**: Optional DNC/TTM-style external memory for complex reasoning
- **GPU Visualization**: Real-time training visualization with Dear PyGui

## Architecture

```
ARC Task → GNCA Encoder → Task Embedding
                              ↓
                      Token Controller → Blueprint (Program)
                              ↓
                    Blueprint Interpreter
                              ↓
                     Spatial Constructor → Output Grid
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arc-universal-constructor.git
cd arc-universal-constructor

# Install dependencies
pip install torch numpy dearpygui requests

# Optional: Install flash-attn for faster attention
pip install flash-attn
```

## Usage

### Demo Mode
```bash
# Run demo with all components
python arc_universal_constructor.py --mode demo --verbose

# With GPU visualization
python arc_universal_constructor.py --mode demo --gpu-viz

# With self-repair demonstration
python arc_universal_constructor.py --mode demo --damage --verbose
```

### Training
```bash
# Train with REINFORCE
python arc_universal_constructor.py --mode train --epochs 10 --lr 1e-4

# Train with memory-augmented controller
python arc_universal_constructor.py --mode train --memory-augmented --memory-slots 32

# Resume from checkpoint
python arc_universal_constructor.py --mode train --checkpoint checkpoint_epoch_5.pt
```

### Evaluation
```bash
# Evaluate on test set
python arc_universal_constructor.py --mode eval --checkpoint model.pt

# Evaluate with damage/self-repair
python arc_universal_constructor.py --mode eval --checkpoint model.pt --damage
```

## Key Components

### 1. GNCA Encoder (`GNCAEncoder`)
- 3×3 depth-wise convolution for spatial perception
- 64-unit MLP update rule with residual connections
- 8 iteration steps by default
- Only 3,008 parameters (highly efficient)

### 2. Token Controller (`TokenController`)
- 4-layer Transformer with Flash-Attention support
- Learnable memory tokens as scratchpad
- Generates action sequences (blueprints)
- Dynamic vocabulary for macro support

### 3. Memory-Augmented Controller (`MemoryAugmentedController`)
- External memory with 32 slots
- Content-based addressing (cosine similarity)
- Read/write operations with usage tracking
- Inspired by DNC/TTM architectures

### 4. Constructor DSL
Base operations:
- `MOVE_ARM`: Move construction arm
- `WRITE`: Write color at current position
- `ERASE`: Clear current cell
- `BRANCH_IF_EMPTY`: Conditional logic
- `FORK_ARM`: Create new construction arm
- `HALT`: End program execution

### 5. Spatial Constructor
- Manages grid canvas and construction arms
- Non-differentiable execution environment
- Supports multiple arms for parallel construction
- Self-repair capabilities with GNCA fabric

## Training Details

The system uses REINFORCE algorithm with composite rewards:
- **Activity reward** (0.1): Encourages writing something
- **Color matching** (0.2): Partial credit for correct colors
- **IoU reward** (0.5): Main objective
- **Exact match bonus** (0.2): Perfect solution bonus

## Performance

- **Model size**: 309,152 parameters (< 700k target)
- **GNCA inference**: 0.33ms on RTX 4080
- **GPU memory**: < 1GB with batch size 1
- **Training speed**: Configurable, tracks tasks/second

## Visualization

The GPU visualizer (`--gpu-viz`) provides:
- Real-time grid visualization (input/output/prediction/construction)
- Neural state heatmaps (GNCA activations, attention, memory)
- Training metrics plots (loss, reward, IoU)
- Blueprint execution animation
- Interactive controls (pause, speed adjustment)

## Advanced Features

### Macro Discovery
The system can discover and learn reusable patterns:
```python
# Compositional test demonstrates macro learning
python arc_universal_constructor.py --mode test-compositional
```

### Self-Repair
Damage injection and repair demonstration:
```python
# Test self-repair capabilities
python arc_universal_constructor.py --mode test-damage --damage-rate 0.15
```

## Citation

If you use this code in your research, please cite:
```bibtex
@software{arc_universal_constructor,
  title={ARC Universal Constructor: Von Neumann Architecture for ARC-AGI-2},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/arc-universal-constructor}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by Von Neumann's Universal Constructor theory
- Uses techniques from DNC, TTM, and DreamCoder
- Built for the ARC-AGI-2 challenge 