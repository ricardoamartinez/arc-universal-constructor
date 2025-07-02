# ARC Universal Constructor

Implementation of a Von Neumann-style universal constructor for ARC-AGI-2 tasks. The system generates explicit construction programs (sequences of opcodes) that are executed step-by-step on a spatial grid.

## What This Actually Is

This is NOT a standard neural network that directly predicts output grids. Instead:

1. A neural controller generates a program (list of action IDs)
2. A non-differentiable interpreter executes these actions on a grid
3. Training uses REINFORCE because the execution is non-differentiable

Think of it as teaching a neural network to write assembly code for a simple spatial computer.

## Concrete Example

Given an ARC task, here's what actually happens:

```
Input: 3x3 grid with a red square in corner
Output: 3x3 grid with blue squares in all corners

What the system generates:
[(2, {'color': 1}), (1, {'dx': 6, 'dy': 5}), (2, {'color': 1}), ...]

Which translates to:
[WRITE(color=1), MOVE_ARM(dx=1, dy=0), WRITE(color=1), 
 MOVE_ARM(dx=0, dy=1), WRITE(color=1), MOVE_ARM(dx=-1, dy=0), 
 WRITE(color=1), HALT]

What actually happens:
Step 0: Write blue at (0,0)
Step 1: Move to (1,0)
Step 2: Write blue at (1,0)
... etc
```

The controller learns to output these action sequences through trial and error, not by copying examples.

## Technical Architecture

```
Input: ARC task examples
  ↓
GNCA Encoder (3k params)
  ↓
Task embedding (64-dim vector)
  ↓
Token-based Memory Controller (TTM-style, 306k params)
  ↓
Action sequence [WRITE, MOVE_ARM, WRITE, ..., HALT]
  ↓
Spatial Interpreter (non-differentiable)
  ↓
Output: Constructed grid
```

### Core Components

**GNCA Encoder**: 
- 3×3 depthwise conv + 64-unit MLP, iterated 8 times
- Converts grid patterns to 64-dim vectors
- 3,008 parameters total

**Controller Options**:

1. **Static Memory Controller** (default):
   - 4-layer transformer (96-dim, 4 heads) 
   - 6 learnable memory tokens (fixed, don't change during inference)
   - Acts as policy network π(action|state) generating construction programs
   - 306,144 parameters

2. **Token Turing Machine Controller** (`--ttm`):
   - True TTM implementation with dynamic memory evolution
   - Read: Z_t = Sr([M_t || I_t]) - combines memory and input
   - Process: O_t = Process(Z_t) - transforms via transformer
   - Write: M_{t+1} = Sw([M_t || O_t || I_t]) - updates memory
   - Memory evolves across blueprint generation steps
   - 854,919 parameters (169% of static controller)

3. **Memory-Augmented Controller** (`--memory-augmented`):
   - Enhanced version with 32-slot external memory (content-based addressing)
   - Combines transformer architecture with DNC-style external memory
   - Read/write operations via attention mechanisms

**Action Vocabulary** (Minimal Turing-Complete DSL):
Following Von Neumann's principle of minimal fixed machinery:

Spatial Operations:
- `MOVE_ARM`: Move by learned parameters dx, dy ∈ {-5..+5}
- `WRITE`: Write learned color parameter ∈ {0..9}
- `READ`: Read cell color into register

Control Flow (True Turing-Complete):
- `JUMP`: Unconditional jump (PC + offset)
- `JUMP_IF_EQUAL`: Jump if comparison flag is true
- `JUMP_IF_NOT_EQUAL`: Jump if comparison flag is false

State Management:
- `SET_REG`: Set register to value
- `INC_REG`/`DEC_REG`: Increment/decrement register
- `COMPARE_REG`: Compare two registers, set flag

Parallelism:
- `FORK_ARM`: Create new construction arm
- `SWITCH_ARM`: Switch active arm

- `HALT`: Stop execution

**Spatial Constructor**:
- Maintains grid state and arm positions
- Executes actions sequentially
- Non-differentiable (gradients don't flow through)

## Installation

```bash
git clone https://github.com/ricardoamartinez/arc-universal-constructor.git
cd arc-universal-constructor
pip install -r requirements.txt
```

Dataset downloads automatically on first run (~30MB).

## Usage

### Quick Test
```bash
# See if it runs
python arc_universal_constructor.py --mode demo

# See what it's actually doing
python arc_universal_constructor.py --mode demo --verbose

# Visualize execution step-by-step
python arc_universal_constructor.py --mode demo --viz

# Use Token Turing Machine (dynamic memory)
python arc_universal_constructor.py --mode demo --ttm
```

### Training
```bash
# Basic training (will be slow and may not converge well)
python arc_universal_constructor.py --mode train --epochs 10

# More realistic attempt
python arc_universal_constructor.py --mode train \
    --epochs 50 \
    --lr 1e-3 \
    --memory-augmented \
    --verbose

# Train with Token Turing Machine
python arc_universal_constructor.py --mode train --ttm --read-tokens 16

# DreamCoder wake-sleep learning
python arc_universal_constructor.py --mode dreamcoder --ttm
```

### What Actually Happens During Training

1. Load ARC task (input/output examples)
2. Encode examples with GNCA → task embedding
3. Controller generates action sequence
4. Execute actions on blank grid
5. Compare final grid to target (IoU + color matching)
6. REINFORCE update: increase probability of action sequences that got high rewards

**Important**: This is policy gradient, not supervised learning. The model learns by trial and error, not by direct supervision.

## Current Limitations

1. ~~**Action Parameters**: Operations like MOVE_ARM use hardcoded parameters. The model can't learn custom movement distances.~~ ✅ FIXED: All actions now have learnable parameters.

2. ~~**No Real Branching**: BRANCH_IF_EMPTY exists but doesn't actually branch execution flow.~~ ✅ FIXED: True PC-based execution with jumps and conditionals.

3. **Macro System**: Macro discovery works but macros are just stored sequences, not parameterized functions.

4. **Reward Function**: Current reward is crude (IoU + color matching). Many ARC tasks need exact precision.

5. **Sample Efficiency**: REINFORCE is extremely sample inefficient. Expect to need many epochs for simple patterns.

6. **No Curriculum**: Trains on all tasks randomly. No easy→hard progression.

## Actual Performance

With default settings on evaluation set:
- Random baseline: ~0.8% solve rate
- After 10 epochs: ~1-2% solve rate (mostly trivial tasks)
- Best observed: ~5% with extensive tuning

This is a research prototype, not a competitive ARC solver.

## Advanced Features

### Self-Repair Test
```bash
# Inject 15% random damage and attempt GNCA-based repair
python arc_universal_constructor.py --mode test-damage --damage-rate 0.15
```

The repair uses a learned GNCA rule, not the controller.

### Macro Discovery
```bash
# Run wake-sleep algorithm to find repeated patterns
python arc_universal_constructor.py --mode test-compositional
```

Finds sequences that appear 3+ times and adds them as new actions.

### DreamCoder Integration
```bash
# Run full DreamCoder wake-sleep cycles
python arc_universal_constructor.py --mode dreamcoder --ttm

# More iterations for better abstraction discovery
python arc_universal_constructor.py --mode dreamcoder --dreamcoder-iterations 10
```

Implements the DreamCoder algorithm:
- **Wake Phase**: Neurally-guided program search with beam search
- **Abstraction Sleep**: MDL-based compression to find common patterns
- **Dream Sleep**: Train recognition model on replays + fantasies
- Discovers reusable abstractions as new macro operations

### GPU Visualizer
```bash
# Real-time training visualization with Dear PyGui
python arc_universal_constructor.py --mode train --gpu-viz
```

Shows:
- Grid states (input/target/prediction/construction)
- Neural activations as heatmaps (GNCA, attention, memory)
- Training metrics and loss curves
- Blueprint execution with construction arms
- Interactive controls (pause, step, speed)

## Memory Requirements

- Base model: ~300MB
- With external memory: ~400MB
- GPU recommended but not required
- Batch size fixed at 1 (architectural constraint)

## Code Structure

- `arc_universal_constructor.py`: Everything in one file (~3000 lines)
- `arc_gpu_visualizer.py`: Dear PyGui visualizer (~600 lines)

Key classes:
- `GNCAEncoder`: Perception module
- `TokenController`: Standard TTM-style controller with learnable memory tokens
- `MemoryAugmentedController`: Enhanced controller with external memory (DNC-style)
- `SpatialConstructor`: Grid environment
- `BlueprintInterpreter`: Executes programs
- `train_reinforce()`: Training loop

## If You Want to Modify

Most likely changes:

1. **Add new operations**: 
   - Add to `ConstructorOps` enum
   - Implement in `BlueprintInterpreter._execute_base_op()`
   - Expand controller output size

2. **Change reward function**:
   - Modify `ConstructionEnvironment.compute_reward()`
   - Current: 0.1×activity + 0.2×color + 0.5×IoU + 0.2×exact

3. **Improve action parameters**:
   - Currently hardcoded in `_execute_base_op()`
   - Would need to predict parameters separately

4. **Add curriculum**:
   - Modify task selection in training loop
   - Maybe sort by grid size or complexity

## Why This Approach?

The hypothesis: explicit program synthesis might generalize better than direct pixel prediction. Results so far: unclear. The system can learn simple patterns but struggles with complex ARC tasks.

**Key Design Principle**: Following Von Neumann's vision - minimal fixed machinery with maximal generality. The DSL contains only essential Turing-complete primitives. Complex behaviors (fill patterns, copy operations, search) must emerge from neural learning to compose these primitives, not from hardcoded operations.

This implements ideas from:
- Von Neumann's self-replicating automata
- Neural program synthesis
- Differentiable Neural Computer (DNC)
- DreamCoder (wake-sleep for library learning)

But it's still experimental and not competitive with other ARC approaches. 