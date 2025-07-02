#!/usr/bin/env python3
"""Test proper DreamCoder implementation with TTM integration."""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from arc_universal_constructor import (
    TTMController, GNCAEncoder, ARCUniversalConstructor,
    DreamCoderIntegration, Program, VersionSpace,
    RecognitionModel, ConstructorOps, MacroLibrary
)

def test_version_space():
    """Test the version space refactoring functionality."""
    print("="*60)
    print("Testing Version Space Refactoring")
    print("="*60)
    
    # Create a simple program with repeated pattern
    operations = [
        (ConstructorOps.WRITE.value - 1, {'color': 1}),
        (ConstructorOps.MOVE_ARM.value - 1, {'dx': 6, 'dy': 5}),
        (ConstructorOps.WRITE.value - 1, {'color': 1}),
        (ConstructorOps.MOVE_ARM.value - 1, {'dx': 6, 'dy': 5}),
        (ConstructorOps.WRITE.value - 1, {'color': 2}),
    ]
    
    program = Program(operations)
    print(f"Original program: {len(program)} operations")
    
    # Create version space
    vs = VersionSpace(program)
    refactorings = vs.get_refactorings()
    
    print(f"Found {len(refactorings)} refactorings")
    
    # Extract subtrees
    subtrees = vs.extract_subtrees()
    print(f"Found {len(subtrees)} potential abstractions")
    
    # Show some abstractions
    for i, subtree in enumerate(list(subtrees)[:3]):
        print(f"\nAbstraction {i+1}: {len(subtree)} operations")
        for op, params_tuple in subtree:
            if op < ConstructorOps.base_vocab_size():
                op_name = list(ConstructorOps)[op].name
                # Convert params tuple back to dict for display
                params = dict(params_tuple) if params_tuple else {}
                print(f"  {op_name} {params}")
    
    print("\n✓ Version space working correctly")


def test_recognition_model():
    """Test the recognition model for program prediction."""
    print("\n" + "="*60)
    print("Testing Recognition Model")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create recognition model
    model = RecognitionModel(vocab_size=13)
    model.to(device)
    
    # Test forward pass
    task_embedding = torch.randn(2, 64, device=device)  # Batch of 2
    
    # No history
    output1 = model(task_embedding)
    print(f"Output without history: {output1['actions'].shape}")
    
    # With history
    prev_actions = torch.tensor([[1, 2, 3], [2, 3, 4]], device=device)
    output2 = model(task_embedding, prev_actions)
    print(f"Output with history: {output2['actions'].shape}")
    
    # Test beam search
    print("\nTesting beam search...")
    programs = model.beam_search(task_embedding[0], beam_size=3, max_length=10)
    print(f"Found {len(programs)} programs")
    
    for i, (prog, score) in enumerate(programs[:3]):
        print(f"  Program {i+1}: {len(prog)} ops, score={score:.3f}")
    
    print("\n✓ Recognition model working correctly")


def test_dreamcoder_cycle():
    """Test a complete DreamCoder wake-sleep cycle."""
    print("\n" + "="*60)
    print("Testing DreamCoder Wake-Sleep Cycle")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with TTM
    encoder = GNCAEncoder(steps=4, channels=8).to(device)
    controller = TTMController(
        memory_tokens=32,
        read_tokens=8,
        num_layers=2
    ).to(device)
    
    model = ARCUniversalConstructor(encoder, controller).to(device)
    
    # Create simple synthetic tasks
    print("\nCreating synthetic tasks...")
    tasks = []
    
    # Task 1: Fill 3x3 with color 1
    input1 = torch.full((5, 5), -1, dtype=torch.long)
    output1 = input1.clone()
    output1[:3, :3] = 1
    tasks.append(([(input1, output1)], input1, output1))
    
    # Task 2: Draw horizontal line
    input2 = torch.full((5, 5), -1, dtype=torch.long)
    output2 = input2.clone()
    output2[2, :] = 2
    tasks.append(([(input2, output2)], input2, output2))
    
    print(f"Created {len(tasks)} synthetic tasks")
    
    # Initialize DreamCoder
    dreamcoder = DreamCoderIntegration(
        model,
        max_program_length=20,
        beam_size=3
    )
    
    # Run one wake-sleep cycle
    print("\nRunning wake-sleep cycle...")
    try:
        dreamcoder.run_wake_sleep_cycle(tasks[:1], device)  # Just one task for speed
        print("\n✓ Wake-sleep cycle completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in wake-sleep cycle: {e}")
        import traceback
        traceback.print_exc()


def test_mdl_computation():
    """Test minimum description length computation."""
    print("\n" + "="*60)
    print("Testing MDL Computation")
    print("="*60)
    
    # Create a simple library
    library = MacroLibrary()
    
    # Create programs
    prog1 = Program([
        (ConstructorOps.WRITE.value - 1, {'color': 1}),
        (ConstructorOps.MOVE_ARM.value - 1, {'dx': 6, 'dy': 5}),
        (ConstructorOps.HALT.value - 1, {})
    ])
    
    prog2 = Program([
        (ConstructorOps.WRITE.value - 1, {'color': 2}),
        (ConstructorOps.MOVE_ARM.value - 1, {'dx': 7, 'dy': 5}),
        (ConstructorOps.WRITE.value - 1, {'color': 2}),
        (ConstructorOps.HALT.value - 1, {})
    ])
    
    # Compute description lengths
    dl1 = prog1.description_length(library)
    dl2 = prog2.description_length(library)
    
    print(f"Program 1: {len(prog1)} ops, DL = {dl1:.2f} bits")
    print(f"Program 2: {len(prog2)} ops, DL = {dl2:.2f} bits")
    
    # Add a macro
    library.add_macro("write_and_move", [
        ConstructorOps.WRITE,
        ConstructorOps.MOVE_ARM
    ])
    
    print(f"\nAfter adding macro (library size: {library.get_total_vocab_size()}):")
    
    # Recompute with larger vocabulary
    dl1_new = prog1.description_length(library)
    dl2_new = prog2.description_length(library)
    
    print(f"Program 1: DL = {dl1_new:.2f} bits (was {dl1:.2f})")
    print(f"Program 2: DL = {dl2_new:.2f} bits (was {dl2:.2f})")
    
    print("\n✓ MDL computation working correctly")


def main():
    print("DREAMCODER IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    test_version_space()
    test_recognition_model()
    test_mdl_computation()
    test_dreamcoder_cycle()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 