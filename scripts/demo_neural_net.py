#!/usr/bin/env python3
"""
Chess Neural Network Demo Script

Demonstrates the neural network functionality by:
1. Creating and inspecting the network architecture
2. Testing forward and backward passes
3. Integrating with encoder/decoder
4. Saving and loading models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    import chess
    from src.models.chess_net import ChessNet, create_chess_net
    from src.models.model_utils import save_model, load_model, get_model_size
    from src.game.encoder import BoardEncoder
    from src.game.decoder import MoveDecoder
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    print("Please install: pip install torch numpy python-chess")
    sys.exit(1)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_architecture():
    """Demo network architecture"""
    print_section("Network Architecture")

    # Create network
    net = create_chess_net()

    print("\nDefault Configuration:")
    config = net.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Count parameters
    size_info = get_model_size(net)
    print(f"\nModel Size:")
    print(f"  Parameters: {size_info['num_parameters']:,}")
    print(f"  Size: {size_info['size_mb']:.2f} MB")

    # Show layers
    print(f"\nNetwork Structure:")
    print(net)


def demo_forward_pass():
    """Demo forward pass"""
    print_section("Forward Pass")

    net = create_chess_net()

    # Create random input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 18, 8, 8)

    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    policy_logits, value = net(dummy_input)

    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")

    print(f"\nPolicy logits range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]")
    print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")

    # Test prediction (with softmax)
    policy_probs, value_pred = net.predict(dummy_input)

    print(f"\nAfter softmax:")
    print(f"Policy probabilities sum: {policy_probs.sum(dim=1)}")
    print(f"Max probability: {policy_probs.max():.4f}")
    print(f"Min probability: {policy_probs.min():.6f}")


def demo_integration():
    """Demo integration with encoder and decoder"""
    print_section("Integration with Encoder/Decoder")

    # Create components
    net = create_chess_net()
    encoder = BoardEncoder()
    decoder = MoveDecoder()

    # Create chess position
    board = chess.Board()
    print("\nStarting position:")
    print(board)

    # Encode board
    board_tensor = encoder.encode(board)
    print(f"\nEncoded shape: {board_tensor.shape}")

    # Convert to PyTorch tensor and add batch dimension
    board_tensor_torch = torch.from_numpy(board_tensor).unsqueeze(0)
    print(f"PyTorch tensor shape: {board_tensor_torch.shape}")

    # Forward pass through network
    policy_logits, value = net(board_tensor_torch)

    print(f"\nNetwork output:")
    print(f"  Policy logits: {policy_logits.shape}")
    print(f"  Value: {value.item():.3f}")

    # Decode to move
    policy_np = policy_logits.detach().numpy()[0]
    best_move = decoder.select_move_greedy(policy_np, board)

    print(f"\nSelected move: {best_move}")

    # Show top 5 moves
    top_moves = decoder.get_top_moves(policy_np, board, top_k=5)
    print("\nTop 5 moves:")
    for i, (move, prob) in enumerate(top_moves, 1):
        print(f"  {i}. {move} ({prob*100:.2f}%)")


def demo_training_step():
    """Demo a simple training step"""
    print_section("Training Step Example")

    net = create_chess_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print("\nCreating dummy training batch...")
    batch_size = 16

    # Dummy data
    positions = torch.randn(batch_size, 18, 8, 8)
    target_policy = torch.randn(batch_size, 4672)  # Should be real policy from MCTS
    target_value = torch.randn(batch_size, 1)  # Should be game outcome

    # Forward pass
    pred_policy, pred_value = net(positions)

    # Compute losses
    policy_loss = torch.nn.functional.mse_loss(pred_policy, target_policy)
    value_loss = torch.nn.functional.mse_loss(pred_value, target_value)
    total_loss = policy_loss + value_loss

    print(f"Losses before update:")
    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Value loss: {value_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("\n✓ Backward pass and weight update completed")


def demo_save_load():
    """Demo save/load functionality"""
    print_section("Save/Load Model")

    import tempfile
    import os

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'demo_model.pth')

        # Create and save model
        print("\nCreating model...")
        net1 = create_chess_net()

        # Make prediction with original model
        x = torch.randn(2, 18, 8, 8)
        policy1, value1 = net1.predict(x)
        print(f"Original model prediction: value={value1[0].item():.3f}")

        # Save model
        print(f"\nSaving model to {filepath}...")
        save_model(net1, filepath, metadata={'demo': True})

        # Load model
        print("\nLoading model...")
        net2 = create_chess_net()
        load_model(filepath, model=net2)

        # Make prediction with loaded model
        policy2, value2 = net2.predict(x)
        print(f"Loaded model prediction: value={value2[0].item():.3f}")

        # Check predictions match
        match = torch.allclose(value1, value2)
        print(f"\nPredictions match: {match} ✓" if match else f"\nPredictions match: {match} ✗")


def demo_complete_pipeline():
    """Demo complete pipeline: position → prediction → move"""
    print_section("Complete Pipeline")

    # Create components
    net = create_chess_net()
    encoder = BoardEncoder()
    decoder = MoveDecoder()

    # Test on a few positions
    positions = [
        ("Starting position", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Endgame", chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"))
    ]

    for name, board in positions:
        print(f"\n{name}:")
        print(board)

        # Encode
        board_np = encoder.encode(board)
        board_torch = torch.from_numpy(board_np).unsqueeze(0)

        # Predict
        policy, value = net.predict(board_torch)

        print(f"  Position value: {value[0].item():.3f}")

        # Decode move
        policy_np = policy.detach().numpy()[0]
        best_move = decoder.select_move_greedy(policy_np, board)

        print(f"  Best move (untrained network): {best_move}")


def main():
    """Run all demos"""
    print("=" * 70)
    print(" " * 20 + "CHESS NEURAL NETWORK DEMO")
    print("=" * 70)

    try:
        demo_architecture()

        choice = input("\nContinue to forward pass demo? (y/n): ").strip().lower()
        if choice == 'y':
            demo_forward_pass()

        choice = input("\nContinue to encoder/decoder integration? (y/n): ").strip().lower()
        if choice == 'y':
            demo_integration()

        choice = input("\nContinue to training step demo? (y/n): ").strip().lower()
        if choice == 'y':
            demo_training_step()

        choice = input("\nContinue to save/load demo? (y/n): ").strip().lower()
        if choice == 'y':
            demo_save_load()

        choice = input("\nContinue to complete pipeline demo? (y/n): ").strip().lower()
        if choice == 'y':
            demo_complete_pipeline()

        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
