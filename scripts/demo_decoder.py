#!/usr/bin/env python3
"""
Move Decoder Demo Script

Demonstrates the move decoder functionality by:
1. Creating random policy vectors
2. Decoding to legal moves
3. Visualizing move probabilities
4. Comparing greedy vs sampling selection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import chess
    import numpy as np
    from src.game.decoder import MoveDecoder
except ImportError as e:
    print(f"Error: Missing dependencies. Please run: pip install python-chess numpy")
    print(f"Details: {e}")
    sys.exit(1)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_decoding():
    """Demo basic move decoding"""
    print_section("Basic Move Decoding")

    decoder = MoveDecoder()
    board = chess.Board()

    print("\nChess Board (Starting Position):")
    print(board)
    print(f"Legal moves: {len(list(board.legal_moves))}")

    # Create random policy
    print("\nCreating random policy vector...")
    policy = decoder.create_random_policy()
    print(f"Policy shape: {policy.shape}")
    print(f"Policy dtype: {policy.dtype}")

    # Show policy statistics
    print(f"\nPolicy statistics:")
    print(f"  Min: {policy.min():.3f}")
    print(f"  Max: {policy.max():.3f}")
    print(f"  Mean: {policy.mean():.3f}")
    print(f"  Std: {policy.std():.3f}")

    # Visualize top moves
    print(decoder.visualize_policy(policy, board, top_k=10))


def demo_greedy_selection():
    """Demo greedy move selection"""
    print_section("Greedy Move Selection")

    decoder = MoveDecoder()
    board = chess.Board()

    print("\nStarting Position:")
    print(board)

    # Create policy with bias toward e2e4
    policy = np.random.randn(4672).astype(np.float32) * 0.1
    e2e4 = chess.Move.from_uci("e2e4")
    e2e4_idx = decoder.encode_move(e2e4)
    policy[e2e4_idx] = 5.0  # Give e2e4 high score

    print("\nPolicy biased toward e2e4...")

    # Select best move (should be e2e4)
    best_move = decoder.select_move_greedy(policy, board)
    print(f"\nBest move (greedy): {best_move}")

    # Verify it's deterministic
    best_move2 = decoder.select_move_greedy(policy, board)
    print(f"Second call: {best_move2}")
    print(f"Deterministic: {best_move == best_move2}")

    # Show top 5 moves
    print(decoder.visualize_policy(policy, board, top_k=5))


def demo_sampling():
    """Demo move sampling with different temperatures"""
    print_section("Move Sampling with Different Temperatures")

    decoder = MoveDecoder()
    board = chess.Board()

    print("\nStarting Position:")
    print(board)

    # Create uniform policy (all moves equal)
    policy = decoder.create_uniform_policy()

    print("\nUniform policy (all moves equally likely)...")
    print(decoder.visualize_policy(policy, board, top_k=8))

    # Sample with different temperatures
    print("\nSampling 10 moves with different temperatures:")
    print("-" * 70)

    for temp in [0.1, 1.0, 2.0]:
        print(f"\nTemperature {temp}:")
        moves = []
        for _ in range(10):
            move = decoder.select_move_sampling(policy, board, temperature=temp)
            moves.append(str(move))

        # Count unique moves
        unique_moves = len(set(moves))
        print(f"  Unique moves: {unique_moves}/10")
        print(f"  Samples: {', '.join(moves[:5])}...")


def demo_endgame_position():
    """Demo decoding in endgame with fewer moves"""
    print_section("Endgame Position (Limited Moves)")

    decoder = MoveDecoder()

    # Simple endgame: King and Pawn vs King
    board = chess.Board(fen="8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")

    print("\nEndgame Position:")
    print(board)
    print(f"Legal moves: {len(list(board.legal_moves))}")

    # Create random policy
    policy = decoder.create_random_policy()

    # Visualize all legal moves
    top_moves = decoder.get_top_moves(policy, board, top_k=20)
    print(f"\nAll {len(top_moves)} legal moves with probabilities:")
    print(decoder.visualize_policy(policy, board, top_k=20))


def demo_tactical_position():
    """Demo decoding in tactical position"""
    print_section("Tactical Position (Back Rank Mate Available)")

    decoder = MoveDecoder()

    # Position with mate in 1
    board = chess.Board(fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

    print("\nPosition (White to move):")
    print(board)
    print("White can play Ra8# for checkmate!")
    print(f"Legal moves: {len(list(board.legal_moves))}")

    # Create policy that prefers Ra8
    policy = np.random.randn(4672).astype(np.float32) * 0.5
    ra8 = chess.Move.from_uci("a1a8")
    ra8_idx = decoder.encode_move(ra8)
    policy[ra8_idx] = 10.0  # Strongly prefer mate

    print("\nPolicy tuned to prefer Ra8#:")
    print(decoder.visualize_policy(policy, board, top_k=8))

    best_move = decoder.select_move_greedy(policy, board)
    print(f"\nBest move: {best_move}")
    print(f"Is mate: {board.is_checkmate() if board.is_game_over() else 'No, will check after move'}")


def demo_move_encoding():
    """Demo move encoding/decoding"""
    print_section("Move Encoding/Decoding")

    decoder = MoveDecoder()

    print("\nTesting move encoding and decoding:")
    print("-" * 70)

    test_moves = [
        ("e2e4", "Pawn push"),
        ("g1f3", "Knight move"),
        ("e1g1", "Kingside castling (as king move)"),
        ("a1h8", "Long diagonal (rook/bishop/queen)"),
        ("e7e8q", "Queen promotion"),
        ("e7e8n", "Knight underpromotion"),
        ("b7c8r", "Rook underpromotion"),
    ]

    for move_uci, description in test_moves:
        try:
            move = chess.Move.from_uci(move_uci)
            policy_idx = decoder.encode_move(move)

            print(f"\n{description}:")
            print(f"  Move: {move_uci}")
            print(f"  Policy index: {policy_idx}")

            if policy_idx is not None:
                decoded = decoder.decode_policy_index(policy_idx)
                from_sq, to_sq, promo = decoded
                print(f"  Decoded: from={chess.square_name(from_sq)}, to={chess.square_name(to_sq)}, promo={promo}")
            else:
                print(f"  Could not encode")
        except Exception as e:
            print(f"  Error: {e}")


def demo_policy_comparison():
    """Demo comparing different policy types"""
    print_section("Policy Type Comparison")

    decoder = MoveDecoder()
    board = chess.Board()

    print("\nStarting Position:")
    print(board)

    # Random policy
    print("\n1. Random Policy:")
    policy_random = decoder.create_random_policy()
    move_random = decoder.select_move_greedy(policy_random, board)
    print(f"   Best move: {move_random}")
    print(decoder.visualize_policy(policy_random, board, top_k=5))

    # Uniform policy
    print("\n2. Uniform Policy (all equal):")
    policy_uniform = decoder.create_uniform_policy()
    move_uniform = decoder.select_move_greedy(policy_uniform, board)
    print(f"   Best move: {move_uniform}")
    print(decoder.visualize_policy(policy_uniform, board, top_k=5))

    # Biased policy
    print("\n3. Biased Policy (prefer central pawn moves):")
    policy_biased = np.random.randn(4672).astype(np.float32) * 0.1

    # Boost central pawn moves
    for move_uci in ["e2e4", "d2d4"]:
        move = chess.Move.from_uci(move_uci)
        idx = decoder.encode_move(move)
        if idx is not None:
            policy_biased[idx] = 3.0

    move_biased = decoder.select_move_greedy(policy_biased, board)
    print(f"   Best move: {move_biased}")
    print(decoder.visualize_policy(policy_biased, board, top_k=5))


def demo_complete_game_simulation():
    """Demo simulating a few moves with decoder"""
    print_section("Simulating Game Moves")

    decoder = MoveDecoder()
    board = chess.Board()

    print("\nSimulating 5 moves with random policies:\n")

    for move_num in range(1, 6):
        if board.is_game_over():
            break

        # Create random policy
        policy = decoder.create_random_policy()

        # Select move (using sampling for variety)
        move = decoder.select_move_sampling(policy, board, temperature=1.0)

        if move is None:
            break

        # Show position before move
        turn = "White" if board.turn == chess.WHITE else "Black"
        print(f"\nMove {move_num} - {turn} to play:")
        print(f"Selected move: {move}")

        # Make move
        board.push(move)

        # Show board
        print(board)

    print(f"\nFinal position after {board.fullmove_number - 1} moves")


def interactive_demo():
    """Interactive demo - decode custom positions"""
    print_section("Interactive Mode")

    decoder = MoveDecoder()

    print("\nEnter FEN positions to decode moves (or 'quit' to exit)")
    print("Press Enter for starting position")

    while True:
        print("\n" + "-" * 70)
        fen = input("\nFEN (or Enter for starting position, 'quit' to exit): ").strip()

        if fen.lower() in ['quit', 'exit', 'q']:
            break

        if not fen:
            board = chess.Board()
        else:
            try:
                board = chess.Board(fen=fen)
            except Exception as e:
                print(f"Invalid FEN: {e}")
                continue

        print("\nChess Board:")
        print(board)
        print(f"Legal moves: {len(list(board.legal_moves))}")

        if board.is_game_over():
            print("Game over!")
            continue

        # Create random policy
        policy = decoder.create_random_policy()

        # Show visualization
        print(decoder.visualize_policy(policy, board, top_k=10))

        # Ask for action
        action = input("\nAction (greedy/sample/top10/skip): ").strip().lower()

        if action == 'greedy':
            move = decoder.select_move_greedy(policy, board)
            print(f"Greedy selection: {move}")
        elif action == 'sample':
            move = decoder.select_move_sampling(policy, board)
            print(f"Sampled move: {move}")
        elif action == 'top10':
            top = decoder.get_top_moves(policy, board, top_k=10)
            print("\nTop 10 moves:")
            for rank, (m, p) in enumerate(top, 1):
                print(f"  {rank}. {m} ({p*100:.2f}%)")


def main():
    """Run all demos"""
    print("=" * 70)
    print(" " * 20 + "MOVE DECODER DEMO")
    print("=" * 70)

    try:
        # Run predefined demos
        demo_basic_decoding()
        demo_greedy_selection()
        demo_sampling()
        demo_endgame_position()
        demo_tactical_position()
        demo_move_encoding()
        demo_policy_comparison()

        # Ask if user wants game simulation
        print("\n" + "=" * 70)
        choice = input("\nSimulate a few game moves? (y/n): ").strip().lower()
        if choice == 'y':
            demo_complete_game_simulation()

        # Ask if user wants interactive mode
        print("\n" + "=" * 70)
        choice = input("\nEnter interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_demo()

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
