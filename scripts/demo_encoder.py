#!/usr/bin/env python3
"""
Board Encoder Demo Script

Demonstrates the board encoding functionality by:
1. Encoding various chess positions
2. Visualizing the encoded planes
3. Showing statistics about the encoding
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import chess
    import numpy as np
    from src.game.encoder import BoardEncoder
except ImportError as e:
    print(f"Error: Missing dependencies. Please run: pip install python-chess numpy")
    print(f"Details: {e}")
    sys.exit(1)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_tensor_stats(tensor, board):
    """Print statistics about the encoded tensor"""
    print(f"\nTensor Statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Memory: {tensor.nbytes} bytes")
    print(f"  Non-zero values: {np.count_nonzero(tensor)}")
    print(f"  Min value: {tensor.min():.3f}")
    print(f"  Max value: {tensor.max():.3f}")
    print(f"  Mean value: {tensor.mean():.3f}")

    # Count pieces on board
    piece_count = sum(np.sum(tensor[i]) for i in range(12))
    print(f"  Total pieces encoded: {int(piece_count)}")


def demo_starting_position():
    """Demo encoding of starting position"""
    print_section("Starting Position")

    encoder = BoardEncoder()
    board = chess.Board()

    print("\nChess Board:")
    print(board)

    tensor = encoder.encode(board)
    print_tensor_stats(tensor, board)

    # Show some interesting planes
    print("\nVisualization of key planes:")
    print(encoder.visualize_plane(tensor, 0))  # White pawns
    print(encoder.visualize_plane(tensor, 6))  # Black pawns
    print(encoder.visualize_plane(tensor, 13))  # Turn
    print(encoder.visualize_plane(tensor, 14))  # White castling


def demo_after_moves():
    """Demo encoding after some moves"""
    print_section("After Opening Moves (1.e4 e5 2.Nf3 Nc6)")

    encoder = BoardEncoder()
    board = chess.Board()

    # Play some moves
    moves = ["e4", "e5", "Nf3", "Nc6"]
    for move in moves:
        board.push_san(move)

    print("\nChess Board:")
    print(board)

    tensor = encoder.encode(board)
    print_tensor_stats(tensor, board)

    print("\nVisualization:")
    print(encoder.visualize_plane(tensor, 0))  # White pawns (e4)
    print(encoder.visualize_plane(tensor, 1))  # White knights (Nf3)
    print(encoder.visualize_plane(tensor, 7))  # Black knights (Nc6)


def demo_en_passant():
    """Demo encoding with en passant"""
    print_section("En Passant Situation")

    encoder = BoardEncoder()
    board = chess.Board()

    # Create en passant opportunity
    moves = ["e4", "a6", "e5", "d5"]
    for move in moves:
        board.push_san(move)

    print("\nChess Board:")
    print(board)
    print(f"En passant square: {board.ep_square}")

    tensor = encoder.encode(board)
    print_tensor_stats(tensor, board)

    print("\nVisualization:")
    print(encoder.visualize_plane(tensor, 0))  # White pawns
    print(encoder.visualize_plane(tensor, 6))  # Black pawns
    print(encoder.visualize_plane(tensor, 16))  # En passant square


def demo_endgame():
    """Demo encoding of endgame position"""
    print_section("Endgame Position (King and Pawn vs King)")

    encoder = BoardEncoder()
    board = chess.Board(fen="8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")

    print("\nChess Board:")
    print(board)

    tensor = encoder.encode(board)
    print_tensor_stats(tensor, board)

    print("\nVisualization:")
    print(encoder.visualize_plane(tensor, 0))  # White pawn
    print(encoder.visualize_plane(tensor, 5))  # White king
    print(encoder.visualize_plane(tensor, 11))  # Black king


def demo_castling_rights():
    """Demo castling rights encoding"""
    print_section("Castling Rights Evolution")

    encoder = BoardEncoder()

    # Starting position - full castling rights
    board1 = chess.Board()
    print("\n1. Starting position (full castling rights):")
    print(board1)
    tensor1 = encoder.encode(board1)
    print(encoder.visualize_plane(tensor1, 14))
    print(encoder.visualize_plane(tensor1, 15))

    # After king moves - no castling rights
    board2 = chess.Board()
    board2.push_san("e4")
    board2.push_san("e5")
    board2.push_san("Ke2")  # King moves
    print("\n2. After white king moves (no white castling):")
    print(board2)
    tensor2 = encoder.encode(board2)
    print(encoder.visualize_plane(tensor2, 14))

    # Partial castling rights
    board3 = chess.Board(fen="r3k2r/8/8/8/8/8/8/R3K3 w Qq - 0 1")
    print("\n3. Partial rights (White queenside, Black kingside):")
    print(board3)
    tensor3 = encoder.encode(board3)
    print(encoder.visualize_plane(tensor3, 14))
    print(encoder.visualize_plane(tensor3, 15))


def demo_all_planes():
    """Show all 18 planes for starting position"""
    print_section("All 18 Planes - Starting Position")

    encoder = BoardEncoder()
    board = chess.Board()

    print("\nChess Board:")
    print(board)

    tensor = encoder.encode(board)

    # Show all planes
    for plane_idx in range(18):
        print(encoder.visualize_plane(tensor, plane_idx))


def demo_tactical_position():
    """Demo encoding of a tactical position"""
    print_section("Tactical Position (Back Rank Mate Threat)")

    encoder = BoardEncoder()
    # Position where white has back rank mate threat
    board = chess.Board(fen="6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")

    print("\nChess Board:")
    print(board)
    print("White can play Ra8# for checkmate!")

    tensor = encoder.encode(board)
    print_tensor_stats(tensor, board)

    print("\nVisualization:")
    print(encoder.visualize_plane(tensor, 3))  # White rooks
    print(encoder.visualize_plane(tensor, 5))  # White king
    print(encoder.visualize_plane(tensor, 11))  # Black king


def interactive_demo():
    """Interactive demo - user can input FEN positions"""
    print_section("Interactive Mode")

    encoder = BoardEncoder()

    print("\nEnter FEN positions to encode (or 'quit' to exit)")
    print("Press Enter for starting position")
    print("\nExample FENs:")
    print("  - Starting: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("  - Endgame: 8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")

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

        tensor = encoder.encode(board)
        print_tensor_stats(tensor, board)

        # Ask which plane to visualize
        print("\nWhich plane to visualize? (0-17, or 'all', or 'skip')")
        plane_choice = input("Plane: ").strip().lower()

        if plane_choice == 'all':
            for i in range(18):
                print(encoder.visualize_plane(tensor, i))
        elif plane_choice == 'skip':
            continue
        else:
            try:
                plane_idx = int(plane_choice)
                if 0 <= plane_idx < 18:
                    print(encoder.visualize_plane(tensor, plane_idx))
                else:
                    print("Invalid plane index (must be 0-17)")
            except ValueError:
                print("Invalid input")


def main():
    """Run all demos"""
    print("=" * 70)
    print(" " * 20 + "BOARD ENCODER DEMO")
    print("=" * 70)

    try:
        # Run predefined demos
        demo_starting_position()
        demo_after_moves()
        demo_en_passant()
        demo_endgame()
        demo_castling_rights()
        demo_tactical_position()

        # Ask if user wants to see all planes
        print("\n" + "=" * 70)
        choice = input("\nShow all 18 planes for starting position? (y/n): ").strip().lower()
        if choice == 'y':
            demo_all_planes()

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
