#!/usr/bin/env python3
"""
Play Chess vs AI - Interactive chess game against the AI

This script allows you to play chess against the AI using the complete
pipeline: Encoder → Neural Network → Decoder → MCTS.

You can choose:
- Your color (White or Black)
- AI strength (number of MCTS simulations)
- Evaluator type (Neural Network, Random, or Heuristic)

The game displays the board, accepts moves in UCI format (e.g., "e2e4"),
and shows statistics about the AI's thinking process.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chess
import time
from typing import Optional

# Import all components
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural network evaluator disabled.")

from src.game.encoder import BoardEncoder
from src.game.decoder import MoveDecoder
from src.mcts.mcts import MCTS
from src.mcts.evaluator import RandomEvaluator, SimpleHeuristicEvaluator

if TORCH_AVAILABLE:
    from src.models.chess_net import ChessNet
    from src.mcts.evaluator import NeuralNetworkEvaluator


def print_board(board: chess.Board, perspective: chess.Color = chess.WHITE) -> None:
    """
    Print chess board in ASCII format.

    Args:
        board: Chess board to display
        perspective: View from which color (WHITE or BLACK)
    """
    print("\n" + "=" * 40)

    # Column labels
    if perspective == chess.WHITE:
        print("  a b c d e f g h")
        ranks = range(7, -1, -1)  # 8 to 1
    else:
        print("  h g f e d c b a")
        ranks = range(0, 8)  # 1 to 8

    for rank in ranks:
        # Rank number
        print(f"{rank + 1} ", end="")

        # Pieces
        if perspective == chess.WHITE:
            files = range(0, 8)  # a to h
        else:
            files = range(7, -1, -1)  # h to a

        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is None:
                print("· ", end="")
            else:
                # Unicode chess pieces for better display
                symbol = piece.unicode_symbol()
                print(f"{symbol} ", end="")

        print(f" {rank + 1}")

    # Column labels again
    if perspective == chess.WHITE:
        print("  a b c d e f g h")
    else:
        print("  h g f e d c b a")

    print("=" * 40)


def get_game_status(board: chess.Board) -> Optional[str]:
    """
    Get game status message.

    Returns:
        Status string if game is over, None otherwise
    """
    if not board.is_game_over():
        return None

    outcome = board.outcome()

    if outcome.winner == chess.WHITE:
        return "White wins by checkmate!"
    elif outcome.winner == chess.BLACK:
        return "Black wins by checkmate!"
    elif board.is_stalemate():
        return "Draw by stalemate"
    elif board.is_insufficient_material():
        return "Draw by insufficient material"
    elif board.can_claim_fifty_moves():
        return "Draw by fifty-move rule"
    elif board.can_claim_threefold_repetition():
        return "Draw by repetition"
    else:
        return "Game over"


def get_player_move(board: chess.Board) -> Optional[chess.Move]:
    """
    Get move from human player.

    Returns:
        chess.Move or None to quit
    """
    while True:
        move_str = input("\nYour move (e.g., 'e2e4', or 'quit' to exit): ").strip().lower()

        if move_str in ['quit', 'exit', 'q']:
            return None

        if move_str == 'help':
            print("\nAvailable commands:")
            print("  - Enter move in UCI format: e2e4, g1f3, etc.")
            print("  - For castling: e1g1 (kingside), e1c1 (queenside)")
            print("  - For promotion: e7e8q (queen), e7e8n (knight), etc.")
            print("  - Type 'legal' to see all legal moves")
            print("  - Type 'quit' to exit")
            continue

        if move_str == 'legal':
            print("\nLegal moves:")
            legal_moves = list(board.legal_moves)
            for i, move in enumerate(legal_moves, 1):
                print(f"  {move}", end="  ")
                if i % 8 == 0:
                    print()  # New line every 8 moves
            print()
            continue

        try:
            move = chess.Move.from_uci(move_str)

            if move in board.legal_moves:
                return move
            else:
                print(f"Illegal move! '{move_str}' is not legal in this position.")
                print("Type 'legal' to see all legal moves, or 'help' for commands.")
        except Exception as e:
            print(f"Invalid move format! Use UCI notation like 'e2e4'.")
            print("Type 'help' for more information.")


def get_ai_move(board: chess.Board, mcts: MCTS, evaluator, show_thinking: bool = True):
    """
    Get move from AI using MCTS.

    Args:
        board: Current board position
        mcts: MCTS instance
        evaluator: Evaluation function
        show_thinking: Whether to show AI thinking process

    Returns:
        Tuple of (move, stats_dict)
    """
    if show_thinking:
        print("\nAI is thinking...")
        print(f"Running {mcts.num_simulations} MCTS simulations...")

    # Measure time
    start_time = time.time()

    # Run MCTS search
    move, root = mcts.select_move(board, evaluator, return_node=True)

    elapsed = time.time() - start_time

    # Gather statistics
    stats = {
        'move': move,
        'time': elapsed,
        'simulations': mcts.num_simulations,
        'root_visits': root.visit_count,
        'root_value': root.q_value,
        'best_child_visits': root.best_child().visit_count if root.best_child() else 0,
        'best_child_value': root.best_child().q_value if root.best_child() else 0.0
    }

    if show_thinking:
        print(f"✓ Search completed in {elapsed:.2f}s")
        print(f"Position value: {stats['root_value']:.3f} (AI's perspective)")
        print(f"Best move visits: {stats['best_child_visits']}/{stats['root_visits']}")

        # Show top 3 moves considered
        visits = root.get_visit_distribution()
        top_3 = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top moves considered:")
        for i, (mv, count) in enumerate(top_3, 1):
            child = root.children[mv]
            print(f"  {i}. {mv} ({count} visits, Q={child.q_value:.3f})")

    return move, stats


def setup_game():
    """
    Setup game: choose color, AI strength, evaluator type.

    Returns:
        Tuple of (player_color, mcts, evaluator)
    """
    print("=" * 60)
    print(" " * 15 + "CHESS vs AI")
    print("=" * 60)

    # Choose color
    print("\nChoose your color:")
    print("  1. White (you move first)")
    print("  2. Black (AI moves first)")

    while True:
        choice = input("Your choice (1 or 2): ").strip()
        if choice == '1':
            player_color = chess.WHITE
            break
        elif choice == '2':
            player_color = chess.BLACK
            break
        else:
            print("Invalid choice! Enter 1 or 2.")

    # Choose AI strength
    print("\nChoose AI strength:")
    print("  1. Weak (10 simulations, ~0.2s per move)")
    print("  2. Normal (50 simulations, ~1s per move)")
    print("  3. Strong (100 simulations, ~2s per move)")
    print("  4. Very Strong (200 simulations, ~4s per move)")
    print("  5. Custom (enter number)")

    while True:
        choice = input("Your choice (1-5): ").strip()
        if choice == '1':
            num_sims = 10
            break
        elif choice == '2':
            num_sims = 50
            break
        elif choice == '3':
            num_sims = 100
            break
        elif choice == '4':
            num_sims = 200
            break
        elif choice == '5':
            try:
                num_sims = int(input("Enter number of simulations (10-800): "))
                if 10 <= num_sims <= 800:
                    break
                else:
                    print("Please enter a number between 10 and 800.")
            except ValueError:
                print("Invalid number!")
        else:
            print("Invalid choice! Enter 1-5.")

    # Choose evaluator
    print("\nChoose AI evaluator:")
    if TORCH_AVAILABLE:
        print("  1. Neural Network (untrained, random-ish play)")
    print("  2. Random (uniform random moves)")
    print("  3. Heuristic (material-based evaluation)")

    while True:
        choice = input(f"Your choice ({'1-3' if TORCH_AVAILABLE else '2-3'}): ").strip()

        if choice == '1' and TORCH_AVAILABLE:
            print("\nLoading neural network...")
            encoder = BoardEncoder()
            model = ChessNet()
            decoder = MoveDecoder()
            evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
            print("✓ Neural network loaded (untrained)")
            break
        elif choice == '2':
            evaluator = RandomEvaluator()
            print("✓ Random evaluator selected")
            break
        elif choice == '3':
            evaluator = SimpleHeuristicEvaluator()
            print("✓ Heuristic evaluator selected")
            break
        else:
            print("Invalid choice!")

    # Create MCTS
    print(f"\nCreating MCTS with {num_sims} simulations...")
    mcts = MCTS(
        num_simulations=num_sims,
        c_puct=1.5,
        temperature=0.1  # Low temperature for more deterministic play
    )

    print("\n" + "=" * 60)
    print("Game setup complete!")
    print(f"You are playing as {'White' if player_color == chess.WHITE else 'Black'}")
    print(f"AI strength: {num_sims} simulations")
    print("=" * 60)

    return player_color, mcts, evaluator


def play_game():
    """
    Main game loop.
    """
    # Setup
    player_color, mcts, evaluator = setup_game()

    # Create board
    board = chess.Board()

    # Game statistics
    move_count = 0
    game_stats = {
        'total_time_ai': 0.0,
        'total_moves': 0
    }

    print("\nStarting game!")
    print("Type 'help' during your turn for available commands.")

    # Main game loop
    while not board.is_game_over():
        # Display board
        print_board(board, perspective=player_color)

        # Show turn
        turn = "White" if board.turn == chess.WHITE else "Black"
        print(f"\nMove {board.fullmove_number} - {turn} to move")

        # Check game status
        if board.is_check():
            print("⚠ King is in check!")

        # Get move
        if board.turn == player_color:
            # Human player
            move = get_player_move(board)

            if move is None:
                print("\nGame aborted by player.")
                return

            print(f"\nYou played: {move}")

        else:
            # AI player
            move, stats = get_ai_move(board, mcts, evaluator, show_thinking=True)
            print(f"\nAI plays: {move}")

            game_stats['total_time_ai'] += stats['time']
            game_stats['total_moves'] += 1

        # Make move
        board.push(move)
        move_count += 1

    # Game over
    print_board(board, perspective=player_color)

    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    status = get_game_status(board)
    print(f"\n{status}")

    # Show final statistics
    print(f"\nGame statistics:")
    print(f"  Total moves: {move_count}")
    if game_stats['total_moves'] > 0:
        avg_time = game_stats['total_time_ai'] / game_stats['total_moves']
        print(f"  AI total thinking time: {game_stats['total_time_ai']:.1f}s")
        print(f"  AI average per move: {avg_time:.2f}s")

    # Show final position details
    outcome = board.outcome()
    if outcome:
        print(f"\nFinal position:")
        print(f"  Result: {outcome.result()}")
        if outcome.winner is not None:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            print(f"  Winner: {winner}")

    print("\n" + "=" * 60)


def main():
    """
    Main entry point.
    """
    try:
        play_game()

        # Ask to play again
        while True:
            choice = input("\nPlay again? (y/n): ").strip().lower()
            if choice == 'y':
                print("\n" * 3)  # Clear space
                play_game()
            elif choice == 'n':
                print("\nThanks for playing!")
                break
            else:
                print("Please enter 'y' or 'n'")

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
