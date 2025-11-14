#!/usr/bin/env python3
"""
Puzzle Testing Script

Questo script testa un modello su puzzle tattici per valutare
le sue capacità di riconoscimento pattern e calcolo tattico.

I puzzle testati includono:
- Mate in 2: Scacco matto in 2 mosse
- (Future: Mate in 3, tactical wins, etc.)

Il test misura:
- Accuracy: % puzzle risolti correttamente
- Performance per categoria (mate-in-2, etc.)
- Performance per difficoltà (easy, medium, hard)

Usage:
    # Test con modello trained
    python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt

    # Test con più simulazioni MCTS (più accurato ma più lento)
    python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --simulations 200

    # Test solo primi N puzzle (per test veloce)
    python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --limit 5

    # Salva risultati
    python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --output results.json

Output esempio:
    Puzzle Testing
    ==================================================
    Total puzzles: 10
    Solved: 7
    Failed: 3
    Accuracy: 70.0%

    By category:
      mate_in_2: 7/10 (70.0%)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse

from src.game.encoder import BoardEncoder
from src.game.decoder import MoveDecoder
from src.models.chess_net import ChessNet
from src.evaluation.puzzles import PuzzleTester, get_builtin_puzzle_set


def load_model(checkpoint_path: str, device: str = 'cpu') -> ChessNet:
    """
    Carica modello da checkpoint.

    Args:
        checkpoint_path: Path al file checkpoint (.pt)
        device: 'cpu' o 'cuda'

    Returns:
        ChessNet model con pesi caricati
    """
    print(f"Loading model from: {checkpoint_path}")

    model = ChessNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Carica state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    print(f"✓ Model loaded (iteration {checkpoint.get('iteration', '?')})")

    return model


def main():
    parser = argparse.ArgumentParser(description='Test model on tactical puzzles')

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (if not specified, uses untrained model)'
    )

    parser.add_argument(
        '--puzzles',
        type=str,
        default='mate_in_2',
        choices=['mate_in_2', 'all'],
        help='Which puzzle set to test (default: mate_in_2)'
    )

    parser.add_argument(
        '--simulations',
        type=int,
        default=100,
        help='MCTS simulations per puzzle (default: 100, higher = more accurate)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Test only first N puzzles (for quick testing)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each puzzle'
    )

    args = parser.parse_args()

    print("="*70)
    print(" "*25 + "PUZZLE TESTING")
    print("="*70)

    # Initialize components
    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()

    # Load model
    if args.model:
        model = load_model(args.model, device=args.device)
    else:
        print("No model specified - using untrained model")
        print("(This will likely perform poorly - train a model first!)")
        model = ChessNet()

    # Load puzzles
    print(f"\nLoading puzzle set: {args.puzzles}")
    puzzles = get_builtin_puzzle_set(args.puzzles)
    print(f"✓ Loaded {len(puzzles)} puzzles")

    # Limit if specified
    if args.limit:
        puzzles = puzzles[:args.limit]
        print(f"  Testing only first {args.limit} puzzles")

    # Create tester
    tester = PuzzleTester(encoder, decoder)

    # Run tests
    print(f"\nMCTS simulations per puzzle: {args.simulations}")
    print("Starting tests...\n")

    results = tester.test_puzzles(
        puzzles=puzzles,
        model=model,
        mcts_simulations=args.simulations,
        verbose=not args.verbose  # If verbose flag, we show per-puzzle detail
    )

    # Show detailed failures if requested
    if args.verbose:
        print("\nDetailed results:")
        for result in results['results']:
            status = "✓" if result['solved'] else "✗"
            print(f"{status} {result['puzzle_id']}: "
                  f"AI={result['ai_move']}, Correct={result['correct_move']}")

    # Save results if requested
    if args.output:
        print(f"\nSaving results to: {args.output}")
        tester.save_results(results, args.output)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    accuracy = results['accuracy']
    print(f"Overall Accuracy: {accuracy:.1%}")

    # Performance interpretation
    if accuracy >= 0.8:
        rating = "Excellent! 🏆"
    elif accuracy >= 0.6:
        rating = "Good! 👍"
    elif accuracy >= 0.4:
        rating = "Decent, but needs improvement 📈"
    elif accuracy >= 0.2:
        rating = "Poor, more training needed 📚"
    else:
        rating = "Very poor, check model & training ⚠️"

    print(f"Rating: {rating}")

    # Comparison to human levels
    print("\nContext:")
    print("  ~90%+ : Strong club player level")
    print("  ~70%+ : Intermediate player")
    print("  ~50%+ : Beginner who knows tactics")
    print("  <30%  : Needs more training")

    print("\n" + "="*70)

    # Return code based on performance
    if accuracy >= 0.5:
        return 0  # Success
    else:
        return 1  # Needs improvement


if __name__ == "__main__":
    sys.exit(main())
