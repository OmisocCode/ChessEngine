#!/usr/bin/env python3
"""
Model Comparison Script

Questo script confronta due modelli facendoli giocare l'uno contro l'altro.
È utile per:

1. **Verificare miglioramento**: Confronta checkpoint nuovi vs vecchi
   Es: model_iter_20.pt vs model_iter_10.pt

2. **A/B Testing**: Confronta architetture o hyperparameters diversi

3. **Selezione checkpoint**: Trova il miglior checkpoint da usare

Il confronto avviene tramite match diretti:
- Modello 1 vs Modello 2
- Numero configurabile di partite
- Alternanza colori per fairness
- Statistiche dettagliate (W-D-L, avg moves, time)

Usage:
    # Confronta due checkpoint
    python scripts/compare_models.py \\
        --model1 checkpoints/model_iter_10.pt \\
        --model2 checkpoints/model_iter_20.pt \\
        --games 50

    # Confronta vs random baseline
    python scripts/compare_models.py \\
        --model1 checkpoints/model_iter_20.pt \\
        --random \\
        --games 100

    # Più simulazioni MCTS (più lento ma più accurato)
    python scripts/compare_models.py \\
        --model1 checkpoints/model_iter_15.pt \\
        --model2 checkpoints/model_iter_20.pt \\
        --games 30 \\
        --simulations 200

Output esempio:
    Model Comparison: Iteration 10 vs Iteration 20
    ================================================
    Total games: 50
    Iteration 10: 15W - 10D - 25L (30.0%)
    Iteration 20: 25W - 10D - 15L (50.0%)
    Winner: Iteration 20
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
from src.evaluation.evaluator import ModelEvaluator


def load_model(checkpoint_path: str, device: str = 'cpu') -> ChessNet:
    """
    Carica modello da checkpoint.

    Args:
        checkpoint_path: Path al file checkpoint (.pt)
        device: 'cpu' o 'cuda'

    Returns:
        Tuple of (model, iteration_number)
    """
    print(f"Loading: {checkpoint_path}")

    model = ChessNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Carica state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    iteration = checkpoint.get('iteration', '?')
    print(f"  ✓ Loaded (iteration {iteration})")

    return model, iteration


def main():
    parser = argparse.ArgumentParser(
        description='Compare two chess models by playing matches'
    )

    parser.add_argument(
        '--model1',
        type=str,
        required=True,
        help='Path to first model checkpoint'
    )

    parser.add_argument(
        '--model2',
        type=str,
        default=None,
        help='Path to second model checkpoint'
    )

    parser.add_argument(
        '--random',
        action='store_true',
        help='Compare model1 against random player (instead of model2)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=50,
        help='Number of games to play (must be even, default: 50)'
    )

    parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='MCTS simulations per move (default: 50)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.random and not args.model2:
        print("Error: Must specify either --model2 or --random")
        return 1

    if args.random and args.model2:
        print("Error: Cannot specify both --model2 and --random")
        return 1

    if args.games % 2 != 0:
        print("Warning: Number of games should be even for fair comparison")
        print(f"  Rounding {args.games} -> {args.games + 1}")
        args.games += 1

    print("="*70)
    print(" "*25 + "MODEL COMPARISON")
    print("="*70)

    # Initialize components
    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    evaluator = ModelEvaluator(encoder, decoder)
    print("✓ Components initialized")

    # Load model 1
    print("\nLoading Model 1...")
    model1, iter1 = load_model(args.model1, device=args.device)
    model1_name = f"Model 1 (iter {iter1})"

    # Load model 2 or use random
    if args.random:
        print("\nModel 2: Random Player")
        model2 = None
        model2_name = "Random Player"
    else:
        print("\nLoading Model 2...")
        model2, iter2 = load_model(args.model2, device=args.device)
        model2_name = f"Model 2 (iter {iter2})"

    print("\n" + "="*70)
    print("MATCH SETUP")
    print("="*70)
    print(f"Competitor 1: {model1_name}")
    print(f"Competitor 2: {model2_name}")
    print(f"Number of games: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print("="*70)

    # Run comparison
    if args.random:
        # Model vs Random
        print("\nRunning evaluation vs random player...")
        results = evaluator.evaluate_vs_random(
            model=model1,
            num_games=args.games,
            mcts_simulations=args.simulations,
            verbose=True
        )

        # Reformat results for consistency
        comparison_results = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'total_games': results['total_games'],
            'model1_wins': results['wins'],
            'model2_wins': results['losses'],
            'draws': results['draws'],
            'model1_win_rate': results['win_rate'],
            'model2_win_rate': results['losses'] / results['total_games'],
            'games': results['games']
        }

    else:
        # Model vs Model
        print("\nRunning head-to-head comparison...")
        comparison_results = evaluator.compare_models(
            model1=model1,
            model2=model2,
            num_games=args.games,
            mcts_simulations=args.simulations,
            model1_name=model1_name,
            model2_name=model2_name,
            verbose=True
        )

    # Save results if requested
    if args.output:
        print(f"\nSaving results to: {args.output}")
        evaluator.save_results(comparison_results, args.output)

    # Additional analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    model1_wins = comparison_results['model1_wins']
    model2_wins = comparison_results['model2_wins']
    draws = comparison_results['draws']
    total = comparison_results['total_games']

    # Win margin
    win_margin = abs(model1_wins - model2_wins)
    print(f"Win margin: {win_margin} games")

    # Strength difference interpretation
    if win_margin / total >= 0.4:
        strength_diff = "Significant difference"
    elif win_margin / total >= 0.2:
        strength_diff = "Moderate difference"
    elif win_margin / total >= 0.1:
        strength_diff = "Small difference"
    else:
        strength_diff = "Minimal difference (nearly equal)"

    print(f"Strength assessment: {strength_diff}")

    # Recommendation
    print("\nRecommendation:")
    if model1_wins > model2_wins + draws/2:
        print(f"  → Use {model1_name}")
    elif model2_wins > model1_wins + draws/2:
        print(f"  → Use {model2_name}")
    else:
        print("  → Models are roughly equal in strength")

    # Statistical confidence (simple)
    # For proper confidence intervals, use more sophisticated methods
    decisive_games = model1_wins + model2_wins  # Exclude draws
    if decisive_games > 0:
        confidence = max(model1_wins, model2_wins) / decisive_games
        print(f"\nConfidence (decisive games): {confidence:.1%}")
        if confidence < 0.6:
            print("  Note: Low confidence - consider more games for reliable comparison")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
