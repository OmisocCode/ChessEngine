#!/usr/bin/env python3
"""
Demo Self-Play Game Viewer

Questo script genera una partita self-play e la mostra in formato PGN,
permettendo di visualizzare le mosse giocate dall'AI.

Le partite possono essere salvate in file PGN e poi aperte con qualsiasi
visualizzatore di scacchi (lichess.org, chess.com, etc.)

Usage:
    python scripts/demo_selfplay_game.py
    python scripts/demo_selfplay_game.py --simulations 50
    python scripts/demo_selfplay_game.py --output game.pgn
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chess
import chess.pgn
import argparse
from datetime import datetime

from src.game.encoder import BoardEncoder
from src.game.decoder import MoveDecoder
from src.models.chess_net import ChessNet
from src.mcts.mcts import MCTS
from src.mcts.evaluator import NeuralNetworkEvaluator
from src.training.self_play import SelfPlayGame


def play_and_save_game(
    num_simulations: int = 50,
    output_file: str = None,
    verbose: bool = True,
    model_path: str = None
):
    """
    Gioca una partita self-play e la salva in formato PGN.

    Args:
        num_simulations: Numero di simulazioni MCTS per mossa
        output_file: Path del file PGN dove salvare (opzionale)
        verbose: Stampa dettagli durante la partita
        model_path: Path a checkpoint model (opzionale, default: rete untrained)

    Returns:
        chess.pgn.Game object con la partita giocata
    """
    if verbose:
        print("=" * 70)
        print("DEMO SELF-PLAY GAME")
        print("=" * 70)
        print(f"\nMCTS simulations per move: {num_simulations}")

    # Inizializza componenti
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    model = ChessNet()

    # Carica modello se specificato
    if model_path:
        if verbose:
            print(f"Loading model from: {model_path}")
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if verbose:
            print("✓ Model loaded")
    else:
        if verbose:
            print("Using untrained network (random play)")

    # Crea evaluator e MCTS
    evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
    mcts = MCTS(num_simulations=num_simulations, c_puct=1.5)

    # Crea self-play game
    game_manager = SelfPlayGame(
        encoder=encoder,
        decoder=decoder,
        mcts=mcts,
        evaluator=evaluator,
        temperature_threshold=30,
        high_temperature=1.0,
        low_temperature=0.1
    )

    if verbose:
        print("\nPlaying game...")
        print("(This may take a while depending on num_simulations)")
        print()

    # Tracciamento mosse per PGN
    board = chess.Board()
    move_list = []
    move_times = []

    # Gioca la partita mossa per mossa
    move_count = 0
    import time

    while not board.is_game_over() and move_count < 500:
        move_count += 1

        # Ottieni temperatura
        temperature = game_manager.get_temperature(move_count)

        if verbose:
            print(f"Move {move_count}: ", end="", flush=True)

        # Esegui MCTS search
        move_start = time.time()
        root = mcts.search(board, evaluator)
        move_time = time.time() - move_start

        # Ottieni policy e seleziona mossa
        mcts_policy_dict = root.get_policy_distribution(temperature=temperature)

        # Seleziona mossa (con temperature)
        if temperature > 0.5:
            import numpy as np
            moves = list(mcts_policy_dict.keys())
            probs = np.array(list(mcts_policy_dict.values()))
            probs = probs / probs.sum()
            chosen_move = np.random.choice(moves, p=probs)
        else:
            chosen_move = max(mcts_policy_dict.items(), key=lambda x: x[1])[0]

        # Salva mossa e tempo
        move_list.append(chosen_move)
        move_times.append(move_time)

        if verbose:
            # Mostra mossa in notazione SAN (Standard Algebraic Notation)
            san_move = board.san(chosen_move)
            print(f"{san_move} ({move_time:.1f}s)")

        # Esegui mossa
        board.push(chosen_move)

    if verbose:
        print()

    # Crea PGN game
    pgn_game = chess.pgn.Game()

    # Aggiungi headers
    pgn_game.headers["Event"] = "Self-Play Training Game"
    pgn_game.headers["Site"] = "ChessEngine AI"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["White"] = "ChessEngine AI"
    pgn_game.headers["Black"] = "ChessEngine AI"

    # Determina risultato
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = "0-1"  # Black wins
        else:
            result = "1-0"  # White wins
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "1/2-1/2"
    elif board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = "1/2-1/2"
    elif move_count >= 500:
        result = "1/2-1/2"
    else:
        result = "*"

    pgn_game.headers["Result"] = result

    # Aggiungi annotazioni
    pgn_game.headers["Annotator"] = "ChessEngine Self-Play"
    pgn_game.headers["MCTSSimulations"] = str(num_simulations)
    if model_path:
        pgn_game.headers["Model"] = model_path

    # Aggiungi mosse al PGN
    node = pgn_game
    for move in move_list:
        node = node.add_variation(move)

    # Stampa statistiche
    if verbose:
        print("=" * 70)
        print("GAME SUMMARY")
        print("=" * 70)
        print(f"Result: {result}")
        print(f"Total moves: {len(move_list)}")
        print(f"Total time: {sum(move_times):.1f}s")
        print(f"Average time per move: {sum(move_times)/len(move_times):.1f}s")
        print()

        # Motivo fine partita
        if board.is_checkmate():
            print("Game ended by: Checkmate")
        elif board.is_stalemate():
            print("Game ended by: Stalemate")
        elif board.is_insufficient_material():
            print("Game ended by: Insufficient material")
        elif board.is_seventyfive_moves():
            print("Game ended by: 75-move rule")
        elif board.is_fivefold_repetition():
            print("Game ended by: Fivefold repetition")
        elif move_count >= 500:
            print("Game ended by: Maximum moves reached")

        print()

    # Salva su file se specificato
    if output_file:
        with open(output_file, 'w') as f:
            print(pgn_game, file=f)
        if verbose:
            print(f"✓ Game saved to: {output_file}")
            print()

    # Stampa PGN
    if verbose:
        print("=" * 70)
        print("PGN FORMAT")
        print("=" * 70)
        print(pgn_game)
        print()

        print("=" * 70)
        print("HOW TO VIEW")
        print("=" * 70)
        print("1. Copy the PGN above (from [Event...] to the last move)")
        print("2. Go to https://lichess.org/paste")
        print("3. Paste the PGN and click 'Import'")
        print("4. You can now view the game with a visual board!")
        print()
        print("Or save to file and open with:")
        print("  - chess.com/analysis")
        print("  - lichess.org/paste")
        print("  - Any chess GUI (Arena, ChessBase, etc.)")
        print("=" * 70)

    return pgn_game


def main():
    parser = argparse.ArgumentParser(description='Generate and view self-play game')
    parser.add_argument('--simulations', type=int, default=50,
                        help='MCTS simulations per move (default: 50)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PGN file (default: print to screen)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: untrained)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity')

    args = parser.parse_args()

    # Gioca e salva partita
    game = play_and_save_game(
        num_simulations=args.simulations,
        output_file=args.output,
        verbose=not args.quiet,
        model_path=args.model
    )

    # Se nessun output file, suggerisci dove salvare
    if not args.output and not args.quiet:
        print("\nTo save this game, run:")
        print(f"  python scripts/demo_selfplay_game.py --output game.pgn")


if __name__ == "__main__":
    main()
