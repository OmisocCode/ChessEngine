"""
Model Evaluation System

Questo modulo implementa il sistema di valutazione per misurare le performance
dell'AI a scacchi. Permette di:

1. **Valutare vs Random Player**: Misura win rate contro mosse casuali
2. **Confrontare Modelli**: Match tra checkpoint diversi (es: iter_5 vs iter_20)
3. **Stimare ELO**: Stima approssimativa del rating ELO
4. **Statistiche Dettagliate**: Win/Draw/Loss, avg game length, tempi, etc.

Il sistema di valutazione è fondamentale per:
- Verificare che il training sta funzionando (win rate aumenta)
- Identificare il miglior checkpoint
- Decidere quando fermare il training
- Validare miglioramenti architetturali

Usage tipico:
    # Valuta modello vs random
    evaluator = ModelEvaluator(model, encoder, decoder)
    results = evaluator.evaluate_vs_random(num_games=100, mcts_sims=50)
    print(f"Win rate: {results['win_rate']:.1%}")

    # Confronta due modelli
    results = evaluator.compare_models(model1, model2, num_games=50)
    print(f"Model 1 won {results['model1_wins']}/{num_games} games")
"""

import chess
import numpy as np
from typing import Dict, Optional, Tuple, Callable
import time
from pathlib import Path
import json

from src.mcts.mcts import MCTS
from src.mcts.evaluator import NeuralNetworkEvaluator, RandomEvaluator


class GameResult:
    """
    Risultato di una singola partita di valutazione.

    Attributes:
        white_player: Nome del giocatore bianco
        black_player: Nome del giocatore nero
        result: Risultato ('1-0', '0-1', '1/2-1/2')
        num_moves: Numero di mosse
        time_taken: Tempo totale partita (secondi)
        termination: Motivo fine partita ('checkmate', 'stalemate', etc.)
    """

    def __init__(
        self,
        white_player: str,
        black_player: str,
        result: str,
        num_moves: int,
        time_taken: float,
        termination: str
    ):
        self.white_player = white_player
        self.black_player = black_player
        self.result = result
        self.num_moves = num_moves
        self.time_taken = time_taken
        self.termination = termination

    def winner(self) -> Optional[str]:
        """Ritorna il nome del vincitore, o None se patta"""
        if self.result == '1-0':
            return self.white_player
        elif self.result == '0-1':
            return self.black_player
        else:
            return None

    def to_dict(self) -> Dict:
        """Converte a dizionario per serializzazione"""
        return {
            'white': self.white_player,
            'black': self.black_player,
            'result': self.result,
            'num_moves': self.num_moves,
            'time_taken': self.time_taken,
            'termination': self.termination
        }


class ModelEvaluator:
    """
    Valutatore per modelli di scacchi.

    Questa classe fornisce metodi per valutare le performance di un modello
    attraverso partite contro baseline (random, altri modelli) e calcolo
    di statistiche aggregate.

    Usage:
        >>> evaluator = ModelEvaluator(model, encoder, decoder)
        >>> results = evaluator.evaluate_vs_random(num_games=100)
        >>> print(f"Win rate: {results['win_rate']:.1%}")
    """

    def __init__(self, encoder, decoder):
        """
        Inizializza il valutatore.

        Args:
            encoder: BoardEncoder instance
            decoder: MoveDecoder instance

        Note: I modelli vengono passati ai metodi di valutazione,
        non al costruttore, per permettere confronti tra modelli diversi.
        """
        self.encoder = encoder
        self.decoder = decoder

    def play_game(
        self,
        white_player_fn: Callable,
        black_player_fn: Callable,
        white_name: str = "White",
        black_name: str = "Black",
        max_moves: int = 500,
        verbose: bool = False
    ) -> GameResult:
        """
        Gioca una singola partita tra due giocatori.

        Questo è il metodo core per evaluation. Gioca una partita completa
        tra due policy functions (che possono essere AI, random, umano, etc.)

        Args:
            white_player_fn: Function che data una board ritorna una mossa
                            Signature: (board: chess.Board) -> chess.Move
            black_player_fn: Function per il nero
            white_name: Nome del giocatore bianco (per logging)
            black_name: Nome del giocatore nero
            max_moves: Numero massimo mosse prima di dichiarare patta
            verbose: Stampa mosse durante la partita

        Returns:
            GameResult object con dettagli della partita

        Example:
            >>> # Crea policy functions
            >>> def ai_policy(board):
            >>>     root = mcts.search(board, evaluator)
            >>>     return root.select_best_move()
            >>>
            >>> def random_policy(board):
            >>>     return np.random.choice(list(board.legal_moves))
            >>>
            >>> result = evaluator.play_game(ai_policy, random_policy)
        """
        board = chess.Board()
        move_count = 0
        start_time = time.time()

        if verbose:
            print(f"\nStarting game: {white_name} (White) vs {black_name} (Black)")

        # Gioca fino a fine partita
        while not board.is_game_over() and move_count < max_moves:
            move_count += 1

            # Seleziona policy function corretta in base al turno
            if board.turn == chess.WHITE:
                player_fn = white_player_fn
                player_name = white_name
            else:
                player_fn = black_player_fn
                player_name = black_name

            # Ottieni mossa dal player
            try:
                move = player_fn(board)

                if verbose:
                    san = board.san(move)
                    print(f"Move {move_count}: {player_name} plays {san}")

                # Esegui mossa
                board.push(move)

            except Exception as e:
                # Se c'è errore, consideriamo il player sconfitto
                if verbose:
                    print(f"Error from {player_name}: {e}")

                # Il player che ha errore perde
                if board.turn == chess.WHITE:
                    result_str = "0-1"  # Black vince
                else:
                    result_str = "1-0"  # White vince

                return GameResult(
                    white_player=white_name,
                    black_player=black_name,
                    result=result_str,
                    num_moves=move_count,
                    time_taken=time.time() - start_time,
                    termination='error'
                )

        elapsed_time = time.time() - start_time

        # Determina risultato e motivo terminazione
        if board.is_checkmate():
            # Il player del turno corrente ha perso (è in matto)
            if board.turn == chess.WHITE:
                result_str = "0-1"
            else:
                result_str = "1-0"
            termination = 'checkmate'

        elif board.is_stalemate():
            result_str = "1/2-1/2"
            termination = 'stalemate'

        elif board.is_insufficient_material():
            result_str = "1/2-1/2"
            termination = 'insufficient_material'

        elif board.is_seventyfive_moves():
            result_str = "1/2-1/2"
            termination = '75_move_rule'

        elif board.is_fivefold_repetition():
            result_str = "1/2-1/2"
            termination = 'fivefold_repetition'

        elif move_count >= max_moves:
            result_str = "1/2-1/2"
            termination = 'max_moves'

        else:
            result_str = "1/2-1/2"
            termination = 'unknown'

        if verbose:
            print(f"\nGame over: {result_str}")
            print(f"Termination: {termination}")
            print(f"Moves: {move_count}, Time: {elapsed_time:.1f}s")

        return GameResult(
            white_player=white_name,
            black_player=black_name,
            result=result_str,
            num_moves=move_count,
            time_taken=elapsed_time,
            termination=termination
        )

    def create_model_policy(
        self,
        model,
        mcts_simulations: int = 50,
        temperature: float = 0.1
    ) -> Callable:
        """
        Crea una policy function da un modello neural network.

        Questa function usa MCTS + neural network per selezionare mosse.

        Args:
            model: ChessNet model
            mcts_simulations: Numero simulazioni MCTS
            temperature: Temperature per sampling (0.1 = quasi greedy)

        Returns:
            Policy function: (board) -> move

        Example:
            >>> policy = evaluator.create_model_policy(model, mcts_sims=100)
            >>> move = policy(board)
        """
        # Crea evaluator e MCTS
        nn_evaluator = NeuralNetworkEvaluator(self.encoder, model, self.decoder)
        mcts = MCTS(num_simulations=mcts_simulations, c_puct=1.5)

        def policy_fn(board: chess.Board) -> chess.Move:
            """Policy che usa MCTS + neural network"""
            root = mcts.search(board, nn_evaluator)
            policy_dict = root.get_policy_distribution(temperature=temperature)

            # Seleziona mossa migliore
            if temperature < 0.5:
                # Greedy: prendi mossa con probabilità massima
                best_move = max(policy_dict.items(), key=lambda x: x[1])[0]
            else:
                # Sample da distribuzione
                moves = list(policy_dict.keys())
                probs = np.array(list(policy_dict.values()))
                probs = probs / probs.sum()
                best_move = np.random.choice(moves, p=probs)

            return best_move

        return policy_fn

    def create_random_policy(self) -> Callable:
        """
        Crea una policy function che gioca mosse casuali.

        Returns:
            Policy function che seleziona move random tra le legali
        """
        def policy_fn(board: chess.Board) -> chess.Move:
            """Policy random"""
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves)

        return policy_fn

    def evaluate_vs_random(
        self,
        model,
        num_games: int = 100,
        mcts_simulations: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Valuta modello contro random player.

        Gioca num_games partite contro un player che fa mosse casuali.
        Metà partite come bianco, metà come nero.

        Args:
            model: ChessNet model da valutare
            num_games: Numero totale partite (deve essere pari)
            mcts_simulations: Simulazioni MCTS per il modello
            verbose: Stampa progresso

        Returns:
            Dictionary con statistiche:
            - 'total_games': Numero partite giocate
            - 'wins': Vittorie del modello
            - 'draws': Patte
            - 'losses': Sconfitte del modello
            - 'win_rate': Percentuale vittorie
            - 'avg_moves': Lunghezza media partite
            - 'avg_time': Tempo medio per partita
            - 'games': Lista di GameResult objects

        Example:
            >>> results = evaluator.evaluate_vs_random(model, num_games=100)
            >>> print(f"Win rate: {results['win_rate']:.1%}")
            >>> print(f"Record: {results['wins']}-{results['draws']}-{results['losses']}")
        """
        if verbose:
            print("="*70)
            print("EVALUATION vs RANDOM PLAYER")
            print("="*70)
            print(f"Total games: {num_games}")
            print(f"MCTS simulations: {mcts_simulations}")
            print()

        # Crea policy functions
        model_policy = self.create_model_policy(model, mcts_simulations, temperature=0.1)
        random_policy = self.create_random_policy()

        # Storage risultati
        games_results = []
        wins = 0
        draws = 0
        losses = 0

        # Gioca partite (metà come bianco, metà come nero)
        for game_idx in range(num_games):
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Playing game {game_idx + 1}/{num_games}...")

            # Alterna colori
            if game_idx % 2 == 0:
                # Modello gioca come bianco
                result = self.play_game(
                    white_player_fn=model_policy,
                    black_player_fn=random_policy,
                    white_name="Model",
                    black_name="Random",
                    verbose=False
                )
            else:
                # Modello gioca come nero
                result = self.play_game(
                    white_player_fn=random_policy,
                    black_player_fn=model_policy,
                    white_name="Random",
                    black_name="Model",
                    verbose=False
                )

            games_results.append(result)

            # Conta vittorie
            winner = result.winner()
            if winner == "Model":
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

        # Calcola statistiche
        win_rate = wins / num_games
        avg_moves = np.mean([r.num_moves for r in games_results])
        avg_time = np.mean([r.time_taken for r in games_results])

        # Conta terminazioni
        termination_counts = {}
        for result in games_results:
            term = result.termination
            termination_counts[term] = termination_counts.get(term, 0) + 1

        if verbose:
            print()
            print("="*70)
            print("RESULTS")
            print("="*70)
            print(f"Record: {wins}W - {draws}D - {losses}L")
            print(f"Win rate: {win_rate:.1%}")
            print(f"Average moves: {avg_moves:.1f}")
            print(f"Average time per game: {avg_time:.1f}s")
            print()
            print("Terminations:")
            for term, count in sorted(termination_counts.items()):
                print(f"  {term}: {count}")
            print("="*70)

        return {
            'total_games': num_games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate,
            'avg_moves': avg_moves,
            'avg_time': avg_time,
            'termination_counts': termination_counts,
            'games': games_results
        }

    def compare_models(
        self,
        model1,
        model2,
        num_games: int = 50,
        mcts_simulations: int = 50,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        verbose: bool = True
    ) -> Dict:
        """
        Confronta due modelli giocando partite tra loro.

        Args:
            model1: Primo modello (ChessNet)
            model2: Secondo modello (ChessNet)
            num_games: Numero partite (deve essere pari)
            mcts_simulations: Simulazioni MCTS per entrambi
            model1_name: Nome del primo modello (per display)
            model2_name: Nome del secondo modello
            verbose: Stampa progresso

        Returns:
            Dictionary con risultati del confronto:
            - 'model1_wins': Vittorie model1
            - 'model2_wins': Vittorie model2
            - 'draws': Patte
            - 'model1_win_rate': Win rate model1
            - 'games': Lista GameResult objects

        Example:
            >>> # Confronta checkpoint diversi
            >>> model_old = load_model('iter_5.pt')
            >>> model_new = load_model('iter_20.pt')
            >>> results = evaluator.compare_models(
            >>>     model_old, model_new,
            >>>     num_games=50,
            >>>     model1_name="Iteration 5",
            >>>     model2_name="Iteration 20"
            >>> )
            >>> print(f"{model2_name} won {results['model2_wins']}/{num_games}")
        """
        if verbose:
            print("="*70)
            print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
            print("="*70)
            print(f"Total games: {num_games}")
            print(f"MCTS simulations: {mcts_simulations}")
            print()

        # Crea policy functions
        policy1 = self.create_model_policy(model1, mcts_simulations, temperature=0.1)
        policy2 = self.create_model_policy(model2, mcts_simulations, temperature=0.1)

        # Storage risultati
        games_results = []
        model1_wins = 0
        model2_wins = 0
        draws = 0

        # Gioca partite
        for game_idx in range(num_games):
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Playing game {game_idx + 1}/{num_games}...")

            # Alterna colori per fairness
            if game_idx % 2 == 0:
                # Model1 bianco, Model2 nero
                result = self.play_game(
                    white_player_fn=policy1,
                    black_player_fn=policy2,
                    white_name=model1_name,
                    black_name=model2_name,
                    verbose=False
                )
            else:
                # Model2 bianco, Model1 nero
                result = self.play_game(
                    white_player_fn=policy2,
                    black_player_fn=policy1,
                    white_name=model2_name,
                    black_name=model1_name,
                    verbose=False
                )

            games_results.append(result)

            # Conta vittorie
            winner = result.winner()
            if winner == model1_name:
                model1_wins += 1
            elif winner == model2_name:
                model2_wins += 1
            else:
                draws += 1

        # Statistiche
        model1_win_rate = model1_wins / num_games
        model2_win_rate = model2_wins / num_games

        if verbose:
            print()
            print("="*70)
            print("RESULTS")
            print("="*70)
            print(f"{model1_name}: {model1_wins}W - {draws}D - {model2_wins}L ({model1_win_rate:.1%})")
            print(f"{model2_name}: {model2_wins}W - {draws}D - {model1_wins}L ({model2_win_rate:.1%})")
            print()
            if model1_wins > model2_wins:
                print(f"Winner: {model1_name}")
            elif model2_wins > model1_wins:
                print(f"Winner: {model2_name}")
            else:
                print("Tie!")
            print("="*70)

        return {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'total_games': num_games,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'model1_win_rate': model1_win_rate,
            'model2_win_rate': model2_win_rate,
            'games': games_results
        }

    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Salva risultati evaluation in file JSON.

        Args:
            results: Dictionary da evaluate_vs_random o compare_models
            filepath: Path dove salvare
        """
        # Rimuovi games objects (non JSON serializable) e salva solo stats
        results_copy = results.copy()
        if 'games' in results_copy:
            results_copy['games'] = [g.to_dict() for g in results_copy['games']]

        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)

        print(f"✓ Results saved to: {filepath}")


def estimate_elo(win_rate: float, opponent_elo: int = 800) -> int:
    """
    Stima approssimativa dell'ELO rating basata su win rate.

    Formula ELO semplificata:
    ELO_diff = -400 * log10((1 - win_rate) / win_rate)
    ELO = opponent_ELO + ELO_diff

    Args:
        win_rate: Win rate contro opponent (0.0 - 1.0)
        opponent_elo: ELO stimato dell'avversario (default: 800 per random)

    Returns:
        ELO rating stimato

    Example:
        >>> elo = estimate_elo(win_rate=0.85, opponent_elo=800)
        >>> print(f"Estimated ELO: {elo}")

    Note: Questa è una stima molto approssimativa. Per ELO accurato
    servono match contro avversari con ELO noto e calibrato.
    """
    # Evita divisione per zero
    if win_rate >= 0.99:
        win_rate = 0.99
    elif win_rate <= 0.01:
        win_rate = 0.01

    # Formula ELO
    elo_diff = -400 * np.log10((1 - win_rate) / win_rate)
    estimated_elo = opponent_elo + elo_diff

    return int(estimated_elo)


if __name__ == "__main__":
    """Demo: evaluation system"""
    print("Model Evaluation Demo")
    print("="*70)

    from src.game.encoder import BoardEncoder
    from src.game.decoder import MoveDecoder
    from src.models.chess_net import ChessNet

    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    model = ChessNet()  # Untrained

    print("Creating evaluator...")
    evaluator = ModelEvaluator(encoder, decoder)

    print("\nRunning quick evaluation (5 games vs random)...")
    results = evaluator.evaluate_vs_random(
        model,
        num_games=6,  # Pari
        mcts_simulations=20,  # Poche sim per demo veloce
        verbose=True
    )

    # Stima ELO
    if results['win_rate'] > 0:
        elo = estimate_elo(results['win_rate'], opponent_elo=800)
        print(f"\nEstimated ELO: ~{elo}")
    else:
        print("\nNo wins - model needs training!")

    print("\n" + "="*70)
    print("Demo completed!")
    print("\nTo use with trained model:")
    print("  1. Train model: python scripts/train.py")
    print("  2. Evaluate: python -m src.evaluation.evaluator")
