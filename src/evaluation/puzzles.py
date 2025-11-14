"""
Tactical Puzzle Evaluation System

Questo modulo implementa il testing su puzzle tattici per valutare
le capacità tattiche dell'AI.

Un puzzle tattico è una posizione scacchistica dove c'è una "soluzione migliore" -
tipicamente una sequenza di mosse che porta a vantaggio materiale decisivo
o scacco matto.

Tipi di puzzle supportati:
- **Mate in 2**: Scacco matto in 2 mosse (1 mossa + risposta + matto)
- **Mate in 3**: Scacco matto in 3 mosse
- **Tactical win**: Vantaggio materiale decisivo (es: vinci donna)

Il sistema carica puzzle da:
1. **Puzzle integrati**: Set di base hardcoded nel modulo
2. **File CSV/JSON**: Puzzle personalizzati in formato standard

Formato puzzle:
    {
        'id': 'mate_in_2_001',
        'fen': 'FEN string della posizione',
        'solution': ['e4e5', 'f7f8q'],  # Mosse in UCI notation
        'category': 'mate_in_2',
        'difficulty': 'easy'
    }

Usage:
    >>> tester = PuzzleTester(model, encoder, decoder)
    >>> results = tester.test_puzzles(puzzle_set='mate_in_2', mcts_sims=100)
    >>> print(f"Solved: {results['solved']}/{results['total']}")
"""

import chess
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class Puzzle:
    """
    Rappresenta un singolo puzzle tattico.

    Attributes:
        puzzle_id: ID univoco del puzzle
        fen: Posizione iniziale in FEN notation
        solution: Lista di mosse soluzione in UCI notation
        category: Categoria puzzle ('mate_in_2', 'mate_in_3', etc.)
        difficulty: Difficoltà ('easy', 'medium', 'hard')
        description: Descrizione opzionale
    """

    def __init__(
        self,
        puzzle_id: str,
        fen: str,
        solution: List[str],
        category: str = 'tactical',
        difficulty: str = 'medium',
        description: str = ''
    ):
        self.puzzle_id = puzzle_id
        self.fen = fen
        self.solution = solution  # UCI moves: ['e2e4', 'e7e5']
        self.category = category
        self.difficulty = difficulty
        self.description = description

    def get_board(self) -> chess.Board:
        """Crea board dalla FEN del puzzle"""
        return chess.Board(self.fen)

    def is_solution_move(self, move: chess.Move, move_index: int = 0) -> bool:
        """
        Verifica se una mossa è la soluzione corretta.

        Args:
            move: Mossa da verificare
            move_index: Indice nella sequenza soluzione (default: 0 = prima mossa)

        Returns:
            True se la mossa corrisponde alla soluzione
        """
        if move_index >= len(self.solution):
            return False

        expected_uci = self.solution[move_index]
        return move.uci() == expected_uci

    def to_dict(self) -> Dict:
        """Converte a dizionario per serializzazione"""
        return {
            'id': self.puzzle_id,
            'fen': self.fen,
            'solution': self.solution,
            'category': self.category,
            'difficulty': self.difficulty,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Puzzle':
        """Crea Puzzle da dizionario"""
        return cls(
            puzzle_id=data['id'],
            fen=data['fen'],
            solution=data['solution'],
            category=data.get('category', 'tactical'),
            difficulty=data.get('difficulty', 'medium'),
            description=data.get('description', '')
        )


# ============================================================================
# BUILT-IN PUZZLE SETS
# ============================================================================

MATE_IN_2_PUZZLES = [
    # Puzzle 1: Back rank mate
    Puzzle(
        puzzle_id='mate2_001',
        fen='6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1',
        solution=['f1f8'],  # Rf8#
        category='mate_in_2',
        difficulty='easy',
        description='Back rank mate with rook'
    ),

    # Puzzle 2: Queen and rook mate
    Puzzle(
        puzzle_id='mate2_002',
        fen='r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1',
        solution=['h5f7'],  # Qxf7#
        category='mate_in_2',
        difficulty='easy',
        description='Scholar\'s mate pattern'
    ),

    # Puzzle 3: Smothered mate setup
    Puzzle(
        puzzle_id='mate2_003',
        fen='6k1/5ppp/5n2/8/8/8/5PPP/4R1K1 w - - 0 1',
        solution=['e1e8'],  # Re8#
        category='mate_in_2',
        difficulty='medium',
        description='Back rank mate with rook'
    ),

    # Puzzle 4: Queen mate in corner
    Puzzle(
        puzzle_id='mate2_004',
        fen='7k/5Qpp/6p1/8/8/8/5PPP/6K1 w - - 0 1',
        solution=['f7f8'],  # Qf8#
        category='mate_in_2',
        difficulty='easy',
        description='Queen mate in corner'
    ),

    # Puzzle 5: Two rooks mate
    Puzzle(
        puzzle_id='mate2_005',
        fen='6k1/5ppp/8/8/8/8/R4PPP/R5K1 w - - 0 1',
        solution=['a2a8'],  # Ra8#
        category='mate_in_2',
        difficulty='easy',
        description='Back rank mate with two rooks'
    ),

    # Puzzle 6: Bishop and queen mate
    Puzzle(
        puzzle_id='mate2_006',
        fen='r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1',
        solution=[],  # Già matto!
        category='mate_in_2',
        difficulty='easy',
        description='Already checkmate - Scholar\'s mate'
    ),

    # Puzzle 7: Knight and queen mate
    Puzzle(
        puzzle_id='mate2_007',
        fen='r1bq1rk1/ppp2ppp/2np1n2/2b1p2Q/2B1P3/2NP1N2/PPP2PPP/R1B2RK1 w - - 0 1',
        solution=['h5h7'],  # Qxh7#
        category='mate_in_2',
        difficulty='medium',
        description='Greek gift sacrifice pattern'
    ),

    # Puzzle 8: Rook and king mate
    Puzzle(
        puzzle_id='mate2_008',
        fen='8/8/8/8/8/3k4/3R4/3K4 w - - 0 1',
        solution=['d2d3'],  # Rd3#
        category='mate_in_2',
        difficulty='easy',
        description='Simple rook mate'
    ),

    # Puzzle 9: Queen mate
    Puzzle(
        puzzle_id='mate2_009',
        fen='5rk1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1',
        solution=['e2e8'],  # Qe8#
        category='mate_in_2',
        difficulty='easy',
        description='Queen back rank mate'
    ),

    # Puzzle 10: Anastasia's mate pattern
    Puzzle(
        puzzle_id='mate2_010',
        fen='5rk1/Q4ppp/8/8/8/8/5PPP/6K1 w - - 0 1',
        solution=['a7a8'],  # Qa8#
        category='mate_in_2',
        difficulty='medium',
        description='Back rank queen mate'
    ),
]


class PuzzleTester:
    """
    Tester per puzzle tattici.

    Questa classe testa un modello su un set di puzzle tattici e
    calcola statistiche (accuracy, puzzle risolti, tempo, etc.)

    Usage:
        >>> tester = PuzzleTester(model, encoder, decoder)
        >>> results = tester.test_puzzles(puzzles=MATE_IN_2_PUZZLES)
        >>> print(f"Accuracy: {results['accuracy']:.1%}")
    """

    def __init__(self, encoder, decoder):
        """
        Inizializza puzzle tester.

        Args:
            encoder: BoardEncoder instance
            decoder: MoveDecoder instance
        """
        self.encoder = encoder
        self.decoder = decoder

    def test_single_puzzle(
        self,
        puzzle: Puzzle,
        model,
        mcts_simulations: int = 100,
        verbose: bool = False
    ) -> Dict:
        """
        Testa il modello su un singolo puzzle.

        Args:
            puzzle: Puzzle object da testare
            model: ChessNet model
            mcts_simulations: Numero simulazioni MCTS
            verbose: Stampa dettagli

        Returns:
            Dictionary con risultato:
            - 'puzzle_id': ID del puzzle
            - 'solved': True se trovata soluzione corretta
            - 'ai_move': Mossa scelta dall'AI (UCI)
            - 'correct_move': Mossa soluzione corretta (UCI)
            - 'matches': True se ai_move == correct_move

        Note: Verifica solo la PRIMA mossa della soluzione.
        Per puzzle multi-move servirebbero verifiche più complesse.
        """
        if verbose:
            print(f"\nTesting puzzle: {puzzle.puzzle_id}")
            print(f"Category: {puzzle.category}")
            print(f"Description: {puzzle.description}")
            print(f"FEN: {puzzle.fen}")

        # Crea board dalla FEN
        board = puzzle.get_board()

        if verbose:
            print(f"Position: {board}")

        # Verifica che ci sia una soluzione
        if len(puzzle.solution) == 0:
            # Puzzle speciale (già matto, etc.)
            if verbose:
                print("Puzzle has no moves (special case)")
            return {
                'puzzle_id': puzzle.puzzle_id,
                'solved': board.is_checkmate(),  # True se già matto
                'ai_move': None,
                'correct_move': None,
                'matches': board.is_checkmate()
            }

        # Usa MCTS per trovare mossa
        from src.mcts.mcts import MCTS
        from src.mcts.evaluator import NeuralNetworkEvaluator

        evaluator = NeuralNetworkEvaluator(self.encoder, model, self.decoder)
        mcts = MCTS(num_simulations=mcts_simulations, c_puct=1.5)

        # Search con MCTS
        root = mcts.search(board, evaluator)

        # Ottieni mossa migliore (greedy, no sampling)
        policy_dict = root.get_policy_distribution(temperature=0.0)
        best_move = max(policy_dict.items(), key=lambda x: x[1])[0]

        # Verifica se corretta
        correct_move_uci = puzzle.solution[0]
        ai_move_uci = best_move.uci()
        is_correct = (ai_move_uci == correct_move_uci)

        if verbose:
            print(f"AI move: {ai_move_uci}")
            print(f"Correct move: {correct_move_uci}")
            print(f"Result: {'✓ SOLVED' if is_correct else '✗ FAILED'}")

        return {
            'puzzle_id': puzzle.puzzle_id,
            'solved': is_correct,
            'ai_move': ai_move_uci,
            'correct_move': correct_move_uci,
            'matches': is_correct,
            'category': puzzle.category,
            'difficulty': puzzle.difficulty
        }

    def test_puzzles(
        self,
        puzzles: List[Puzzle],
        model,
        mcts_simulations: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Testa il modello su una lista di puzzle.

        Args:
            puzzles: Lista di Puzzle objects
            model: ChessNet model
            mcts_simulations: Simulazioni MCTS per puzzle
            verbose: Stampa progresso

        Returns:
            Dictionary con statistiche:
            - 'total': Numero totale puzzle testati
            - 'solved': Numero puzzle risolti
            - 'accuracy': Percentuale risolti (0.0 - 1.0)
            - 'by_category': Accuracy per categoria
            - 'by_difficulty': Accuracy per difficoltà
            - 'results': Lista risultati individuali

        Example:
            >>> results = tester.test_puzzles(MATE_IN_2_PUZZLES, model)
            >>> print(f"Solved {results['solved']}/{results['total']} puzzles")
            >>> print(f"Accuracy: {results['accuracy']:.1%}")
        """
        if verbose:
            print("="*70)
            print("PUZZLE TESTING")
            print("="*70)
            print(f"Total puzzles: {len(puzzles)}")
            print(f"MCTS simulations: {mcts_simulations}")
            print()

        results_list = []
        solved_count = 0

        # Testa ogni puzzle
        for idx, puzzle in enumerate(puzzles):
            if verbose:
                print(f"[{idx+1}/{len(puzzles)}] Testing {puzzle.puzzle_id}...", end=" ")

            result = self.test_single_puzzle(
                puzzle, model, mcts_simulations, verbose=False
            )

            results_list.append(result)

            if result['solved']:
                solved_count += 1
                if verbose:
                    print("✓ SOLVED")
            else:
                if verbose:
                    print(f"✗ FAILED (AI: {result['ai_move']}, Correct: {result['correct_move']})")

        # Calcola statistiche aggregate
        total = len(puzzles)
        accuracy = solved_count / total if total > 0 else 0

        # Statistiche per categoria
        by_category = {}
        for result in results_list:
            cat = result['category']
            if cat not in by_category:
                by_category[cat] = {'total': 0, 'solved': 0}
            by_category[cat]['total'] += 1
            if result['solved']:
                by_category[cat]['solved'] += 1

        # Calcola accuracy per categoria
        for cat in by_category:
            total_cat = by_category[cat]['total']
            solved_cat = by_category[cat]['solved']
            by_category[cat]['accuracy'] = solved_cat / total_cat if total_cat > 0 else 0

        # Statistiche per difficoltà
        by_difficulty = {}
        for result in results_list:
            diff = result['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = {'total': 0, 'solved': 0}
            by_difficulty[diff]['total'] += 1
            if result['solved']:
                by_difficulty[diff]['solved'] += 1

        # Calcola accuracy per difficoltà
        for diff in by_difficulty:
            total_diff = by_difficulty[diff]['total']
            solved_diff = by_difficulty[diff]['solved']
            by_difficulty[diff]['accuracy'] = solved_diff / total_diff if total_diff > 0 else 0

        if verbose:
            print()
            print("="*70)
            print("RESULTS")
            print("="*70)
            print(f"Total puzzles: {total}")
            print(f"Solved: {solved_count}")
            print(f"Failed: {total - solved_count}")
            print(f"Accuracy: {accuracy:.1%}")
            print()

            if by_category:
                print("By category:")
                for cat, stats in by_category.items():
                    print(f"  {cat}: {stats['solved']}/{stats['total']} ({stats['accuracy']:.1%})")
                print()

            if by_difficulty:
                print("By difficulty:")
                for diff, stats in by_difficulty.items():
                    print(f"  {diff}: {stats['solved']}/{stats['total']} ({stats['accuracy']:.1%})")

            print("="*70)

        return {
            'total': total,
            'solved': solved_count,
            'failed': total - solved_count,
            'accuracy': accuracy,
            'by_category': by_category,
            'by_difficulty': by_difficulty,
            'results': results_list
        }

    def load_puzzles_from_file(self, filepath: str) -> List[Puzzle]:
        """
        Carica puzzle da file JSON.

        Il file deve contenere una lista di puzzle in formato:
        [
            {
                "id": "puzzle_001",
                "fen": "FEN string",
                "solution": ["e2e4", "e7e5"],
                "category": "mate_in_2",
                "difficulty": "easy"
            },
            ...
        ]

        Args:
            filepath: Path al file JSON

        Returns:
            Lista di Puzzle objects
        """
        with open(filepath, 'r') as f:
            puzzle_data = json.load(f)

        puzzles = [Puzzle.from_dict(p) for p in puzzle_data]
        return puzzles

    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Salva risultati test in file JSON.

        Args:
            results: Dictionary da test_puzzles
            filepath: Path dove salvare
        """
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to: {filepath}")


def get_builtin_puzzle_set(name: str) -> List[Puzzle]:
    """
    Ottieni set di puzzle integrato per nome.

    Args:
        name: Nome del set ('mate_in_2', 'all')

    Returns:
        Lista di Puzzle objects

    Example:
        >>> puzzles = get_builtin_puzzle_set('mate_in_2')
        >>> print(f"Loaded {len(puzzles)} puzzles")
    """
    if name == 'mate_in_2':
        return MATE_IN_2_PUZZLES
    elif name == 'all':
        return MATE_IN_2_PUZZLES  # Espandi quando aggiungi altri set
    else:
        raise ValueError(f"Unknown puzzle set: {name}")


if __name__ == "__main__":
    """Demo: puzzle testing"""
    print("Puzzle Testing Demo")
    print("="*70)

    from src.game.encoder import BoardEncoder
    from src.game.decoder import MoveDecoder
    from src.models.chess_net import ChessNet

    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    model = ChessNet()  # Untrained

    print("Creating puzzle tester...")
    tester = PuzzleTester(encoder, decoder)

    print(f"\nLoading {len(MATE_IN_2_PUZZLES)} mate-in-2 puzzles...")
    puzzles = get_builtin_puzzle_set('mate_in_2')

    print("\nTesting first 3 puzzles (quick demo)...")
    results = tester.test_puzzles(
        puzzles[:3],
        model,
        mcts_simulations=50,  # Poche per demo veloce
        verbose=True
    )

    print("\n" + "="*70)
    print("Demo completed!")
    print("\nNote: Untrained model will likely fail most puzzles.")
    print("Train model first, then test puzzles for better results.")
    print("\nTo test with trained model:")
    print("  python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt")
