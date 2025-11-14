"""
Self-Play Game Generation for Chess AI Training

This module implements self-play game generation, which is the core of
AlphaZero-style training. The AI plays against itself using MCTS to generate
high-quality training data.

The self-play process:
1. Start with a new game (initial position)
2. For each move:
   a. Run MCTS search to get improved policy
   b. Sample move from MCTS policy (with temperature)
   c. Record training example: (board, mcts_policy, outcome)
   d. Make the move
3. Continue until game ends
4. Assign final outcome to all positions (backfill value labels)
5. Return training data for neural network

Key concepts:
- **MCTS-improved policy**: The visit distribution from MCTS is better than
  the raw neural network policy, so we use it as the training target
- **Temperature schedule**: High temperature (1.0) for first N moves to ensure
  diversity, then low temperature (0.1) for optimal play
- **Data augmentation**: Can apply board symmetries (not implemented yet)
- **Outcome backfill**: The final game result becomes the value target for
  all positions in the game
"""

import chess
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import time
from dataclasses import dataclass


@dataclass
class TrainingExample:
    """
    Single training example from self-play.

    This represents one position from a self-play game, along with the
    improved policy from MCTS and the final game outcome.

    Attributes:
        board_tensor: Encoded board position (18, 8, 8)
                     This is the input to the neural network
        mcts_policy: MCTS-improved policy vector (4672,)
                    This is the target for policy head training
                    Each element is the visit probability for a move
        value: Game outcome from this position's perspective
              +1.0 = this side won
              0.0 = draw
              -1.0 = this side lost
              This is the target for value head training
        move_number: Which move in the game (for analysis)

    The neural network learns to predict:
    - Policy: What move would MCTS choose? (imitate MCTS)
    - Value: What is the outcome of this position? (predict winner)
    """
    board_tensor: np.ndarray  # Shape: (18, 8, 8)
    mcts_policy: np.ndarray   # Shape: (4672,)
    value: float              # Scalar in [-1, 1]
    move_number: int          # For tracking/analysis


class SelfPlayGame:
    """
    Manager for a single self-play game.

    This class orchestrates one complete game where the AI plays against itself.
    It uses MCTS to select moves and collects training data at each step.

    The self-play game process:
    1. Start from initial position (or custom position)
    2. Loop until game ends:
       - Run MCTS to get improved policy
       - Sample move from policy (with temperature)
       - Record (board, policy) for training
       - Make move
    3. When game ends, backfill outcomes to all positions
    4. Return training examples

    Attributes:
        encoder: BoardEncoder to convert positions to tensors
        decoder: MoveDecoder to convert policies to move vectors
        mcts: MCTS instance for move selection
        evaluator: Neural network evaluator for MCTS
        temperature_schedule: Function that returns temperature for a given move number
    """

    def __init__(
        self,
        encoder,
        decoder,
        mcts,
        evaluator,
        temperature_threshold: int = 30,
        high_temperature: float = 1.0,
        low_temperature: float = 0.1
    ):
        """
        Initialize self-play game manager.

        Args:
            encoder: BoardEncoder instance
            decoder: MoveDecoder instance
            mcts: MCTS instance with num_simulations configured
            evaluator: NeuralNetworkEvaluator instance
            temperature_threshold: Move number to switch from high to low temp
                                  High temp for first N moves ensures diversity
                                  Low temp afterwards for optimal play
            high_temperature: Temperature for early moves (default: 1.0)
                            1.0 = sample proportional to visit counts
            low_temperature: Temperature for later moves (default: 0.1)
                           →0 = greedy, always pick most visited move

        Example:
            >>> encoder = BoardEncoder()
            >>> decoder = MoveDecoder()
            >>> mcts = MCTS(num_simulations=100)
            >>> evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
            >>> game = SelfPlayGame(encoder, decoder, mcts, evaluator)
            >>> examples = game.play_game()
        """
        self.encoder = encoder
        self.decoder = decoder
        self.mcts = mcts
        self.evaluator = evaluator
        self.temperature_threshold = temperature_threshold
        self.high_temperature = high_temperature
        self.low_temperature = low_temperature

    def get_temperature(self, move_number: int) -> float:
        """
        Get temperature for current move number.

        Temperature schedule:
        - First 30 moves: temp = 1.0 (diverse, exploratory play)
        - After move 30: temp = 0.1 (deterministic, optimal play)

        This ensures:
        - Training data has diverse positions (not always same opening)
        - Games still finish optimally (not random endgames)

        Args:
            move_number: Current move number (starts at 1)

        Returns:
            Temperature value for move selection
        """
        if move_number <= self.temperature_threshold:
            return self.high_temperature
        else:
            return self.low_temperature

    def play_game(
        self,
        starting_position: Optional[chess.Board] = None,
        max_moves: int = 500
    ) -> Tuple[List[TrainingExample], Dict]:
        """
        Play one complete self-play game and collect training data.

        This is the main method that generates training examples.
        It plays a full game, collecting (board, mcts_policy) pairs,
        then assigns the final outcome as the value target.

        Args:
            starting_position: Optional starting position (default: standard start)
            max_moves: Maximum moves before declaring draw (default: 500)

        Returns:
            Tuple of (training_examples, game_info):
            - training_examples: List of TrainingExample objects
            - game_info: Dictionary with game statistics
              - 'result': Game result string
              - 'num_moves': Total moves in game
              - 'outcome': Final outcome (+1/0/-1)
              - 'time_taken': Total time for game

        The training examples are ready to be added to the replay buffer
        and used for neural network training.

        Example:
            >>> game = SelfPlayGame(encoder, decoder, mcts, evaluator)
            >>> examples, info = game.play_game()
            >>> print(f"Generated {len(examples)} training examples")
            >>> print(f"Game result: {info['result']}")
        """
        # Initialize game
        if starting_position is None:
            board = chess.Board()
        else:
            board = starting_position.copy()

        # Storage for training data (before outcome is known)
        # We store (board_tensor, mcts_policy_vector, player_color) tuples
        # The value will be filled in later based on game outcome
        game_history = []

        # Game statistics
        move_count = 0
        start_time = time.time()

        # Play game until it ends
        while not board.is_game_over() and move_count < max_moves:
            move_count += 1

            # Get current player (for value assignment later)
            current_player = board.turn

            # Encode current position
            board_tensor = self.encoder.encode(board)

            # Get temperature for this move
            temperature = self.get_temperature(move_count)

            # Run MCTS to get improved policy
            # This is the key step: MCTS search improves the raw neural network policy
            root = self.mcts.search(board, self.evaluator)

            # Get MCTS visit distribution as policy
            # This policy is better than the raw NN policy because it includes search
            mcts_policy_dict = root.get_policy_distribution(temperature=temperature)

            # Convert policy dictionary to full 4672-dimensional vector
            # Most entries will be 0 (illegal moves), legal moves have probabilities
            mcts_policy_vector = self._policy_dict_to_vector(mcts_policy_dict, board)

            # Store this position and policy for training
            # Note: We don't know the value yet (game hasn't ended)
            game_history.append({
                'board_tensor': board_tensor,
                'mcts_policy_vector': mcts_policy_vector,
                'player': current_player,
                'move_number': move_count
            })

            # Select move from MCTS policy
            # With high temperature: sample stochastically (diversity)
            # With low temperature: pick best move (optimal play)
            if temperature > 0.5:
                # Sample from distribution
                moves = list(mcts_policy_dict.keys())
                probs = np.array(list(mcts_policy_dict.values()))
                # Ensure normalization
                probs = probs / probs.sum()
                chosen_move = np.random.choice(moves, p=probs)
            else:
                # Greedy: pick most visited
                chosen_move = max(mcts_policy_dict.items(), key=lambda x: x[1])[0]

            # Make the move
            board.push(chosen_move)

        # Game has ended - determine outcome
        elapsed_time = time.time() - start_time

        if board.is_game_over():
            outcome = board.outcome()

            # Determine result string
            if outcome.winner == chess.WHITE:
                result_str = "1-0"
                white_value = 1.0
                black_value = -1.0
            elif outcome.winner == chess.BLACK:
                result_str = "0-1"
                white_value = -1.0
                black_value = 1.0
            else:
                result_str = "1/2-1/2"
                white_value = 0.0
                black_value = 0.0
        else:
            # Max moves reached - declare draw
            result_str = "1/2-1/2 (max moves)"
            white_value = 0.0
            black_value = 0.0

        # Create training examples with outcome filled in
        # IMPORTANT: The value is from the perspective of the player to move
        training_examples = []

        for position_data in game_history:
            # Get value from this player's perspective
            if position_data['player'] == chess.WHITE:
                value = white_value
            else:
                value = black_value

            # Create training example
            example = TrainingExample(
                board_tensor=position_data['board_tensor'],
                mcts_policy=position_data['mcts_policy_vector'],
                value=value,
                move_number=position_data['move_number']
            )

            training_examples.append(example)

        # Compile game info
        game_info = {
            'result': result_str,
            'num_moves': move_count,
            'outcome': white_value,  # From white's perspective
            'time_taken': elapsed_time,
            'avg_time_per_move': elapsed_time / move_count if move_count > 0 else 0
        }

        return training_examples, game_info

    def _policy_dict_to_vector(
        self,
        policy_dict: Dict[chess.Move, float],
        board: chess.Board
    ) -> np.ndarray:
        """
        Convert policy dictionary to full 4672-dimensional policy vector.

        The policy_dict only contains legal moves with their probabilities.
        We need to convert this to a full vector where:
        - Legal move positions have their probabilities
        - Illegal move positions have 0

        Args:
            policy_dict: Dictionary mapping moves to probabilities
            board: Current board position (needed for move encoding)

        Returns:
            Policy vector of shape (4672,) with probabilities

        Note: The returned vector should sum to 1.0 (or very close)
        """
        # Initialize zero vector
        policy_vector = np.zeros(4672, dtype=np.float32)

        # Fill in probabilities for legal moves
        for move, prob in policy_dict.items():
            # Get policy index for this move using the decoder's lookup table
            # The key format is (from_square, to_square, promotion)
            move_key = (move.from_square, move.to_square, move.promotion)
            policy_idx = self.decoder.move_to_policy_index.get(move_key)

            if policy_idx is not None:
                policy_vector[policy_idx] = prob

        return policy_vector


class SelfPlayWorker:
    """
    Worker that generates multiple self-play games.

    This class manages the generation of many self-play games for training.
    It can run games sequentially or (in future) in parallel.

    Usage in training loop:
    1. Create worker with current neural network
    2. Generate N games
    3. Collect all training examples
    4. Add to replay buffer
    5. Train network on batch from replay buffer
    6. Repeat

    Attributes:
        encoder: BoardEncoder instance
        decoder: MoveDecoder instance
        model: Neural network (ChessNet)
        mcts_config: Configuration for MCTS (num_simulations, etc.)
    """

    def __init__(
        self,
        encoder,
        decoder,
        model,
        mcts_config: Optional[Dict] = None
    ):
        """
        Initialize self-play worker.

        Args:
            encoder: BoardEncoder instance
            decoder: MoveDecoder instance
            model: ChessNet neural network
            mcts_config: Dictionary with MCTS parameters:
                - num_simulations: Number of MCTS simulations per move
                - c_puct: Exploration constant
                - temperature_threshold: Move to switch temperature
                - high_temperature: Temperature for early moves
                - low_temperature: Temperature for late moves

        Example:
            >>> worker = SelfPlayWorker(
            >>>     encoder, decoder, model,
            >>>     mcts_config={'num_simulations': 100, 'c_puct': 1.5}
            >>> )
            >>> examples = worker.generate_games(num_games=50)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.model = model

        # Default MCTS configuration
        if mcts_config is None:
            mcts_config = {}

        self.num_simulations = mcts_config.get('num_simulations', 50)
        self.c_puct = mcts_config.get('c_puct', 1.5)
        self.temperature_threshold = mcts_config.get('temperature_threshold', 30)
        self.high_temperature = mcts_config.get('high_temperature', 1.0)
        self.low_temperature = mcts_config.get('low_temperature', 0.1)

    def generate_games(
        self,
        num_games: int,
        verbose: bool = True
    ) -> Tuple[List[TrainingExample], List[Dict]]:
        """
        Generate multiple self-play games.

        This is the main method for generating training data.
        It plays num_games complete games and collects all training examples.

        Args:
            num_games: Number of games to generate
            verbose: Whether to print progress

        Returns:
            Tuple of (all_examples, game_infos):
            - all_examples: List of all TrainingExample objects from all games
            - game_infos: List of game info dicts (one per game)

        Example:
            >>> worker = SelfPlayWorker(encoder, decoder, model)
            >>> examples, infos = worker.generate_games(num_games=50)
            >>> print(f"Generated {len(examples)} training examples from {len(infos)} games")
            >>> win_rate = sum(1 for info in infos if info['outcome'] > 0) / len(infos)
            >>> print(f"White win rate: {win_rate:.2%}")
        """
        # Import here to avoid circular dependency
        from src.mcts.mcts import MCTS
        from src.mcts.evaluator import NeuralNetworkEvaluator

        # Create evaluator with current network
        evaluator = NeuralNetworkEvaluator(self.encoder, self.model, self.decoder)

        # Create MCTS instance
        mcts = MCTS(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=1.0  # Will be overridden by game's schedule
        )

        # Create game manager
        game = SelfPlayGame(
            encoder=self.encoder,
            decoder=self.decoder,
            mcts=mcts,
            evaluator=evaluator,
            temperature_threshold=self.temperature_threshold,
            high_temperature=self.high_temperature,
            low_temperature=self.low_temperature
        )

        # Generate games
        all_examples = []
        game_infos = []

        if verbose:
            print(f"Generating {num_games} self-play games...")
            print(f"MCTS simulations per move: {self.num_simulations}")
            print(f"Temperature threshold: {self.temperature_threshold}")

        for game_idx in range(num_games):
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  Game {game_idx + 1}/{num_games}...")

            # Play one game
            examples, info = game.play_game()

            # Collect data
            all_examples.extend(examples)
            game_infos.append(info)

        if verbose:
            # Print summary statistics
            total_examples = len(all_examples)
            avg_moves = np.mean([info['num_moves'] for info in game_infos])
            avg_time = np.mean([info['time_taken'] for info in game_infos])

            # Count results
            white_wins = sum(1 for info in game_infos if '1-0' in info['result'])
            black_wins = sum(1 for info in game_infos if '0-1' in info['result'])
            draws = sum(1 for info in game_infos if '1/2' in info['result'])

            print(f"\nSelf-play generation complete!")
            print(f"  Total training examples: {total_examples}")
            print(f"  Average moves per game: {avg_moves:.1f}")
            print(f"  Average time per game: {avg_time:.1f}s")
            print(f"  Results: {white_wins} White wins, {black_wins} Black wins, {draws} draws")
            print(f"  White win rate: {white_wins/num_games:.1%}")

        return all_examples, game_infos


def play_single_game(
    encoder,
    decoder,
    model,
    num_simulations: int = 50,
    verbose: bool = False
) -> Tuple[List[TrainingExample], Dict]:
    """
    Convenience function to play a single self-play game.

    Useful for quick testing or generating a small amount of data.

    Args:
        encoder: BoardEncoder instance
        decoder: MoveDecoder instance
        model: ChessNet neural network
        num_simulations: MCTS simulations per move
        verbose: Whether to print game progress

    Returns:
        Tuple of (training_examples, game_info)

    Example:
        >>> examples, info = play_single_game(encoder, decoder, model)
        >>> print(f"Game finished: {info['result']}")
        >>> print(f"Generated {len(examples)} training examples")
    """
    from src.mcts.mcts import MCTS
    from src.mcts.evaluator import NeuralNetworkEvaluator

    # Create components
    evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
    mcts = MCTS(num_simulations=num_simulations, c_puct=1.5)

    # Create game
    game = SelfPlayGame(encoder, decoder, mcts, evaluator)

    # Play
    if verbose:
        print("Playing self-play game...")

    examples, info = game.play_game()

    if verbose:
        print(f"Game finished: {info['result']}")
        print(f"Moves: {info['num_moves']}")
        print(f"Time: {info['time_taken']:.1f}s")
        print(f"Training examples: {len(examples)}")

    return examples, info


if __name__ == "__main__":
    """Demo: generate a self-play game with an untrained network"""
    print("Self-Play Demo")
    print("=" * 60)

    # Import dependencies
    from src.game.encoder import BoardEncoder
    from src.game.decoder import MoveDecoder
    from src.models.chess_net import ChessNet

    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    model = ChessNet()  # Untrained

    print("Playing one self-play game...")
    print("(This will be slow and random since network is untrained)")

    # Play game with very few simulations for demo
    examples, info = play_single_game(
        encoder, decoder, model,
        num_simulations=10,  # Very few for demo speed
        verbose=True
    )

    print(f"\n{'-'*60}")
    print("Training examples generated:")
    print(f"  Total examples: {len(examples)}")
    print(f"  First example shape: {examples[0].board_tensor.shape}")
    print(f"  Policy shape: {examples[0].mcts_policy.shape}")
    print(f"  Value: {examples[0].value}")

    # Show some statistics
    print(f"\nPolicy statistics (first position):")
    policy = examples[0].mcts_policy
    print(f"  Non-zero entries: {np.count_nonzero(policy)}")
    print(f"  Sum of probabilities: {policy.sum():.6f}")
    print(f"  Max probability: {policy.max():.4f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNote: With an untrained network, games are random.")
    print("After training, the network will learn to play better.")
