"""
Neural Network Evaluator for MCTS

This module provides the bridge between MCTS and the neural network.
It combines the encoder, neural network, and decoder to create an
evaluation function that MCTS can use.

The evaluator:
1. Takes a chess board position
2. Encodes it to tensor format
3. Passes through neural network
4. Decodes policy output to move probabilities
5. Returns (policy_dict, value) for MCTS
"""

import chess
import numpy as np
from typing import Dict, Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NeuralNetworkEvaluator:
    """
    Evaluator that uses a neural network to evaluate positions for MCTS.

    This class wraps the encoder, neural network, and decoder into a single
    evaluation function that MCTS can use. It handles all the conversions
    between chess positions and neural network tensors.

    The evaluation process:
    1. Board → Encoder → Tensor (18, 8, 8)
    2. Tensor → Neural Network → (Policy logits, Value)
    3. Policy logits → Decoder → {Move: Probability}
    4. Return (policy_dict, value)

    Attributes:
        encoder: BoardEncoder instance
        model: ChessNet neural network
        decoder: MoveDecoder instance
        device: torch device ('cpu' or 'cuda')
    """

    def __init__(self, encoder, model, decoder, device: str = 'cpu'):
        """
        Initialize the neural network evaluator.

        Args:
            encoder: BoardEncoder instance for encoding positions
            model: ChessNet instance for evaluation
            decoder: MoveDecoder instance for decoding moves
            device: Device to run model on ('cpu' or 'cuda')

        Example:
            >>> from src.game.encoder import BoardEncoder
            >>> from src.models.chess_net import ChessNet
            >>> from src.game.decoder import MoveDecoder
            >>>
            >>> encoder = BoardEncoder()
            >>> model = ChessNet()
            >>> decoder = MoveDecoder()
            >>> evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralNetworkEvaluator")

        self.encoder = encoder
        self.model = model
        self.decoder = decoder
        self.device = device

        # Move model to device and set to evaluation mode
        self.model.to(device)
        self.model.eval()

    def evaluate(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate a board position using the neural network.

        This is the main evaluation function used by MCTS.
        It returns both a policy (move probabilities) and a value (position evaluation).

        Args:
            board: Chess board position to evaluate

        Returns:
            Tuple of (policy_dict, value):
            - policy_dict: Dictionary mapping legal moves to probabilities
            - value: Position evaluation in range [-1, 1]
                    +1 = winning for side to move
                    0 = drawn
                    -1 = losing for side to move

        Example:
            >>> board = chess.Board()
            >>> policy, value = evaluator.evaluate(board)
            >>> print(f"Position value: {value:.3f}")
            >>> print(f"Best move: {max(policy.items(), key=lambda x: x[1])[0]}")
        """
        # Step 1: Encode board to numpy array
        # This converts the chess position to a tensor with 18 feature planes
        board_np = self.encoder.encode(board)  # Shape: (18, 8, 8)

        # Step 2: Convert to PyTorch tensor and add batch dimension
        # Neural network expects batch input: (batch_size, channels, height, width)
        board_tensor = torch.from_numpy(board_np).unsqueeze(0)  # Shape: (1, 18, 8, 8)
        board_tensor = board_tensor.to(self.device)

        # Step 3: Get neural network prediction
        # Disable gradient computation for efficiency (we're not training)
        with torch.no_grad():
            policy_logits, value_tensor = self.model(board_tensor)

        # Step 4: Convert outputs to numpy
        # Remove batch dimension [0] since we only have one position
        policy_logits_np = policy_logits[0].cpu().numpy()  # Shape: (4672,)
        value = value_tensor[0].item()  # Scalar value

        # Step 5: Decode policy logits to move probabilities
        # This converts the 4672 policy outputs to probabilities for legal moves only
        policy_dict = self.decoder.policy_to_move_probabilities(
            policy_logits_np,
            board,
            temperature=1.0
        )

        # Return policy and value for MCTS to use
        return policy_dict, value

    def evaluate_batch(
        self,
        boards: list
    ) -> list:
        """
        Evaluate multiple positions in a batch for efficiency.

        Batch evaluation is faster than evaluating positions one by one
        because the GPU can process multiple positions in parallel.

        Args:
            boards: List of chess.Board positions to evaluate

        Returns:
            List of (policy_dict, value) tuples, one per board

        Example:
            >>> boards = [chess.Board(), chess.Board("...fen...")]
            >>> results = evaluator.evaluate_batch(boards)
            >>> for policy, value in results:
            >>>     print(f"Value: {value:.3f}")
        """
        if not boards:
            return []

        # Step 1: Encode all boards
        board_tensors = []
        for board in boards:
            board_np = self.encoder.encode(board)
            board_tensors.append(torch.from_numpy(board_np))

        # Stack into batch: list of (18,8,8) -> (batch_size, 18, 8, 8)
        batch_tensor = torch.stack(board_tensors).to(self.device)

        # Step 2: Forward pass through network
        with torch.no_grad():
            policy_logits_batch, value_batch = self.model(batch_tensor)

        # Step 3: Decode each position
        results = []
        for i, board in enumerate(boards):
            # Extract results for this board
            policy_logits_np = policy_logits_batch[i].cpu().numpy()
            value = value_batch[i].item()

            # Decode to move probabilities
            policy_dict = self.decoder.policy_to_move_probabilities(
                policy_logits_np,
                board,
                temperature=1.0
            )

            results.append((policy_dict, value))

        return results

    def __call__(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Allow evaluator to be called like a function.

        This enables using the evaluator directly with MCTS:
        mcts.search(board, evaluator)

        Args:
            board: Chess board to evaluate

        Returns:
            (policy_dict, value) tuple
        """
        return self.evaluate(board)


class RandomEvaluator:
    """
    Random evaluator for testing and baseline comparison.

    This evaluator returns:
    - Uniform policy (all legal moves equally likely)
    - Random value

    Useful for:
    - Testing MCTS without a trained network
    - Baseline comparison
    - Debugging

    Example:
        >>> evaluator = RandomEvaluator()
        >>> policy, value = evaluator(chess.Board())
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random evaluator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)

    def evaluate(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position with uniform policy and random value.

        Args:
            board: Chess board position

        Returns:
            (policy_dict, value) where:
            - policy_dict: Uniform probabilities over legal moves
            - value: Random value in [-1, 1]
        """
        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves (game over)
            return {}, 0.0

        # Uniform policy: all moves equally likely
        prob = 1.0 / len(legal_moves)
        policy_dict = {move: prob for move in legal_moves}

        # Random value
        value = np.random.uniform(-1.0, 1.0)

        return policy_dict, value

    def __call__(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """Allow calling as function"""
        return self.evaluate(board)


class SimpleHeuristicEvaluator:
    """
    Simple heuristic evaluator based on material count.

    This evaluator:
    - Uses material balance for value
    - Biases policy toward captures
    - No neural network required

    Piece values:
    - Pawn: 1
    - Knight: 3
    - Bishop: 3
    - Rook: 5
    - Queen: 9
    - King: infinite (game over if captured)

    Example:
        >>> evaluator = SimpleHeuristicEvaluator()
        >>> policy, value = evaluator(chess.Board())
    """

    # Standard piece values
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Infinite, but we use 0 since game ends
    }

    def evaluate(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position using material balance and simple policy.

        Args:
            board: Chess board position

        Returns:
            (policy_dict, value) where:
            - policy_dict: Biased toward captures and checks
            - value: Material balance normalized to [-1, 1]
        """
        # Calculate material balance
        value = self._material_balance(board)

        # Create policy biased toward good moves
        policy_dict = self._create_policy(board)

        return policy_dict, value

    def _material_balance(self, board: chess.Board) -> float:
        """
        Calculate material balance from current player's perspective.

        Returns value in approximately [-1, 1] range.
        """
        # Count material for each side
        white_material = 0
        black_material = 0

        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                          chess.ROOK, chess.QUEEN]:
            white_material += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]

        # Material difference
        diff = white_material - black_material

        # Normalize to approximately [-1, 1]
        # Maximum reasonable material difference is about 39 (queen + 2 rooks)
        normalized = np.tanh(diff / 10.0)

        # Return from current player's perspective
        if board.turn == chess.WHITE:
            return normalized
        else:
            return -normalized

    def _create_policy(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Create policy biased toward good moves.

        Prioritizes:
        1. Captures (higher material gain = higher probability)
        2. Checks
        3. Other moves

        Returns uniform-ish distribution with bias.
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return {}

        # Assign scores to moves
        move_scores = {}
        for move in legal_moves:
            score = 1.0  # Base score

            # Bonus for captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    score += self.PIECE_VALUES.get(captured_piece.piece_type, 0)

            # Bonus for checks
            board.push(move)
            if board.is_check():
                score += 2.0
            board.pop()

            move_scores[move] = score

        # Convert scores to probabilities (softmax-like)
        total_score = sum(move_scores.values())
        policy_dict = {move: score / total_score for move, score in move_scores.items()}

        return policy_dict

    def __call__(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """Allow calling as function"""
        return self.evaluate(board)


if __name__ == "__main__":
    """Demo: test evaluators"""
    import chess

    print("Evaluator Demo")
    print("=" * 60)

    # Test random evaluator
    print("\n1. Random Evaluator:")
    random_eval = RandomEvaluator(seed=42)
    board = chess.Board()

    policy, value = random_eval(board)
    print(f"   Value: {value:.3f}")
    print(f"   Policy moves: {len(policy)}")
    print(f"   Policy sum: {sum(policy.values()):.3f}")

    # Test heuristic evaluator
    print("\n2. Heuristic Evaluator:")
    heuristic_eval = SimpleHeuristicEvaluator()

    policy, value = heuristic_eval(board)
    print(f"   Value: {value:.3f}")
    top_3 = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   Top 3 moves:")
    for move, prob in top_3:
        print(f"     {move}: {prob:.4f}")

    # Test with position after e4
    print("\n3. After 1.e4:")
    board.push_san("e4")
    policy, value = heuristic_eval(board)
    print(f"   Value (Black's perspective): {value:.3f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNote: For NeuralNetworkEvaluator demo, run:")
    print("  python scripts/demo_mcts.py")
