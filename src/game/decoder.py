"""
Move Decoder - Convert neural network output to chess moves

This module handles the conversion of neural network policy output
(4672-dimensional vector) into legal chess moves.

The policy output uses a 73-plane encoding for each of the 64 squares,
representing all possible move types from that square:
- 56 queen-style moves (7 directions × 8 distances)
- 8 knight moves
- 9 underpromotions (3 directions × 3 piece types)

Total: 73 × 64 = 4672 possible move encodings
"""

import numpy as np
import chess
from typing import Optional, Tuple, List


class MoveDecoder:
    """
    Decodes neural network policy output into legal chess moves.

    The decoder maps the 4672-dimensional policy vector to actual chess moves,
    filtering for legality and providing methods to select moves based on
    the policy probabilities.
    """

    def __init__(self):
        """Initialize the MoveDecoder with move encoding tables."""
        self.num_move_planes = 73
        self.num_squares = 64
        self.policy_size = self.num_move_planes * self.num_squares  # 4672

        # Build encoding/decoding tables
        self._build_move_lookup_tables()

    def _build_move_lookup_tables(self):
        """
        Build lookup tables for encoding/decoding moves.

        Creates mappings between:
        - (from_square, to_square, promotion) -> policy_index
        - policy_index -> (from_square, direction_plane)
        """
        self.move_to_policy_index = {}
        self.policy_index_to_move_info = {}

        # Direction vectors for queen moves (N, NE, E, SE, S, SW, W, NW)
        self.queen_directions = [
            (0, 1),   # North
            (1, 1),   # NE
            (1, 0),   # East
            (1, -1),  # SE
            (0, -1),  # South
            (-1, -1), # SW
            (-1, 0),  # West
            (-1, 1),  # NW
        ]

        # Knight move offsets
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

        # Build mappings for each square
        for from_square in range(64):
            from_row = from_square // 8
            from_col = from_square % 8

            plane_idx = 0

            # Queen moves (56 planes: 7 directions × 8 distances)
            for dir_idx, (drow, dcol) in enumerate(self.queen_directions):
                for distance in range(1, 8):
                    to_row = from_row + drow * distance
                    to_col = from_col + dcol * distance

                    # Check if destination is on board
                    if 0 <= to_row < 8 and 0 <= to_col < 8:
                        to_square = to_row * 8 + to_col
                        policy_idx = from_square * self.num_move_planes + plane_idx

                        # Store mapping (without promotion for queen moves)
                        self.move_to_policy_index[(from_square, to_square, None)] = policy_idx
                        self.policy_index_to_move_info[policy_idx] = (from_square, to_square, None)

                    plane_idx += 1

            # Knight moves (8 planes)
            for drow, dcol in self.knight_moves:
                to_row = from_row + drow
                to_col = from_col + dcol

                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_square = to_row * 8 + to_col
                    policy_idx = from_square * self.num_move_planes + plane_idx

                    self.move_to_policy_index[(from_square, to_square, None)] = policy_idx
                    self.policy_index_to_move_info[policy_idx] = (from_square, to_square, None)

                plane_idx += 1

            # Underpromotions (9 planes: 3 directions × 3 piece types)
            # Directions: NW, N, NE for white pawns (left, straight, right)
            underpromote_directions = [-1, 0, 1]  # column offsets only
            underpromote_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

            for dcol in underpromote_directions:
                for piece_type in underpromote_pieces:
                    # For white pawns moving up (row increases)
                    to_row = from_row + 1
                    to_col = from_col + dcol

                    if 0 <= to_row < 8 and 0 <= to_col < 8:
                        to_square = to_row * 8 + to_col
                        policy_idx = from_square * self.num_move_planes + plane_idx

                        # Store with promotion piece type
                        self.move_to_policy_index[(from_square, to_square, piece_type)] = policy_idx
                        self.policy_index_to_move_info[policy_idx] = (from_square, to_square, piece_type)

                    plane_idx += 1

    def decode_policy_index(self, policy_index: int) -> Optional[Tuple[int, int, Optional[int]]]:
        """
        Decode a policy index to move information.

        Args:
            policy_index: Index in policy vector (0-4671)

        Returns:
            Tuple of (from_square, to_square, promotion) or None if invalid
        """
        return self.policy_index_to_move_info.get(policy_index)

    def encode_move(self, move: chess.Move) -> Optional[int]:
        """
        Encode a chess move to policy index.

        Args:
            move: python-chess Move object

        Returns:
            Policy index (0-4671) or None if move cannot be encoded
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        # For queen promotion, treat as no promotion in encoding
        # (queen moves are already in the queen move planes)
        if promotion == chess.QUEEN:
            promotion = None

        return self.move_to_policy_index.get((from_square, to_square, promotion))

    def policy_to_move_probabilities(
        self,
        policy_logits: np.ndarray,
        board: chess.Board,
        temperature: float = 1.0
    ) -> dict:
        """
        Convert policy logits to move probabilities for legal moves only.

        Args:
            policy_logits: Raw policy output from neural network (4672,)
            board: Current board position
            temperature: Sampling temperature (1.0 = unchanged, <1 = greedy, >1 = random)

        Returns:
            Dictionary mapping chess.Move -> probability (normalized over legal moves)
        """
        # Get all legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return {}

        # Extract logits for legal moves
        move_logits = []
        move_objects = []

        for move in legal_moves:
            policy_idx = self.encode_move(move)

            if policy_idx is not None:
                move_logits.append(policy_logits[policy_idx])
                move_objects.append(move)
            else:
                # If move cannot be encoded, assign very low probability
                move_logits.append(-1e6)
                move_objects.append(move)

        move_logits = np.array(move_logits)

        # Apply temperature
        if temperature != 1.0:
            move_logits = move_logits / temperature

        # Compute softmax to get probabilities
        # Subtract max for numerical stability
        move_logits_max = np.max(move_logits)
        exp_logits = np.exp(move_logits - move_logits_max)
        probabilities = exp_logits / np.sum(exp_logits)

        # Create move -> probability mapping
        move_probs = {}
        for move, prob in zip(move_objects, probabilities):
            move_probs[move] = prob

        return move_probs

    def select_move_greedy(
        self,
        policy_logits: np.ndarray,
        board: chess.Board
    ) -> Optional[chess.Move]:
        """
        Select the move with highest probability (greedy selection).

        Args:
            policy_logits: Raw policy output from neural network (4672,)
            board: Current board position

        Returns:
            chess.Move with highest probability, or None if no legal moves
        """
        move_probs = self.policy_to_move_probabilities(policy_logits, board, temperature=1.0)

        if not move_probs:
            return None

        # Return move with highest probability
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        return best_move

    def select_move_sampling(
        self,
        policy_logits: np.ndarray,
        board: chess.Board,
        temperature: float = 1.0
    ) -> Optional[chess.Move]:
        """
        Sample a move according to the policy distribution.

        Args:
            policy_logits: Raw policy output from neural network (4672,)
            board: Current board position
            temperature: Sampling temperature (1.0 = unchanged, <1 = greedy, >1 = random)

        Returns:
            Sampled chess.Move, or None if no legal moves
        """
        move_probs = self.policy_to_move_probabilities(policy_logits, board, temperature)

        if not move_probs:
            return None

        # Sample according to probabilities
        moves = list(move_probs.keys())
        probs = list(move_probs.values())

        selected_move = np.random.choice(moves, p=probs)
        return selected_move

    def get_top_moves(
        self,
        policy_logits: np.ndarray,
        board: chess.Board,
        top_k: int = 5
    ) -> List[Tuple[chess.Move, float]]:
        """
        Get top-k moves with highest probabilities.

        Args:
            policy_logits: Raw policy output from neural network (4672,)
            board: Current board position
            top_k: Number of top moves to return

        Returns:
            List of (move, probability) tuples, sorted by probability (descending)
        """
        move_probs = self.policy_to_move_probabilities(policy_logits, board)

        if not move_probs:
            return []

        # Sort by probability (descending)
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        return sorted_moves[:top_k]

    def create_random_policy(self) -> np.ndarray:
        """
        Create a random policy vector (for testing/baseline).

        Returns:
            Random policy logits of shape (4672,)
        """
        return np.random.randn(self.policy_size).astype(np.float32)

    def create_uniform_policy(self) -> np.ndarray:
        """
        Create a uniform policy vector (all moves equally likely).

        Returns:
            Uniform policy logits of shape (4672,)
        """
        return np.zeros(self.policy_size, dtype=np.float32)

    def move_to_string(self, move: chess.Move) -> str:
        """
        Convert move to human-readable string.

        Args:
            move: chess.Move object

        Returns:
            String representation (e.g., "e2e4", "e7e8q")
        """
        return move.uci()

    def visualize_policy(
        self,
        policy_logits: np.ndarray,
        board: chess.Board,
        top_k: int = 10
    ) -> str:
        """
        Create human-readable visualization of policy.

        Args:
            policy_logits: Raw policy output from neural network (4672,)
            board: Current board position
            top_k: Number of top moves to show

        Returns:
            String with formatted policy visualization
        """
        result = [f"\nPolicy Visualization (Top {top_k} moves):"]
        result.append("-" * 50)

        top_moves = self.get_top_moves(policy_logits, board, top_k)

        if not top_moves:
            result.append("No legal moves available")
            return "\n".join(result)

        result.append(f"{'Rank':<6} {'Move':<8} {'Probability':<12} {'Bar'}")
        result.append("-" * 50)

        for rank, (move, prob) in enumerate(top_moves, 1):
            # Create probability bar
            bar_length = int(prob * 40)
            bar = "█" * bar_length

            move_str = self.move_to_string(move)
            prob_str = f"{prob:.4f} ({prob*100:.2f}%)"

            result.append(f"{rank:<6} {move_str:<8} {prob_str:<12} {bar}")

        return "\n".join(result)


# Module-level convenience functions
def decode_move_greedy(policy_logits: np.ndarray, board: chess.Board) -> Optional[chess.Move]:
    """
    Convenience function to decode policy and select best move.

    Args:
        policy_logits: Policy output from neural network (4672,)
        board: Current board position

    Returns:
        Best legal move according to policy
    """
    decoder = MoveDecoder()
    return decoder.select_move_greedy(policy_logits, board)


def decode_move_sampling(
    policy_logits: np.ndarray,
    board: chess.Board,
    temperature: float = 1.0
) -> Optional[chess.Move]:
    """
    Convenience function to decode policy and sample a move.

    Args:
        policy_logits: Policy output from neural network (4672,)
        board: Current board position
        temperature: Sampling temperature

    Returns:
        Sampled legal move according to policy
    """
    decoder = MoveDecoder()
    return decoder.select_move_sampling(policy_logits, board, temperature)


if __name__ == "__main__":
    """Demo: decode random policy and visualize"""
    print("Move Decoder Demo")
    print("=" * 60)

    # Create decoder
    decoder = MoveDecoder()

    # Starting position
    board = chess.Board()
    print("\nStarting position:")
    print(board)
    print(f"\nLegal moves: {len(list(board.legal_moves))}")

    # Create random policy
    policy = decoder.create_random_policy()
    print(f"\nRandom policy shape: {policy.shape}")

    # Visualize policy
    print(decoder.visualize_policy(policy, board, top_k=10))

    # Select best move
    best_move = decoder.select_move_greedy(policy, board)
    print(f"\nBest move (greedy): {best_move}")

    # Sample moves with different temperatures
    print("\n" + "=" * 60)
    print("Sampling with different temperatures:")

    for temp in [0.5, 1.0, 2.0]:
        sampled_move = decoder.select_move_sampling(policy, board, temperature=temp)
        print(f"  Temperature {temp}: {sampled_move}")

    # Test with endgame position (fewer moves)
    print("\n" + "=" * 60)
    print("Endgame position (King vs King + Pawn):")
    board_endgame = chess.Board(fen="8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    print(board_endgame)
    print(f"Legal moves: {len(list(board_endgame.legal_moves))}")

    policy_endgame = decoder.create_random_policy()
    print(decoder.visualize_policy(policy_endgame, board_endgame, top_k=5))

    # Test move encoding/decoding
    print("\n" + "=" * 60)
    print("Move encoding/decoding test:")

    test_moves = [
        chess.Move.from_uci("e2e4"),  # Pawn push
        chess.Move.from_uci("g1f3"),  # Knight move
        chess.Move.from_uci("e1g1"),  # Castling (represented as king move)
    ]

    for move in test_moves:
        policy_idx = decoder.encode_move(move)
        print(f"  {move}: policy index = {policy_idx}")

        if policy_idx is not None:
            decoded = decoder.decode_policy_index(policy_idx)
            print(f"    Decoded: {decoded}")
