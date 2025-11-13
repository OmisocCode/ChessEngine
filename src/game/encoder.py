"""
Board Encoder - Convert chess.Board to neural network input tensor

This module handles the conversion of chess positions (chess.Board objects)
into tensor representations suitable for neural network input.

Encoding format: 18 planes × 8×8
- Planes 0-5: White pieces (P, N, B, R, Q, K)
- Planes 6-11: Black pieces (P, N, B, R, Q, K)
- Plane 12: Repetition counter (for draw detection)
- Plane 13: Turn color (1.0 = white to move, 0.0 = black to move)
- Plane 14: White castling rights (kingside/queenside encoded)
- Plane 15: Black castling rights (kingside/queenside encoded)
- Plane 16: En passant square
- Plane 17: Halfmove clock (normalized by 100 for 50-move rule)
"""

import numpy as np
import chess


class BoardEncoder:
    """
    Encodes chess board positions into tensor format for neural network input.

    The encoding uses 18 feature planes to represent all relevant information
    about a chess position including piece positions, castling rights, en passant,
    and move counters.
    """

    # Piece type mapping to plane indices (0-5 for white, 6-11 for black)
    PIECE_TO_PLANE = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    def __init__(self):
        """Initialize the BoardEncoder."""
        self.num_planes = 18
        self.board_size = 8

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encode a chess board position into a tensor.

        Args:
            board: python-chess Board object representing the position

        Returns:
            np.ndarray of shape (18, 8, 8) with float32 dtype

        Example:
            >>> encoder = BoardEncoder()
            >>> board = chess.Board()
            >>> tensor = encoder.encode(board)
            >>> tensor.shape
            (18, 8, 8)
        """
        # Initialize empty tensor
        tensor = np.zeros((self.num_planes, self.board_size, self.board_size), dtype=np.float32)

        # Encode piece positions (planes 0-11)
        self._encode_pieces(board, tensor)

        # Encode repetition counter (plane 12)
        self._encode_repetitions(board, tensor)

        # Encode turn color (plane 13)
        self._encode_turn(board, tensor)

        # Encode castling rights (planes 14-15)
        self._encode_castling(board, tensor)

        # Encode en passant (plane 16)
        self._encode_en_passant(board, tensor)

        # Encode halfmove clock (plane 17)
        self._encode_halfmove_clock(board, tensor)

        return tensor

    def _encode_pieces(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode piece positions into planes 0-11.

        Planes 0-5: White pieces (P, N, B, R, Q, K)
        Planes 6-11: Black pieces (P, N, B, R, Q, K)
        """
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Get piece type plane (0-5)
                piece_plane = self.PIECE_TO_PLANE[piece.piece_type]

                # Add 6 for black pieces
                if piece.color == chess.BLACK:
                    piece_plane += 6

                # Convert square to (row, col)
                row, col = self._square_to_coords(square)

                # Set the position in the tensor
                tensor[piece_plane, row, col] = 1.0

    def _encode_repetitions(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode position repetition count (plane 12).

        This helps the network detect threefold repetition draws.
        Values are normalized: 0.0 (no repetition), 0.5 (seen once), 1.0 (seen twice or more)
        """
        # Count how many times this position has occurred
        # board.is_repetition(count) checks if position repeated 'count' times
        if board.can_claim_threefold_repetition():
            repetition_value = 1.0  # Position repeated 2+ times
        elif board.is_repetition(2):
            repetition_value = 0.5  # Position seen once before
        else:
            repetition_value = 0.0  # First time seeing this position

        tensor[12, :, :] = repetition_value

    def _encode_turn(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode whose turn it is (plane 13).

        1.0 = white to move
        0.0 = black to move
        """
        turn_value = 1.0 if board.turn == chess.WHITE else 0.0
        tensor[13, :, :] = turn_value

    def _encode_castling(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode castling rights (planes 14-15).

        Plane 14: White castling rights
        Plane 15: Black castling rights

        Each plane is filled with a value:
        - 0.0: no castling rights
        - 0.5: one side available (kingside or queenside)
        - 1.0: both sides available
        """
        # White castling
        white_kingside = board.has_kingside_castling_rights(chess.WHITE)
        white_queenside = board.has_queenside_castling_rights(chess.WHITE)
        white_value = (int(white_kingside) + int(white_queenside)) / 2.0
        tensor[14, :, :] = white_value

        # Black castling
        black_kingside = board.has_kingside_castling_rights(chess.BLACK)
        black_queenside = board.has_queenside_castling_rights(chess.BLACK)
        black_value = (int(black_kingside) + int(black_queenside)) / 2.0
        tensor[15, :, :] = black_value

    def _encode_en_passant(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode en passant square (plane 16).

        If there's an en passant square, set that position to 1.0.
        All other positions remain 0.0.
        """
        if board.ep_square is not None:
            row, col = self._square_to_coords(board.ep_square)
            tensor[16, row, col] = 1.0

    def _encode_halfmove_clock(self, board: chess.Board, tensor: np.ndarray) -> None:
        """
        Encode halfmove clock (plane 17).

        Normalized by 100 to get values in [0, 1] range.
        The 50-move rule means game is draw at 100 halfmoves (50 full moves).
        """
        halfmove_value = min(board.halfmove_clock / 100.0, 1.0)
        tensor[17, :, :] = halfmove_value

    def _square_to_coords(self, square: int) -> tuple:
        """
        Convert chess square index to (row, col) coordinates.

        Args:
            square: Square index (0-63) where 0=a1, 63=h8

        Returns:
            Tuple of (row, col) where row=0 is rank 1, col=0 is file a

        Example:
            >>> encoder = BoardEncoder()
            >>> encoder._square_to_coords(chess.A1)
            (0, 0)
            >>> encoder._square_to_coords(chess.H8)
            (7, 7)
        """
        row = square // 8
        col = square % 8
        return row, col

    def decode_plane_name(self, plane_idx: int) -> str:
        """
        Get human-readable name for a plane index.

        Useful for debugging and visualization.

        Args:
            plane_idx: Plane index (0-17)

        Returns:
            String description of the plane
        """
        if plane_idx < 6:
            pieces = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
            return f"White {pieces[plane_idx]}"
        elif plane_idx < 12:
            pieces = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
            return f"Black {pieces[plane_idx - 6]}"
        elif plane_idx == 12:
            return "Repetitions"
        elif plane_idx == 13:
            return "Turn (1=White, 0=Black)"
        elif plane_idx == 14:
            return "White Castling Rights"
        elif plane_idx == 15:
            return "Black Castling Rights"
        elif plane_idx == 16:
            return "En Passant Square"
        elif plane_idx == 17:
            return "Halfmove Clock"
        else:
            return f"Unknown plane {plane_idx}"

    def visualize_plane(self, tensor: np.ndarray, plane_idx: int) -> str:
        """
        Create ASCII visualization of a single plane.

        Args:
            tensor: Encoded tensor of shape (18, 8, 8)
            plane_idx: Which plane to visualize (0-17)

        Returns:
            String with ASCII representation of the plane
        """
        if plane_idx >= self.num_planes:
            return f"Invalid plane index {plane_idx}"

        plane = tensor[plane_idx]
        result = [f"\n{self.decode_plane_name(plane_idx)}:"]
        result.append("  a b c d e f g h")

        # Display from rank 8 to rank 1 (top to bottom)
        for row in range(7, -1, -1):
            rank = row + 1
            line = f"{rank} "
            for col in range(8):
                value = plane[row, col]
                if value > 0.9:
                    line += "█ "
                elif value > 0.4:
                    line += "▓ "
                elif value > 0.0:
                    line += "░ "
                else:
                    line += "· "
            result.append(line)

        return "\n".join(result)


# Module-level convenience function
def encode_board(board: chess.Board) -> np.ndarray:
    """
    Convenience function to encode a board without creating an encoder instance.

    Args:
        board: python-chess Board object

    Returns:
        np.ndarray of shape (18, 8, 8)
    """
    encoder = BoardEncoder()
    return encoder.encode(board)


if __name__ == "__main__":
    """Demo: encode and visualize starting position"""
    print("Board Encoder Demo")
    print("=" * 60)

    # Create encoder and starting position
    encoder = BoardEncoder()
    board = chess.Board()

    print("\nStarting position:")
    print(board)

    # Encode
    tensor = encoder.encode(board)
    print(f"\nEncoded tensor shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")

    # Visualize some interesting planes
    print(encoder.visualize_plane(tensor, 0))  # White pawns
    print(encoder.visualize_plane(tensor, 5))  # White king
    print(encoder.visualize_plane(tensor, 13))  # Turn
    print(encoder.visualize_plane(tensor, 14))  # White castling

    # Test after a move
    print("\n" + "=" * 60)
    print("After 1.e4:")
    board.push_san("e4")
    print(board)

    tensor2 = encoder.encode(board)
    print(encoder.visualize_plane(tensor2, 0))  # White pawns (e2 pawn moved)
    print(encoder.visualize_plane(tensor2, 13))  # Turn (now black)
