"""
Test Suite for Board Encoder

Tests the encoding of chess positions into neural network input tensors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import chess
from src.game.encoder import BoardEncoder, encode_board


class TestBoardEncoderBasics:
    """Test basic encoder functionality"""

    def test_encoder_initialization(self):
        """Test encoder can be created"""
        encoder = BoardEncoder()
        assert encoder.num_planes == 18
        assert encoder.board_size == 8

    def test_encode_returns_correct_shape(self):
        """Test encoded tensor has correct shape"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        assert tensor.shape == (18, 8, 8)
        assert tensor.dtype == np.float32

    def test_encode_empty_board(self):
        """Test encoding empty board (no pieces)"""
        encoder = BoardEncoder()
        board = chess.Board(fen=None)  # Empty board
        board.clear()

        tensor = encoder.encode(board)

        # All piece planes should be empty
        for plane_idx in range(12):
            assert np.sum(tensor[plane_idx]) == 0, f"Plane {plane_idx} should be empty"

    def test_convenience_function(self):
        """Test module-level encode_board function"""
        board = chess.Board()
        tensor = encode_board(board)

        assert tensor.shape == (18, 8, 8)
        assert isinstance(tensor, np.ndarray)


class TestPieceEncoding:
    """Test encoding of piece positions"""

    def test_starting_position_white_pieces(self):
        """Test white pieces in starting position"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        # White pawns (plane 0) - should have 8 pawns on rank 2 (row 1)
        white_pawns = tensor[0]
        assert np.sum(white_pawns) == 8
        assert np.sum(white_pawns[1, :]) == 8  # All on rank 2

        # White knights (plane 1) - b1 and g1
        white_knights = tensor[1]
        assert np.sum(white_knights) == 2
        assert white_knights[0, 1] == 1.0  # b1
        assert white_knights[0, 6] == 1.0  # g1

        # White bishops (plane 2) - c1 and f1
        white_bishops = tensor[2]
        assert np.sum(white_bishops) == 2
        assert white_bishops[0, 2] == 1.0  # c1
        assert white_bishops[0, 5] == 1.0  # f1

        # White rooks (plane 3) - a1 and h1
        white_rooks = tensor[3]
        assert np.sum(white_rooks) == 2
        assert white_rooks[0, 0] == 1.0  # a1
        assert white_rooks[0, 7] == 1.0  # h1

        # White queen (plane 4) - d1
        white_queen = tensor[4]
        assert np.sum(white_queen) == 1
        assert white_queen[0, 3] == 1.0  # d1

        # White king (plane 5) - e1
        white_king = tensor[5]
        assert np.sum(white_king) == 1
        assert white_king[0, 4] == 1.0  # e1

    def test_starting_position_black_pieces(self):
        """Test black pieces in starting position"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        # Black pawns (plane 6) - should have 8 pawns on rank 7 (row 6)
        black_pawns = tensor[6]
        assert np.sum(black_pawns) == 8
        assert np.sum(black_pawns[6, :]) == 8  # All on rank 7

        # Black king (plane 11) - e8
        black_king = tensor[11]
        assert np.sum(black_king) == 1
        assert black_king[7, 4] == 1.0  # e8

    def test_piece_movement(self):
        """Test piece encoding after moves"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Move white pawn e2-e4
        board.push_san("e4")
        tensor = encoder.encode(board)

        white_pawns = tensor[0]
        # Should still have 8 pawns
        assert np.sum(white_pawns) == 8
        # e2 should be empty, e4 should have pawn
        assert white_pawns[1, 4] == 0.0  # e2 empty
        assert white_pawns[3, 4] == 1.0  # e4 occupied

    def test_single_piece_position(self):
        """Test encoding single piece on board"""
        encoder = BoardEncoder()

        # King vs King endgame
        board = chess.Board(fen="8/8/8/4k3/8/8/8/4K3 w - - 0 1")
        tensor = encoder.encode(board)

        # White king on e1
        white_king = tensor[5]
        assert np.sum(white_king) == 1
        assert white_king[0, 4] == 1.0

        # Black king on e5
        black_king = tensor[11]
        assert np.sum(black_king) == 1
        assert black_king[4, 4] == 1.0

        # All other piece planes should be empty
        for plane_idx in range(12):
            if plane_idx not in [5, 11]:
                assert np.sum(tensor[plane_idx]) == 0


class TestGameStateEncoding:
    """Test encoding of game state information"""

    def test_turn_encoding_white(self):
        """Test turn encoding when white to move"""
        encoder = BoardEncoder()
        board = chess.Board()  # White to move
        tensor = encoder.encode(board)

        turn_plane = tensor[13]
        assert np.all(turn_plane == 1.0), "Turn plane should be all 1.0 for white"

    def test_turn_encoding_black(self):
        """Test turn encoding when black to move"""
        encoder = BoardEncoder()
        board = chess.Board()
        board.push_san("e4")  # Now black to move
        tensor = encoder.encode(board)

        turn_plane = tensor[13]
        assert np.all(turn_plane == 0.0), "Turn plane should be all 0.0 for black"

    def test_castling_rights_initial(self):
        """Test castling rights encoding in starting position"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        # Both sides have both castling rights
        white_castling = tensor[14]
        black_castling = tensor[15]

        assert np.all(white_castling == 1.0), "White should have full castling rights"
        assert np.all(black_castling == 1.0), "Black should have full castling rights"

    def test_castling_rights_after_king_move(self):
        """Test castling rights lost after king moves"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Move white king (loses castling rights)
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Ke2")  # King moves
        tensor = encoder.encode(board)

        white_castling = tensor[14]
        assert np.all(white_castling == 0.0), "White should lose castling rights"

    def test_castling_rights_partial(self):
        """Test partial castling rights (one side only)"""
        encoder = BoardEncoder()

        # Position where white has queenside only, black has kingside only
        board = chess.Board(fen="r3k2r/8/8/8/8/8/8/R3K3 w Qq - 0 1")
        tensor = encoder.encode(board)

        white_castling = tensor[14]
        black_castling = tensor[15]

        # Each should have one right (0.5)
        assert np.all(white_castling == 0.5), "White should have partial castling"
        assert np.all(black_castling == 0.5), "Black should have partial castling"

    def test_en_passant_encoding(self):
        """Test en passant square encoding"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Create en passant opportunity
        board.push_san("e4")
        board.push_san("a6")
        board.push_san("e5")
        board.push_san("d5")  # Black pawn moves two squares, creates ep

        tensor = encoder.encode(board)
        ep_plane = tensor[16]

        # d6 should be marked as en passant square (row 5, col 3)
        assert ep_plane[5, 3] == 1.0, "d6 should be en passant square"
        assert np.sum(ep_plane) == 1.0, "Only one square should be marked"

    def test_no_en_passant(self):
        """Test en passant encoding when no ep available"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        ep_plane = tensor[16]
        assert np.sum(ep_plane) == 0, "No en passant in starting position"

    def test_halfmove_clock_initial(self):
        """Test halfmove clock in starting position"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        halfmove_plane = tensor[17]
        assert np.all(halfmove_plane == 0.0), "Halfmove clock should be 0 initially"

    def test_halfmove_clock_progression(self):
        """Test halfmove clock increases"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Make quiet moves (no pawn moves or captures)
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")

        tensor = encoder.encode(board)
        halfmove_plane = tensor[17]

        # Should be 4/100 = 0.04
        expected_value = 4.0 / 100.0
        assert np.all(halfmove_plane == expected_value), f"Halfmove clock should be {expected_value}"

    def test_halfmove_clock_reset_on_pawn_move(self):
        """Test halfmove clock resets on pawn move"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Make quiet moves
        board.push_san("Nf3")
        board.push_san("Nf6")
        # Pawn move resets clock
        board.push_san("e4")

        tensor = encoder.encode(board)
        halfmove_plane = tensor[17]

        assert np.all(halfmove_plane == 0.0), "Halfmove clock should reset after pawn move"

    def test_repetition_none(self):
        """Test repetition counter when no repetitions"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        repetition_plane = tensor[12]
        assert np.all(repetition_plane == 0.0), "No repetitions in starting position"

    def test_repetition_detection(self):
        """Test repetition counter when position repeats"""
        encoder = BoardEncoder()
        board = chess.Board()

        # Create repetition by moving knights back and forth
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")

        tensor = encoder.encode(board)
        repetition_plane = tensor[12]

        # Position should be repeated (value > 0)
        assert np.all(repetition_plane > 0.0), "Repetition should be detected"


class TestSquareConversion:
    """Test square coordinate conversion"""

    def test_corner_squares(self):
        """Test conversion of corner squares"""
        encoder = BoardEncoder()

        # a1 = square 0
        row, col = encoder._square_to_coords(chess.A1)
        assert (row, col) == (0, 0)

        # h1 = square 7
        row, col = encoder._square_to_coords(chess.H1)
        assert (row, col) == (0, 7)

        # a8 = square 56
        row, col = encoder._square_to_coords(chess.A8)
        assert (row, col) == (7, 0)

        # h8 = square 63
        row, col = encoder._square_to_coords(chess.H8)
        assert (row, col) == (7, 7)

    def test_central_squares(self):
        """Test conversion of central squares"""
        encoder = BoardEncoder()

        # e4 = square 28
        row, col = encoder._square_to_coords(chess.E4)
        assert (row, col) == (3, 4)

        # d5 = square 35
        row, col = encoder._square_to_coords(chess.D5)
        assert (row, col) == (4, 3)


class TestVisualization:
    """Test visualization functions"""

    def test_plane_name_decoding(self):
        """Test plane name decoding"""
        encoder = BoardEncoder()

        assert encoder.decode_plane_name(0) == "White Pawn"
        assert encoder.decode_plane_name(5) == "White King"
        assert encoder.decode_plane_name(6) == "Black Pawn"
        assert encoder.decode_plane_name(11) == "Black King"
        assert encoder.decode_plane_name(12) == "Repetitions"
        assert encoder.decode_plane_name(13) == "Turn (1=White, 0=Black)"
        assert encoder.decode_plane_name(14) == "White Castling Rights"
        assert encoder.decode_plane_name(15) == "Black Castling Rights"
        assert encoder.decode_plane_name(16) == "En Passant Square"
        assert encoder.decode_plane_name(17) == "Halfmove Clock"

    def test_visualize_plane_returns_string(self):
        """Test visualize_plane returns valid string"""
        encoder = BoardEncoder()
        board = chess.Board()
        tensor = encoder.encode(board)

        visualization = encoder.visualize_plane(tensor, 0)
        assert isinstance(visualization, str)
        assert "White Pawn" in visualization
        assert len(visualization) > 0


class TestEdgeCases:
    """Test edge cases and special positions"""

    def test_promoted_pieces(self):
        """Test encoding position with promoted pieces"""
        encoder = BoardEncoder()

        # Position with white queen on h8 (promoted pawn)
        board = chess.Board(fen="7Q/8/8/8/8/8/8/4K2k w - - 0 1")
        tensor = encoder.encode(board)

        white_queen = tensor[4]
        assert white_queen[7, 7] == 1.0, "Promoted queen should be encoded"

    def test_maximum_pieces(self):
        """Test encoding position with many pieces"""
        encoder = BoardEncoder()
        board = chess.Board()  # Starting position has 32 pieces

        tensor = encoder.encode(board)

        # Count total pieces encoded
        total_pieces = sum(np.sum(tensor[i]) for i in range(12))
        assert total_pieces == 32, "Should encode all 32 starting pieces"

    def test_checkmate_position(self):
        """Test encoding checkmate position"""
        encoder = BoardEncoder()

        # Back rank mate
        board = chess.Board(fen="6k1/5ppp/8/8/8/8/8/R3K2R w - - 0 1")
        board.push_san("Ra8#")

        tensor = encoder.encode(board)

        # Should still encode correctly even in checkmate
        assert tensor.shape == (18, 8, 8)
        # White rook should be on a8
        white_rooks = tensor[3]
        assert white_rooks[7, 0] == 1.0

    def test_stalemate_position(self):
        """Test encoding stalemate position"""
        encoder = BoardEncoder()

        # Stalemate position
        board = chess.Board(fen="k7/8/1K6/8/8/8/8/1Q6 b - - 0 1")

        tensor = encoder.encode(board)

        # Should encode correctly
        assert tensor.shape == (18, 8, 8)
        # Turn should indicate black to move
        assert np.all(tensor[13] == 0.0)


if __name__ == "__main__":
    """Run tests with pytest or manually"""
    import pytest
    import sys

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))
