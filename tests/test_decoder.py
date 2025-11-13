"""
Test Suite for Move Decoder

Tests the decoding of neural network policy output into chess moves.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import chess
from src.game.decoder import MoveDecoder, decode_move_greedy, decode_move_sampling


class TestMoveDecoderBasics:
    """Test basic decoder functionality"""

    def test_decoder_initialization(self):
        """Test decoder can be created"""
        decoder = MoveDecoder()
        assert decoder.num_move_planes == 73
        assert decoder.num_squares == 64
        assert decoder.policy_size == 4672

    def test_policy_size_correct(self):
        """Test policy size is 73 × 64 = 4672"""
        decoder = MoveDecoder()
        assert decoder.policy_size == 73 * 64

    def test_random_policy_shape(self):
        """Test random policy creation"""
        decoder = MoveDecoder()
        policy = decoder.create_random_policy()

        assert policy.shape == (4672,)
        assert policy.dtype == np.float32

    def test_uniform_policy_shape(self):
        """Test uniform policy creation"""
        decoder = MoveDecoder()
        policy = decoder.create_uniform_policy()

        assert policy.shape == (4672,)
        assert np.all(policy == 0.0)

    def test_lookup_tables_created(self):
        """Test that lookup tables are populated"""
        decoder = MoveDecoder()

        # Should have mappings for many moves
        assert len(decoder.move_to_policy_index) > 1000
        assert len(decoder.policy_index_to_move_info) > 1000


class TestMoveEncoding:
    """Test encoding of chess moves to policy indices"""

    def test_encode_pawn_push(self):
        """Test encoding pawn push (e2-e4)"""
        decoder = MoveDecoder()
        move = chess.Move.from_uci("e2e4")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None
        assert 0 <= policy_idx < 4672

    def test_encode_knight_move(self):
        """Test encoding knight move (g1-f3)"""
        decoder = MoveDecoder()
        move = chess.Move.from_uci("g1f3")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None

    def test_encode_bishop_diagonal(self):
        """Test encoding bishop diagonal move"""
        decoder = MoveDecoder()
        # Long diagonal move
        move = chess.Move.from_uci("a1h8")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None

    def test_encode_queen_promotion(self):
        """Test encoding queen promotion"""
        decoder = MoveDecoder()
        # Queen promotion
        move = chess.Move.from_uci("e7e8q")

        policy_idx = decoder.encode_move(move)
        # Queen promotion should be encoded (treated as regular move)
        assert policy_idx is not None

    def test_encode_knight_promotion(self):
        """Test encoding knight underpromotion"""
        decoder = MoveDecoder()
        # Knight underpromotion
        move = chess.Move.from_uci("e7e8n")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None

    def test_encode_rook_promotion(self):
        """Test encoding rook underpromotion"""
        decoder = MoveDecoder()
        move = chess.Move.from_uci("e7e8r")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None

    def test_encode_bishop_promotion(self):
        """Test encoding bishop underpromotion"""
        decoder = MoveDecoder()
        move = chess.Move.from_uci("e7e8b")

        policy_idx = decoder.encode_move(move)
        assert policy_idx is not None


class TestMoveDecoding:
    """Test decoding of policy indices to moves"""

    def test_decode_valid_index(self):
        """Test decoding valid policy index"""
        decoder = MoveDecoder()

        # Get a valid index
        move = chess.Move.from_uci("e2e4")
        policy_idx = decoder.encode_move(move)

        # Decode it back
        decoded = decoder.decode_policy_index(policy_idx)
        assert decoded is not None

        from_square, to_square, promotion = decoded
        assert from_square == chess.E2
        assert to_square == chess.E4

    def test_decode_invalid_index(self):
        """Test decoding invalid policy index"""
        decoder = MoveDecoder()

        # Invalid index
        decoded = decoder.decode_policy_index(99999)
        assert decoded is None

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding are consistent"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Test several moves
        test_moves = ["e2e4", "g1f3", "d2d4", "b1c3"]

        for move_uci in test_moves:
            move = chess.Move.from_uci(move_uci)
            policy_idx = decoder.encode_move(move)

            if policy_idx is not None:
                decoded = decoder.decode_policy_index(policy_idx)
                assert decoded is not None

                from_sq, to_sq, promo = decoded
                assert from_sq == move.from_square
                assert to_sq == move.to_square


class TestPolicyToProbabilities:
    """Test conversion of policy to move probabilities"""

    def test_policy_to_probabilities_starting_position(self):
        """Test policy conversion in starting position"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # Should have probabilities for all 20 legal moves
        assert len(move_probs) == 20

        # All probabilities should sum to 1.0
        total_prob = sum(move_probs.values())
        assert abs(total_prob - 1.0) < 1e-5

        # All moves should be legal
        legal_moves = set(board.legal_moves)
        for move in move_probs.keys():
            assert move in legal_moves

    def test_policy_to_probabilities_endgame(self):
        """Test policy conversion in endgame with few moves"""
        decoder = MoveDecoder()
        # King can only move to 5 squares
        board = chess.Board(fen="8/8/8/8/8/8/8/K7 w - - 0 1")

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # King on a1 has 3 legal moves (b1, b2, a2)
        assert len(move_probs) == 3

        # Probabilities sum to 1
        assert abs(sum(move_probs.values()) - 1.0) < 1e-5

    def test_policy_to_probabilities_no_moves(self):
        """Test policy conversion when no legal moves (stalemate)"""
        decoder = MoveDecoder()
        # Stalemate position
        board = chess.Board(fen="k7/8/1K6/8/8/8/8/1Q6 b - - 0 1")

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # No legal moves
        assert len(move_probs) == 0

    def test_temperature_effect(self):
        """Test that temperature affects probability distribution"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Create policy with one strong preference
        policy = np.full(4672, -10.0, dtype=np.float32)
        # Make e2e4 strongly preferred
        e2e4 = chess.Move.from_uci("e2e4")
        e2e4_idx = decoder.encode_move(e2e4)
        policy[e2e4_idx] = 10.0

        # Low temperature (greedy)
        probs_low_temp = decoder.policy_to_move_probabilities(policy, board, temperature=0.1)
        e2e4_prob_low = probs_low_temp[e2e4]

        # High temperature (more random)
        probs_high_temp = decoder.policy_to_move_probabilities(policy, board, temperature=2.0)
        e2e4_prob_high = probs_high_temp[e2e4]

        # Low temperature should give higher probability to best move
        assert e2e4_prob_low > e2e4_prob_high


class TestMoveSelection:
    """Test move selection methods"""

    def test_select_move_greedy_starting_position(self):
        """Test greedy move selection"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        move = decoder.select_move_greedy(policy, board)

        # Should return a legal move
        assert move is not None
        assert move in board.legal_moves

    def test_select_move_greedy_deterministic(self):
        """Test that greedy selection is deterministic"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()

        # Select twice with same policy
        move1 = decoder.select_move_greedy(policy, board)
        move2 = decoder.select_move_greedy(policy, board)

        # Should return same move
        assert move1 == move2

    def test_select_move_sampling(self):
        """Test sampling move selection"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        move = decoder.select_move_sampling(policy, board)

        # Should return a legal move
        assert move is not None
        assert move in board.legal_moves

    def test_select_move_sampling_temperature(self):
        """Test sampling with different temperatures"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()

        # Sample with different temperatures
        move_low = decoder.select_move_sampling(policy, board, temperature=0.1)
        move_high = decoder.select_move_sampling(policy, board, temperature=2.0)

        # Both should be legal
        assert move_low in board.legal_moves
        assert move_high in board.legal_moves

    def test_select_move_no_legal_moves(self):
        """Test move selection when no legal moves"""
        decoder = MoveDecoder()
        # Stalemate
        board = chess.Board(fen="k7/8/1K6/8/8/8/8/1Q6 b - - 0 1")

        policy = decoder.create_random_policy()

        move_greedy = decoder.select_move_greedy(policy, board)
        move_sample = decoder.select_move_sampling(policy, board)

        # Both should return None
        assert move_greedy is None
        assert move_sample is None


class TestTopMoves:
    """Test top-k move selection"""

    def test_get_top_moves_starting_position(self):
        """Test getting top-k moves"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        top_moves = decoder.get_top_moves(policy, board, top_k=5)

        # Should return 5 moves
        assert len(top_moves) == 5

        # Each should be (move, probability) tuple
        for move, prob in top_moves:
            assert isinstance(move, chess.Move)
            assert 0.0 <= prob <= 1.0
            assert move in board.legal_moves

        # Should be sorted by probability (descending)
        probs = [prob for _, prob in top_moves]
        assert probs == sorted(probs, reverse=True)

    def test_get_top_moves_more_than_available(self):
        """Test getting more moves than available"""
        decoder = MoveDecoder()
        # Position with only 3 legal moves
        board = chess.Board(fen="8/8/8/8/8/8/8/K7 w - - 0 1")

        policy = decoder.create_random_policy()
        top_moves = decoder.get_top_moves(policy, board, top_k=10)

        # Should return only 3 (all available moves)
        assert len(top_moves) == 3

    def test_get_top_moves_empty_position(self):
        """Test top moves with no legal moves"""
        decoder = MoveDecoder()
        # Stalemate
        board = chess.Board(fen="k7/8/1K6/8/8/8/8/1Q6 b - - 0 1")

        policy = decoder.create_random_policy()
        top_moves = decoder.get_top_moves(policy, board, top_k=5)

        # Should return empty list
        assert len(top_moves) == 0


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_decode_move_greedy_function(self):
        """Test decode_move_greedy convenience function"""
        board = chess.Board()
        policy = np.random.randn(4672).astype(np.float32)

        move = decode_move_greedy(policy, board)

        assert move is not None
        assert move in board.legal_moves

    def test_decode_move_sampling_function(self):
        """Test decode_move_sampling convenience function"""
        board = chess.Board()
        policy = np.random.randn(4672).astype(np.float32)

        move = decode_move_sampling(policy, board, temperature=1.0)

        assert move is not None
        assert move in board.legal_moves


class TestVisualization:
    """Test visualization methods"""

    def test_move_to_string(self):
        """Test move to string conversion"""
        decoder = MoveDecoder()

        move = chess.Move.from_uci("e2e4")
        move_str = decoder.move_to_string(move)

        assert move_str == "e2e4"

    def test_visualize_policy_returns_string(self):
        """Test visualize_policy returns valid string"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        viz = decoder.visualize_policy(policy, board, top_k=5)

        assert isinstance(viz, str)
        assert len(viz) > 0
        assert "Policy Visualization" in viz


class TestSpecialMoves:
    """Test handling of special moves"""

    def test_castling_kingside(self):
        """Test encoding castling moves"""
        decoder = MoveDecoder()

        # Castling is encoded as king move
        move = chess.Move.from_uci("e1g1")  # White kingside castling
        policy_idx = decoder.encode_move(move)

        assert policy_idx is not None

    def test_castling_queenside(self):
        """Test queenside castling"""
        decoder = MoveDecoder()

        move = chess.Move.from_uci("e1c1")  # White queenside castling
        policy_idx = decoder.encode_move(move)

        assert policy_idx is not None

    def test_en_passant_capture(self):
        """Test en passant encoding"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Create en passant situation
        board.push_san("e4")
        board.push_san("a6")
        board.push_san("e5")
        board.push_san("d5")  # Creates ep opportunity

        # En passant capture
        ep_move = chess.Move.from_uci("e5d6")
        policy_idx = decoder.encode_move(ep_move)

        assert policy_idx is not None

        # Should be legal
        assert ep_move in board.legal_moves


class TestPolicyFiltering:
    """Test that only legal moves are considered"""

    def test_illegal_moves_filtered(self):
        """Test that illegal moves are not in probability distribution"""
        decoder = MoveDecoder()
        board = chess.Board()

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # All returned moves must be legal
        legal_moves = set(board.legal_moves)
        for move in move_probs.keys():
            assert move in legal_moves

    def test_only_legal_moves_have_probability(self):
        """Test that all legal moves get some probability"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Uniform policy (all equal)
        policy = decoder.create_uniform_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # Should have entry for each legal move
        legal_moves = set(board.legal_moves)
        returned_moves = set(move_probs.keys())

        # All legal moves should be represented
        assert returned_moves == legal_moves


class TestEdgeCases:
    """Test edge cases and special positions"""

    def test_position_with_one_legal_move(self):
        """Test position where only one move is legal"""
        decoder = MoveDecoder()

        # Position where only king move is legal (escaping check)
        board = chess.Board(fen="k7/8/8/8/8/8/8/K6R b - - 0 1")
        # Black king must move (not in check but limited moves)

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # Should have probabilities
        assert len(move_probs) > 0

        # Total probability should be 1.0
        assert abs(sum(move_probs.values()) - 1.0) < 1e-5

    def test_forced_mate_in_one(self):
        """Test position with mate in one"""
        decoder = MoveDecoder()

        # Back rank mate available
        board = chess.Board(fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

        policy = decoder.create_random_policy()

        # Make Ra8# have highest probability artificially
        ra8 = chess.Move.from_uci("a1a8")
        ra8_idx = decoder.encode_move(ra8)

        policy = np.full(4672, -10.0, dtype=np.float32)
        policy[ra8_idx] = 10.0

        best_move = decoder.select_move_greedy(policy, board)

        # Should select the mate
        assert best_move == ra8

    def test_checkmate_position(self):
        """Test that checkmate position has no moves"""
        decoder = MoveDecoder()

        # Checkmate position
        board = chess.Board(fen="6k1/5ppp/8/8/8/8/8/R6R w - - 0 1")
        board.push_san("Ra8#")

        policy = decoder.create_random_policy()
        move_probs = decoder.policy_to_move_probabilities(policy, board)

        # No legal moves
        assert len(move_probs) == 0


class TestConsistency:
    """Test consistency across multiple calls"""

    def test_greedy_selection_consistent(self):
        """Test that greedy selection gives same result"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Fixed policy
        np.random.seed(42)
        policy = decoder.create_random_policy()

        # Multiple selections
        moves = [decoder.select_move_greedy(policy, board) for _ in range(10)]

        # All should be the same
        assert len(set(str(m) for m in moves)) == 1

    def test_sampling_varies(self):
        """Test that sampling gives different results"""
        decoder = MoveDecoder()
        board = chess.Board()

        # Uniform policy
        policy = decoder.create_uniform_policy()

        # Multiple samples
        np.random.seed(None)  # Reset seed
        moves = [decoder.select_move_sampling(policy, board) for _ in range(20)]

        # With uniform policy and 20 legal moves, should get some variety
        unique_moves = len(set(str(m) for m in moves))
        assert unique_moves > 1  # Should have at least 2 different moves


if __name__ == "__main__":
    """Run tests with pytest or manually"""
    import pytest
    import sys

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))
