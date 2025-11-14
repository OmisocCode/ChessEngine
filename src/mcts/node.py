"""
MCTS Node - Tree node for Monte Carlo Tree Search

This module implements the node structure for the MCTS algorithm.
Each node represents a game state (board position) and stores:
- Visit statistics (how many times visited)
- Value estimates (average outcome)
- Prior probabilities (from neural network policy)
- Child nodes (possible next moves)

The node uses the PUCT (Predictor + Upper Confidence bounds applied to Trees)
formula to balance exploration and exploitation during tree traversal.
"""

import chess
from typing import Optional, Dict, List
import math


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.

    Each node represents a specific board position and maintains statistics
    about its value through simulations. Nodes are connected in a tree
    structure where edges represent moves.

    The node stores:
    - Visit count N(s): How many times this position was visited
    - Total value W(s): Sum of all values backed up through this node
    - Prior probability P(s,a): Neural network's initial move probability
    - Children: Dictionary mapping moves to child nodes

    Attributes:
        board: Chess board position this node represents
        parent: Parent node (None for root)
        move: Move that led from parent to this node (None for root)
        prior: Prior probability P(s,a) from neural network (0.0 for root)
        children: Dictionary mapping chess.Move -> MCTSNode
        visit_count: Number of times this node was visited during search
        total_value: Sum of all values backed up through this node
        is_expanded: Whether this node has been expanded (children created)
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional['MCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        """
        Initialize a new MCTS node.

        Args:
            board: Chess board position for this node
            parent: Parent node (None for root)
            move: Move that led to this position (None for root)
            prior: Prior probability from neural network (0.0 for root)

        Example:
            >>> board = chess.Board()
            >>> root = MCTSNode(board)
            >>> # Create child after move e2e4
            >>> board_after = board.copy()
            >>> move = chess.Move.from_uci("e2e4")
            >>> board_after.push(move)
            >>> child = MCTSNode(board_after, parent=root, move=move, prior=0.15)
        """
        # Board position this node represents
        # We store a copy to avoid issues with board modifications
        self.board = board.copy()

        # Tree structure
        self.parent = parent  # Parent node (None for root)
        self.move = move      # Move that led here (None for root)

        # Prior probability P(s,a) from neural network
        # This represents how "good" the NN thinks this move is
        # before any search has been done
        self.prior = prior

        # Children nodes: maps chess.Move -> MCTSNode
        # Initially empty, populated during expansion
        self.children: Dict[chess.Move, MCTSNode] = {}

        # Visit statistics
        # N(s): Number of times this node was visited during MCTS
        self.visit_count = 0

        # W(s): Total value accumulated through this node
        # Value is from the perspective of the player who MADE the move
        # to reach this node (i.e., the parent's player)
        self.total_value = 0.0

        # Expansion flag
        # Set to True after we've created all child nodes
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """
        Q-value: Average value of this node.

        Q(s,a) = W(s,a) / N(s,a)

        This represents the average outcome when playing through this node.
        Returns 0.0 if node has never been visited.

        Returns:
            Average value in range [-1, 1]
            - +1: This move leads to winning positions
            - 0: This move leads to drawn positions
            - -1: This move leads to losing positions

        Note: Value is from the perspective of the player who made
        the move to reach this node.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_score(self, parent_visit_count: int, c_puct: float = 1.5) -> float:
        """
        Calculate UCT (Upper Confidence bound applied to Trees) score.

        This is the PUCT formula used in AlphaZero:
        UCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))

        The formula balances:
        - Exploitation: Q(s,a) - moves that have been good so far
        - Exploration: The second term - moves that haven't been tried much

        Args:
            parent_visit_count: Number of times parent node was visited
            c_puct: Exploration constant (higher = more exploration)
                    Typical values: 1.0 - 5.0
                    AlphaZero uses 1.5

        Returns:
            UCT score (higher is better for selection)

        The exploration term sqrt(N(parent)) / (1 + N(child)) means:
        - As parent is visited more, we explore more
        - As child is visited more, exploration bonus decreases
        """
        # Exploitation term: average value from this node
        q = self.q_value

        # Exploration term: encourages trying less-visited moves
        # The prior P(s,a) biases exploration toward moves the NN thinks are good
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)

        # Total UCT score
        return q + exploration

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """
        Select best child node using UCT formula.

        This is the SELECTION phase of MCTS.
        Chooses the child with highest UCT score to descend into.

        Args:
            c_puct: Exploration constant for UCT formula

        Returns:
            Child node with highest UCT score

        Raises:
            ValueError: If node has no children

        Note: This should only be called on expanded nodes.
        """
        if not self.children:
            raise ValueError("Cannot select child from node with no children")

        # Calculate UCT score for each child and select maximum
        # We pass our visit_count as parent_visit_count to children
        best_child = max(
            self.children.values(),
            key=lambda child: child.uct_score(self.visit_count, c_puct)
        )

        return best_child

    def expand(self, policy_probs: Dict[chess.Move, float]) -> None:
        """
        Expand this node by creating children for all legal moves.

        This is the EXPANSION phase of MCTS.
        Creates a child node for each legal move in the current position,
        with prior probabilities from the neural network.

        Args:
            policy_probs: Dictionary mapping legal moves to probabilities
                         from neural network policy head

        Note: After expansion, is_expanded is set to True

        Example:
            >>> node = MCTSNode(chess.Board())
            >>> # Get policy from neural network
            >>> policy_probs = {
            >>>     chess.Move.from_uci("e2e4"): 0.25,
            >>>     chess.Move.from_uci("d2d4"): 0.20,
            >>>     # ... other moves ...
            >>> }
            >>> node.expand(policy_probs)
            >>> len(node.children)
            20  # Starting position has 20 legal moves
        """
        # Create a child node for each legal move
        for move in self.board.legal_moves:
            # Get prior probability for this move
            # If move not in policy_probs (shouldn't happen with proper decoder),
            # use a small default probability
            prior = policy_probs.get(move, 1e-6)

            # Create new board with move applied
            child_board = self.board.copy()
            child_board.push(move)

            # Create child node
            child_node = MCTSNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prior
            )

            # Add to children dictionary
            self.children[move] = child_node

        # Mark as expanded
        self.is_expanded = True

    def backup(self, value: float) -> None:
        """
        Backup (backpropagate) value through the tree to the root.

        This is the BACKPROPAGATION phase of MCTS.
        Updates visit counts and values for this node and all ancestors.

        Value is negated at each level because:
        - If a position is good for White (+1), it's bad for Black (-1)
        - We alternate between players going up the tree

        Args:
            value: Evaluation from neural network value head
                   Range [-1, 1] from current player's perspective

        Example:
            >>> # After getting value from neural network
            >>> leaf_node.backup(value=0.7)
            >>> # This updates leaf_node, its parent, grandparent, etc.
        """
        # Start from this node and go up to root
        node = self
        current_value = value

        while node is not None:
            # Update statistics for this node
            node.visit_count += 1
            node.total_value += current_value

            # Move to parent
            node = node.parent

            # Negate value for opponent
            # If position is good for current player (+value),
            # it's bad for opponent (-value)
            current_value = -current_value

    def is_leaf(self) -> bool:
        """
        Check if this is a leaf node (not expanded yet).

        Returns:
            True if node has not been expanded (no children),
            False otherwise
        """
        return not self.is_expanded

    def is_terminal(self) -> bool:
        """
        Check if this node represents a terminal game state.

        A position is terminal if:
        - Checkmate (game over)
        - Stalemate (draw)
        - Insufficient material (draw)
        - Etc.

        Returns:
            True if game is over at this position, False otherwise
        """
        return self.board.is_game_over()

    def get_visit_distribution(self) -> Dict[chess.Move, int]:
        """
        Get visit count distribution over children.

        This is useful for:
        - Selecting final move (choose most visited)
        - Analyzing search tree
        - Creating training data

        Returns:
            Dictionary mapping moves to visit counts

        Example:
            >>> visits = node.get_visit_distribution()
            >>> # Find most visited move
            >>> best_move = max(visits.items(), key=lambda x: x[1])[0]
        """
        return {move: child.visit_count for move, child in self.children.items()}

    def get_policy_distribution(self, temperature: float = 1.0) -> Dict[chess.Move, float]:
        """
        Get move probability distribution based on visit counts.

        Converts visit counts to probabilities using temperature:
        - temperature = 1.0: Proportional to visits
        - temperature → 0: Approaches greedy (select most visited)
        - temperature > 1: More random

        Formula: P(move) = visits^(1/temp) / sum(all visits^(1/temp))

        Args:
            temperature: Controls randomness (default 1.0)

        Returns:
            Dictionary mapping moves to probabilities (sum to 1.0)

        Example:
            >>> # For deterministic play (choose best)
            >>> policy = node.get_policy_distribution(temperature=0.1)
            >>>
            >>> # For more exploration
            >>> policy = node.get_policy_distribution(temperature=1.5)
        """
        if not self.children:
            return {}

        # Get visit counts
        visits = self.get_visit_distribution()

        # Apply temperature
        if temperature == 0:
            # Greedy: give all probability to most visited
            best_move = max(visits.items(), key=lambda x: x[1])[0]
            return {move: 1.0 if move == best_move else 0.0 for move in visits}

        # Temperature > 0: smooth distribution
        visit_counts = [v ** (1.0 / temperature) for v in visits.values()]
        total = sum(visit_counts)

        # Normalize to probabilities
        if total == 0:
            # All visit counts are 0, use uniform
            prob = 1.0 / len(visits)
            return {move: prob for move in visits}

        policy = {}
        for move, count in zip(visits.keys(), visit_counts):
            policy[move] = count / total

        return policy

    def best_child(self) -> Optional['MCTSNode']:
        """
        Get child with highest visit count.

        This is typically used to select the final move after MCTS search.

        Returns:
            Child node with most visits, or None if no children

        Example:
            >>> # After running MCTS search
            >>> best = root.best_child()
            >>> final_move = best.move
            >>> print(f"Best move: {final_move}")
        """
        if not self.children:
            return None

        return max(self.children.values(), key=lambda child: child.visit_count)

    def __repr__(self) -> str:
        """
        String representation of node for debugging.

        Returns:
            String with node information
        """
        move_str = str(self.move) if self.move else "root"
        return (f"MCTSNode(move={move_str}, visits={self.visit_count}, "
                f"q={self.q_value:.3f}, children={len(self.children)})")


if __name__ == "__main__":
    """Demo: create and test MCTS nodes"""
    import chess

    print("MCTS Node Demo")
    print("=" * 60)

    # Create root node
    board = chess.Board()
    root = MCTSNode(board)

    print(f"\nRoot node: {root}")
    print(f"Is leaf: {root.is_leaf()}")
    print(f"Is terminal: {root.is_terminal()}")

    # Simulate expansion with dummy policy
    print("\nExpanding root...")
    policy_probs = {move: 1.0 / 20 for move in board.legal_moves}
    root.expand(policy_probs)

    print(f"Children created: {len(root.children)}")
    print(f"Is leaf: {root.is_leaf()}")

    # Select best child
    print("\nSelecting child using UCT...")
    child = root.select_child(c_puct=1.5)
    print(f"Selected: {child}")

    # Simulate backup
    print("\nBackuping value 0.5 through child...")
    child.backup(0.5)

    print(f"Root visits: {root.visit_count}")
    print(f"Root value: {root.total_value:.3f}")
    print(f"Child visits: {child.visit_count}")
    print(f"Child value: {child.total_value:.3f}")

    # Get visit distribution
    print("\nVisit distribution:")
    visits = root.get_visit_distribution()
    top_5 = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]
    for move, count in top_5:
        print(f"  {move}: {count} visits")

    print("\n" + "=" * 60)
    print("Demo completed!")
