"""
Monte Carlo Tree Search - Main algorithm implementation

This module implements the MCTS algorithm for chess move selection.
MCTS builds a search tree by repeatedly simulating games and collecting
statistics about which moves lead to good outcomes.

The algorithm has 4 main phases:
1. SELECTION: Traverse tree using UCT to find a leaf node
2. EXPANSION: Create children for the leaf node
3. EVALUATION: Use neural network to evaluate the position
4. BACKPROPAGATION: Update statistics back to root

This implementation follows AlphaZero's approach:
- Uses neural network for evaluation (no random playouts)
- Uses policy network to guide tree search
- Adds Dirichlet noise to root for exploration
"""

import chess
import numpy as np
from typing import Optional, Callable, Dict, Tuple
from src.mcts.node import MCTSNode


class MCTS:
    """
    Monte Carlo Tree Search for chess.

    This class implements the MCTS algorithm that builds a search tree
    by simulating games and learning which moves are best.

    The search process:
    1. Start at root (current position)
    2. Repeat num_simulations times:
       a. Select a path down the tree using UCT
       b. Expand the leaf by creating children
       c. Evaluate the leaf with neural network
       d. Backpropagate the value up the tree
    3. Return best move based on visit counts

    Attributes:
        c_puct: Exploration constant for UCT formula
                Higher values = more exploration
                Typical range: 1.0 - 5.0
        num_simulations: Number of MCTS simulations to run
                        More simulations = stronger play but slower
        temperature: Controls randomness in final move selection
                    1.0 = proportional to visits
                    →0 = greedy (most visited)
                    >1 = more random
        dirichlet_alpha: Parameter for Dirichlet noise at root
                        Controls exploration at root node
        dirichlet_epsilon: Weight of Dirichlet noise
                          0.25 = 25% noise, 75% policy
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        num_simulations: int = 50,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize MCTS with hyperparameters.

        Args:
            c_puct: Exploration constant for UCT (default: 1.5)
            num_simulations: Number of simulations per move (default: 50)
            temperature: Move selection temperature (default: 1.0)
            dirichlet_alpha: Dirichlet noise alpha (default: 0.3)
            dirichlet_epsilon: Dirichlet noise weight (default: 0.25)

        Example:
            >>> # Create MCTS with default settings
            >>> mcts = MCTS()
            >>>
            >>> # More simulations for stronger play
            >>> mcts_strong = MCTS(num_simulations=200)
            >>>
            >>> # More exploration
            >>> mcts_explore = MCTS(c_puct=3.0)
        """
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(
        self,
        board: chess.Board,
        evaluate_fn: Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]
    ) -> MCTSNode:
        """
        Run MCTS search from a given board position.

        This is the main entry point for MCTS search.
        Runs num_simulations iterations of the 4 MCTS phases,
        then returns the root node with accumulated statistics.

        Args:
            board: Starting chess position
            evaluate_fn: Function that evaluates a position
                        Input: chess.Board
                        Output: (policy_dict, value)
                        - policy_dict: {move: probability}
                        - value: position evaluation [-1, 1]

        Returns:
            Root MCTSNode with search statistics

        Example:
            >>> def evaluate(board):
            >>>     # Simple random evaluation
            >>>     policy = {m: 1.0/len(list(board.legal_moves))
            >>>              for m in board.legal_moves}
            >>>     value = 0.0
            >>>     return policy, value
            >>>
            >>> mcts = MCTS(num_simulations=100)
            >>> root = mcts.search(chess.Board(), evaluate_fn=evaluate)
            >>> best_move = root.best_child().move
        """
        # Create root node
        root = MCTSNode(board)

        # Evaluate root position to get initial policy
        policy_probs, root_value = evaluate_fn(board)

        # Add Dirichlet noise to root policy for exploration
        # This encourages the search to try different moves at the root
        policy_probs = self._add_dirichlet_noise(policy_probs)

        # Expand root node
        root.expand(policy_probs)

        # Run simulations
        for simulation in range(self.num_simulations):
            # Each simulation: selection → expansion → evaluation → backprop
            self._simulate(root, evaluate_fn)

        return root

    def _simulate(
        self,
        root: MCTSNode,
        evaluate_fn: Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]
    ) -> None:
        """
        Run one MCTS simulation (all 4 phases).

        This method implements one complete iteration of MCTS:
        1. SELECTION: Walk down tree selecting best children
        2. EXPANSION: Create children for leaf node
        3. EVALUATION: Get value from neural network
        4. BACKPROPAGATION: Update statistics up the tree

        Args:
            root: Root node to start simulation from
            evaluate_fn: Function to evaluate positions
        """
        node = root
        search_path = [node]  # Track path for backpropagation

        # PHASE 1: SELECTION
        # Traverse tree using UCT until we reach a leaf or terminal node
        while not node.is_leaf() and not node.is_terminal():
            # Select child with best UCT score
            node = node.select_child(self.c_puct)
            search_path.append(node)

        # PHASE 2: EXPANSION
        # If we reached a non-terminal leaf, expand it
        if not node.is_terminal():
            # Evaluate the position to get policy and value
            policy_probs, value = evaluate_fn(node.board)

            # Expand node by creating children
            node.expand(policy_probs)

            # Use the value from evaluation
        else:
            # Terminal node: use actual game result
            value = self._get_terminal_value(node.board)

        # PHASE 4: BACKPROPAGATION
        # Update all nodes in the search path
        # Note: We start from the leaf and go up to root
        # The value is from the perspective of the player at the leaf,
        # so it needs to be negated as we go up
        node.backup(value)

    def _add_dirichlet_noise(
        self,
        policy: Dict[chess.Move, float]
    ) -> Dict[chess.Move, float]:
        """
        Add Dirichlet noise to policy for exploration.

        Dirichlet noise adds randomness to the policy at the root node,
        which encourages exploring different moves during self-play training.

        Formula: policy = (1 - ε) * policy + ε * noise
        where noise ~ Dir(α)

        Args:
            policy: Original policy probabilities

        Returns:
            Policy with Dirichlet noise added

        Note: This should only be used at the root node during self-play.
        During actual play, you may want to disable this.
        """
        if not policy:
            return policy

        # Get moves and probabilities
        moves = list(policy.keys())
        probs = np.array(list(policy.values()))

        # Generate Dirichlet noise
        # Alpha controls the concentration:
        # - Small alpha (0.03): concentrated, few moves get high noise
        # - Large alpha (1.0): uniform, all moves get similar noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        # Mix original policy with noise
        # epsilon controls the mixing ratio
        mixed_probs = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise

        # Create new policy dictionary
        noisy_policy = {move: prob for move, prob in zip(moves, mixed_probs)}

        return noisy_policy

    def _get_terminal_value(self, board: chess.Board) -> float:
        """
        Get value for terminal positions (checkmate, stalemate, draw).

        Returns value from perspective of current player:
        - +1: Current player wins (opponent is checkmated)
        - 0: Draw (stalemate, insufficient material, etc.)
        - -1: Current player loses (current player is checkmated)

        Args:
            board: Terminal board position

        Returns:
            Value in [-1, 1]
        """
        # Check game outcome
        outcome = board.outcome()

        if outcome is None:
            # Not actually terminal (shouldn't happen)
            return 0.0

        # Checkmate: winner exists
        if outcome.winner is not None:
            # If current player's turn and they're checkmated, they lose
            # But checkmate means the game is over, so current player is the one who got mated
            if outcome.winner == board.turn:
                return 1.0  # Current side won
            else:
                return -1.0  # Current side lost
        else:
            # Draw (stalemate, insufficient material, etc.)
            return 0.0

    def select_move(
        self,
        board: chess.Board,
        evaluate_fn: Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]],
        return_node: bool = False
    ) -> chess.Move:
        """
        Select best move using MCTS search.

        Convenience method that runs search and returns the best move.

        Args:
            board: Current board position
            evaluate_fn: Position evaluation function
            return_node: If True, return (move, root_node), else just move

        Returns:
            Best chess.Move (or tuple of (move, node) if return_node=True)

        Example:
            >>> mcts = MCTS(num_simulations=100)
            >>> move = mcts.select_move(board, evaluate_fn)
            >>> board.push(move)
        """
        # Run search
        root = self.search(board, evaluate_fn)

        # Get best child
        best_child = root.best_child()

        if best_child is None:
            # No legal moves (game over)
            raise ValueError("No legal moves available")

        if return_node:
            return best_child.move, root
        else:
            return best_child.move

    def get_action_probs(
        self,
        board: chess.Board,
        evaluate_fn: Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]],
        temperature: Optional[float] = None
    ) -> Dict[chess.Move, float]:
        """
        Get move probability distribution after MCTS search.

        This is useful for:
        - Training: creating training data with MCTS-improved policy
        - Sampling: selecting moves according to search probabilities

        Args:
            board: Current board position
            evaluate_fn: Position evaluation function
            temperature: Override default temperature (optional)

        Returns:
            Dictionary mapping moves to probabilities (sum to 1.0)

        Example:
            >>> mcts = MCTS(num_simulations=100)
            >>> probs = mcts.get_action_probs(board, evaluate_fn)
            >>> # Sample move according to probabilities
            >>> moves = list(probs.keys())
            >>> move_probs = list(probs.values())
            >>> chosen = np.random.choice(moves, p=move_probs)
        """
        # Use instance temperature if not overridden
        if temperature is None:
            temperature = self.temperature

        # Run search
        root = self.search(board, evaluate_fn)

        # Get policy from visit distribution
        policy = root.get_policy_distribution(temperature=temperature)

        return policy


def create_mcts(config: Optional[Dict] = None) -> MCTS:
    """
    Factory function to create MCTS with configuration.

    Args:
        config: Dictionary with MCTS parameters (optional)

    Returns:
        MCTS instance

    Example:
        >>> # Default MCTS
        >>> mcts = create_mcts()
        >>>
        >>> # Custom configuration
        >>> config = {
        >>>     'num_simulations': 200,
        >>>     'c_puct': 2.0,
        >>>     'temperature': 1.0
        >>> }
        >>> mcts = create_mcts(config)
    """
    if config is None:
        config = {}

    return MCTS(
        c_puct=config.get('c_puct', 1.5),
        num_simulations=config.get('num_simulations', 50),
        temperature=config.get('temperature', 1.0),
        dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
        dirichlet_epsilon=config.get('dirichlet_epsilon', 0.25)
    )


if __name__ == "__main__":
    """Demo: run MCTS with random evaluation"""
    import chess

    print("MCTS Algorithm Demo")
    print("=" * 60)

    def random_evaluate(board: chess.Board):
        """Simple random evaluation function for demo"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0

        # Uniform policy
        policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
        # Random value
        value = np.random.uniform(-1, 1)

        return policy, value

    # Create MCTS
    mcts = MCTS(num_simulations=10)  # Small number for demo

    print("\nRunning MCTS search on starting position...")
    print(f"Simulations: {mcts.num_simulations}")

    # Starting position
    board = chess.Board()
    print(f"\nBoard:\n{board}")

    # Run search
    root = mcts.search(board, random_evaluate)

    print(f"\nSearch completed!")
    print(f"Root visits: {root.visit_count}")
    print(f"Root value: {root.q_value:.3f}")

    # Get visit distribution
    print("\nTop 5 moves by visits:")
    visits = root.get_visit_distribution()
    top_5 = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]
    for move, count in top_5:
        child = root.children[move]
        print(f"  {move}: {count} visits, Q={child.q_value:.3f}")

    # Select best move
    best_child = root.best_child()
    print(f"\nBest move: {best_child.move}")

    # Get policy distribution
    policy = root.get_policy_distribution(temperature=1.0)
    print(f"\nPolicy probabilities (sum={sum(policy.values()):.3f})")

    print("\n" + "=" * 60)
    print("Demo completed!")
