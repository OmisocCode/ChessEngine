"""
Replay Buffer for Training Data Storage and Sampling

This module implements a replay buffer (also called experience replay) for
storing and sampling training data from self-play games.

The replay buffer serves several important purposes:
1. **Decorrelation**: Training examples from one game are highly correlated
   (consecutive positions are similar). By storing many games and sampling
   randomly, we break this correlation and improve training stability.

2. **Data efficiency**: We can reuse training data multiple times by sampling
   different batches from the buffer.

3. **Curriculum learning**: By keeping recent games and discarding old ones,
   the network trains on data from its current skill level, not from when it
   was much weaker.

The buffer stores TrainingExample objects from self-play and provides
methods to sample random batches for training.

Typical usage in training loop:
1. Play N self-play games
2. Add all training examples to replay buffer
3. Sample M batches from replay buffer
4. Train network on these batches
5. Repeat
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training data.

    This buffer stores training examples (board, policy, value) from self-play
    games and provides efficient random sampling for training.

    Key features:
    - Fixed maximum size (FIFO: oldest data is discarded when full)
    - Random sampling for decorrelation
    - Batch sampling with shuffling
    - Statistics tracking

    Attributes:
        max_size: Maximum number of training examples to store
        buffer: Deque storing TrainingExample objects
        num_games_added: Counter for number of games added (for stats)

    Design choices:
    - Using deque for O(1) append and automatic size limiting
    - Storing TrainingExample objects directly (not separated)
    - Random sampling without replacement within a batch
    """

    def __init__(self, max_size: int = 50000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of training examples to store
                     When buffer is full, oldest examples are removed (FIFO)
                     Typical values: 10,000 - 100,000

        Example:
            >>> buffer = ReplayBuffer(max_size=50000)
            >>> # Add training data
            >>> buffer.add_games(training_examples)
            >>> # Sample batch for training
            >>> batch = buffer.sample_batch(batch_size=64)
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.num_games_added = 0

    def add_example(self, example) -> None:
        """
        Add a single training example to the buffer.

        Args:
            example: TrainingExample object with:
                    - board_tensor: (18, 8, 8) array
                    - mcts_policy: (4672,) array
                    - value: float
                    - move_number: int

        Note: If buffer is at max_size, the oldest example is automatically
        removed (deque handles this with maxlen).
        """
        self.buffer.append(example)

    def add_games(self, examples: List, game_infos: Optional[List] = None) -> None:
        """
        Add training examples from one or more self-play games.

        This is the typical way to add data to the buffer: after generating
        self-play games, add all their training examples at once.

        Args:
            examples: List of TrainingExample objects from self-play
            game_infos: Optional list of game info dicts (for logging)

        Example:
            >>> # Generate self-play games
            >>> examples, infos = worker.generate_games(num_games=50)
            >>> # Add to replay buffer
            >>> buffer.add_games(examples, infos)
            >>> print(f"Buffer size: {len(buffer)}")
        """
        for example in examples:
            self.add_example(example)

        # Update counter (count number of games, not examples)
        if game_infos is not None:
            self.num_games_added += len(game_infos)

    def sample_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample a random batch of training examples.

        This is the main method used during training. It samples batch_size
        random examples from the buffer and returns them as numpy arrays
        ready for neural network training.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Tuple of (board_batch, policy_batch, value_batch) if buffer has enough data:
            - board_batch: (batch_size, 18, 8, 8) - Board encodings
            - policy_batch: (batch_size, 4672) - MCTS-improved policies
            - value_batch: (batch_size,) - Game outcomes

            Returns None if buffer doesn't have enough examples

        Example:
            >>> buffer = ReplayBuffer()
            >>> # ... add data ...
            >>> board, policy, value = buffer.sample_batch(batch_size=64)
            >>> # Train network
            >>> loss = train_step(model, board, policy, value)

        Note: Sampling is without replacement within a batch (no duplicates)
        """
        # Check if buffer has enough data
        if len(self.buffer) < batch_size:
            return None

        # Sample random indices without replacement
        # This ensures no duplicate examples in a batch
        indices = random.sample(range(len(self.buffer)), batch_size)

        # Gather examples
        sampled_examples = [self.buffer[i] for i in indices]

        # Convert to numpy arrays for neural network training
        # Stack creates arrays with batch dimension first
        board_batch = np.stack([ex.board_tensor for ex in sampled_examples])  # (batch_size, 18, 8, 8)
        policy_batch = np.stack([ex.mcts_policy for ex in sampled_examples])  # (batch_size, 4672)
        value_batch = np.array([ex.value for ex in sampled_examples], dtype=np.float32)  # (batch_size,)

        return board_batch, policy_batch, value_batch

    def sample_multiple_batches(
        self,
        batch_size: int,
        num_batches: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample multiple random batches from the buffer.

        This is useful for training on multiple batches in one call.
        Each batch is independently sampled, so there may be some overlap
        between batches (same example could appear in multiple batches).

        Args:
            batch_size: Size of each batch
            num_batches: Number of batches to sample

        Returns:
            List of (board_batch, policy_batch, value_batch) tuples

        Example:
            >>> batches = buffer.sample_multiple_batches(batch_size=64, num_batches=10)
            >>> for board, policy, value in batches:
            >>>     loss = train_step(model, board, policy, value)
        """
        batches = []

        for _ in range(num_batches):
            batch = self.sample_batch(batch_size)
            if batch is None:
                break  # Not enough data
            batches.append(batch)

        return batches

    def clear(self) -> None:
        """
        Clear all data from the buffer.

        Useful when you want to start fresh with new data.
        """
        self.buffer.clear()
        self.num_games_added = 0

    def __len__(self) -> int:
        """
        Get current number of examples in buffer.

        Returns:
            Number of training examples currently stored

        Example:
            >>> buffer = ReplayBuffer(max_size=50000)
            >>> buffer.add_games(examples)
            >>> print(f"Buffer contains {len(buffer)} examples")
        """
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough data for training.

        Args:
            min_size: Minimum number of examples needed

        Returns:
            True if buffer has at least min_size examples

        Example:
            >>> if buffer.is_ready(min_size=1000):
            >>>     batch = buffer.sample_batch(64)
            >>>     train_step(model, batch)
        """
        return len(self.buffer) >= min_size

    def get_statistics(self) -> dict:
        """
        Get statistics about the replay buffer.

        Returns:
            Dictionary with buffer statistics:
            - 'size': Current number of examples
            - 'max_size': Maximum capacity
            - 'usage_percent': Percentage of capacity used
            - 'num_games_added': Total games added (lifetime)

        Example:
            >>> stats = buffer.get_statistics()
            >>> print(f"Buffer: {stats['size']}/{stats['max_size']} ({stats['usage_percent']:.1f}%)")
        """
        size = len(self.buffer)
        usage_percent = (size / self.max_size * 100) if self.max_size > 0 else 0

        return {
            'size': size,
            'max_size': self.max_size,
            'usage_percent': usage_percent,
            'num_games_added': self.num_games_added
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer (advanced version).

    This is an extension of ReplayBuffer that samples examples based on
    priority rather than uniformly. Examples with higher training loss
    are sampled more often.

    Benefits:
    - Focuses training on "hard" examples (where network makes mistakes)
    - Can improve sample efficiency

    Drawbacks:
    - More complex implementation
    - Slightly slower sampling
    - Risk of overfitting to hard examples

    Note: This is an advanced feature and may not be necessary for initial
    training. The basic ReplayBuffer is usually sufficient.

    Implementation note: This is a placeholder for future enhancement.
    For now, it just inherits from ReplayBuffer (uniform sampling).
    To implement true prioritization, you would:
    1. Store priorities alongside examples
    2. Update priorities based on TD-error or loss
    3. Sample proportional to priorities
    4. Apply importance sampling weights
    """

    def __init__(self, max_size: int = 50000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)

        Note: Currently behaves like regular ReplayBuffer.
        Full prioritization implementation is left as future work.
        """
        super().__init__(max_size)
        self.alpha = alpha
        # TODO: Add priority tracking (SumTree or similar data structure)
        # TODO: Add importance sampling weights

    # TODO: Override sample_batch to use priorities
    # TODO: Add update_priorities method


def create_replay_buffer(config: Optional[dict] = None) -> ReplayBuffer:
    """
    Factory function to create replay buffer from configuration.

    Args:
        config: Dictionary with buffer parameters:
               - 'max_size': Maximum buffer size (default: 50000)
               - 'buffer_type': 'standard' or 'prioritized' (default: 'standard')

    Returns:
        ReplayBuffer instance

    Example:
        >>> config = {'max_size': 100000, 'buffer_type': 'standard'}
        >>> buffer = create_replay_buffer(config)
    """
    if config is None:
        config = {}

    max_size = config.get('max_size', 50000)
    buffer_type = config.get('buffer_type', 'standard')

    if buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(max_size)
    else:
        return ReplayBuffer(max_size)


if __name__ == "__main__":
    """Demo: test replay buffer functionality"""
    print("Replay Buffer Demo")
    print("=" * 60)

    # Create dummy training examples for testing
    from src.training.self_play import TrainingExample
    import numpy as np

    print("\nCreating test data...")

    # Create some fake training examples
    test_examples = []
    for i in range(100):
        example = TrainingExample(
            board_tensor=np.random.rand(18, 8, 8).astype(np.float32),
            mcts_policy=np.random.rand(4672).astype(np.float32),
            value=np.random.choice([-1.0, 0.0, 1.0]),
            move_number=i % 50 + 1
        )
        test_examples.append(example)

    print(f"Created {len(test_examples)} test examples")

    # Create replay buffer
    print("\nInitializing replay buffer...")
    buffer = ReplayBuffer(max_size=500)

    # Add examples
    print("Adding examples to buffer...")
    buffer.add_games(test_examples)

    # Show statistics
    stats = buffer.get_statistics()
    print(f"\nBuffer statistics:")
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Usage: {stats['usage_percent']:.1f}%")

    # Sample a batch
    print("\nSampling batch...")
    batch_size = 16
    batch_data = buffer.sample_batch(batch_size)

    if batch_data is not None:
        board_batch, policy_batch, value_batch = batch_data
        print(f"Batch sampled successfully:")
        print(f"  Board batch shape: {board_batch.shape}")
        print(f"  Policy batch shape: {policy_batch.shape}")
        print(f"  Value batch shape: {value_batch.shape}")
        print(f"  Value distribution: {np.bincount((value_batch + 1).astype(int))}")

    # Sample multiple batches
    print("\nSampling multiple batches...")
    batches = buffer.sample_multiple_batches(batch_size=16, num_batches=5)
    print(f"Sampled {len(batches)} batches")

    # Test buffer overflow
    print("\nTesting buffer overflow (adding more than max_size)...")
    overflow_examples = []
    for i in range(600):  # More than max_size=500
        example = TrainingExample(
            board_tensor=np.random.rand(18, 8, 8).astype(np.float32),
            mcts_policy=np.random.rand(4672).astype(np.float32),
            value=0.0,
            move_number=1
        )
        overflow_examples.append(example)

    buffer.add_games(overflow_examples)
    stats = buffer.get_statistics()
    print(f"After overflow:")
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Usage: {stats['usage_percent']:.1f}%")
    print(f"  (Oldest examples were automatically discarded)")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nReplay buffer is ready for use in training loop.")
