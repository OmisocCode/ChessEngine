"""
Neural Network Trainer for Chess AI

This module implements the training loop for the chess neural network.
It handles the supervised learning from self-play data, computing losses
for both policy and value heads, and updating the network weights.

The training process follows AlphaZero's approach:
1. Sample a batch of training examples from replay buffer
2. Forward pass through network to get predictions
3. Compute losses:
   - Policy loss: Cross-entropy between MCTS policy and network policy
   - Value loss: Mean squared error between game outcome and network value
4. Backpropagate gradients and update weights
5. Track metrics and save checkpoints

Loss functions:
- **Policy loss**: We want the network to predict what MCTS would choose
  L_policy = -Σ p_mcts * log(p_network)  (cross-entropy)
  This teaches the network to imitate MCTS search

- **Value loss**: We want the network to predict game outcomes
  L_value = (z - v)²  (mean squared error)
  where z is the actual game outcome (-1, 0, +1) and v is prediction

- **Total loss**: L = L_policy + c * L_value
  where c is a weight parameter (typically c=1.0)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
import time
from pathlib import Path


class ChessTrainer:
    """
    Trainer for the chess neural network.

    This class manages the training process: computing losses, updating weights,
    tracking metrics, and saving checkpoints.

    The trainer uses two loss functions:
    1. Policy loss (cross-entropy): Teaches network to predict MCTS policy
    2. Value loss (MSE): Teaches network to predict game outcomes

    Attributes:
        model: ChessNet neural network
        optimizer: PyTorch optimizer (Adam)
        device: 'cpu' or 'cuda'
        policy_weight: Weight for policy loss in total loss
        value_weight: Weight for value loss in total loss
        metrics: Dictionary tracking training metrics
    """

    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: ChessNet neural network to train
            learning_rate: Learning rate for optimizer (default: 0.001)
                          Adam optimizer is relatively insensitive to this
                          Typical range: 0.0001 - 0.01
            weight_decay: L2 regularization strength (default: 1e-4)
                         Helps prevent overfitting
            policy_weight: Weight for policy loss (default: 1.0)
            value_weight: Weight for value loss (default: 1.0)
            device: 'cpu' or 'cuda' for GPU training

        Example:
            >>> model = ChessNet()
            >>> trainer = ChessTrainer(model, learning_rate=0.001)
            >>> # Train on a batch
            >>> loss = trainer.train_batch(boards, policies, values)
        """
        self.model = model
        self.device = device
        self.policy_weight = policy_weight
        self.value_weight = value_weight

        # Move model to device
        self.model.to(device)

        # Create optimizer
        # Adam is the standard choice for neural network training
        # It adapts learning rates for each parameter automatically
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Metrics tracking
        self.metrics = {
            'train_steps': 0,
            'total_loss_history': [],
            'policy_loss_history': [],
            'value_loss_history': [],
            'policy_accuracy_history': [],
            'value_accuracy_history': []
        }

    def compute_losses(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute policy loss, value loss, and total loss.

        Args:
            policy_logits: Network policy output (batch_size, 4672)
                          Raw logits before softmax
            value_pred: Network value output (batch_size, 1)
                       Predicted position value in [-1, 1]
            policy_target: MCTS-improved policy target (batch_size, 4672)
                          Probability distribution from MCTS visit counts
            value_target: Game outcome target (batch_size,)
                         Actual game result: +1, 0, or -1

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)

        Loss formulas:
        - Policy: Cross-entropy between MCTS policy and network policy
          This is the KL divergence: Σ p_target * log(p_target / p_pred)
          Implemented as: -Σ p_target * log_softmax(logits)

        - Value: Mean squared error between game outcome and prediction
          (z - v)² where z ∈ {-1, 0, 1} and v ∈ [-1, 1]

        - Total: policy_weight * policy_loss + value_weight * value_loss
        """
        # Policy loss: Cross-entropy between MCTS policy and network policy
        # We use log_softmax + negative log likelihood for numerical stability
        # This is equivalent to: -Σ target * log(softmax(logits))

        # Step 1: Apply log_softmax to logits
        # log_softmax(x) = log(exp(x) / Σexp(x)) = x - log(Σexp(x))
        # This is more numerically stable than log(softmax(x))
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)

        # Step 2: Multiply by target and sum (cross-entropy)
        # We want: -Σ target * log(pred)
        # Since we have log(pred), we do: -Σ target * log_pred
        policy_loss = -torch.sum(policy_target * log_probs, dim=1)

        # Average over batch
        policy_loss = policy_loss.mean()

        # Value loss: Mean squared error
        # Remove extra dimension from value_pred: (batch, 1) -> (batch,)
        value_pred = value_pred.squeeze(1)

        # MSE: mean((prediction - target)²)
        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)

        # Total loss: weighted combination
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss

        return total_loss, policy_loss, value_loss

    def train_batch(
        self,
        boards: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray
    ) -> Dict[str, float]:
        """
        Train on a single batch of data.

        This is one training step: forward pass, compute loss, backward pass,
        update weights.

        Args:
            boards: Batch of board encodings (batch_size, 18, 8, 8)
            policies: Batch of MCTS policies (batch_size, 4672)
            values: Batch of game outcomes (batch_size,)

        Returns:
            Dictionary with loss values:
            - 'total_loss': Combined loss
            - 'policy_loss': Policy cross-entropy loss
            - 'value_loss': Value MSE loss
            - 'policy_accuracy': Accuracy of top-1 policy prediction
            - 'value_accuracy': Percentage of value predictions within 0.5

        Example:
            >>> # Sample batch from replay buffer
            >>> boards, policies, values = replay_buffer.sample_batch(64)
            >>> # Train on batch
            >>> losses = trainer.train_batch(boards, policies, values)
            >>> print(f"Total loss: {losses['total_loss']:.4f}")
        """
        # Set model to training mode
        # This enables dropout (if any) and batch norm training mode
        self.model.train()

        # Convert numpy arrays to PyTorch tensors
        boards_tensor = torch.from_numpy(boards).to(self.device)
        policies_tensor = torch.from_numpy(policies).to(self.device)
        values_tensor = torch.from_numpy(values).to(self.device)

        # Zero gradients from previous step
        # PyTorch accumulates gradients, so we need to clear them
        self.optimizer.zero_grad()

        # Forward pass: get network predictions
        policy_logits, value_pred = self.model(boards_tensor)

        # Compute losses
        total_loss, policy_loss, value_loss = self.compute_losses(
            policy_logits, value_pred,
            policies_tensor, values_tensor
        )

        # Backward pass: compute gradients
        # This computes ∂loss/∂weights for all parameters
        total_loss.backward()

        # Gradient clipping (optional, helps with stability)
        # Limits the magnitude of gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update weights using optimizer
        # This performs: weights = weights - learning_rate * gradients
        self.optimizer.step()

        # Compute accuracy metrics (for monitoring)
        with torch.no_grad():
            # Policy accuracy: percentage where network's top choice matches MCTS top choice
            policy_preds = torch.argmax(policy_logits, dim=1)
            policy_targets = torch.argmax(policies_tensor, dim=1)
            policy_accuracy = (policy_preds == policy_targets).float().mean().item()

            # Value accuracy: percentage within 0.5 of target
            value_pred_squeezed = value_pred.squeeze(1)
            value_error = torch.abs(value_pred_squeezed - values_tensor)
            value_accuracy = (value_error < 0.5).float().mean().item()

        # Update metrics
        self.metrics['train_steps'] += 1
        self.metrics['total_loss_history'].append(total_loss.item())
        self.metrics['policy_loss_history'].append(policy_loss.item())
        self.metrics['value_loss_history'].append(value_loss.item())
        self.metrics['policy_accuracy_history'].append(policy_accuracy)
        self.metrics['value_accuracy_history'].append(value_accuracy)

        # Return losses as dictionary
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_accuracy': policy_accuracy,
            'value_accuracy': value_accuracy
        }

    def train_epoch(
        self,
        replay_buffer,
        batch_size: int = 64,
        num_batches: int = 100,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch (multiple batches).

        An epoch consists of sampling and training on num_batches batches
        from the replay buffer.

        Args:
            replay_buffer: ReplayBuffer containing training data
            batch_size: Size of each batch
            num_batches: Number of batches to train on
            verbose: Whether to print progress

        Returns:
            Dictionary with average metrics over the epoch

        Example:
            >>> # Train for one epoch
            >>> epoch_stats = trainer.train_epoch(
            >>>     replay_buffer, batch_size=64, num_batches=100
            >>> )
            >>> print(f"Avg loss: {epoch_stats['avg_total_loss']:.4f}")
        """
        if not replay_buffer.is_ready(batch_size):
            if verbose:
                print(f"Replay buffer not ready (size: {len(replay_buffer)}, need: {batch_size})")
            return {}

        # Track losses for averaging
        epoch_losses = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'policy_accuracy': [],
            'value_accuracy': []
        }

        start_time = time.time()

        # Train on num_batches batches
        for batch_idx in range(num_batches):
            # Sample batch
            batch_data = replay_buffer.sample_batch(batch_size)

            if batch_data is None:
                if verbose:
                    print(f"Could not sample batch (buffer size: {len(replay_buffer)})")
                break

            boards, policies, values = batch_data

            # Train on batch
            losses = self.train_batch(boards, policies, values)

            # Collect losses
            for key in epoch_losses:
                epoch_losses[key].append(losses[key])

            # Print progress
            if verbose and (batch_idx + 1) % 20 == 0:
                avg_loss = np.mean(epoch_losses['total_loss'][-20:])
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time

        # Compute averages
        avg_stats = {
            'avg_' + key: np.mean(values) for key, values in epoch_losses.items()
        }
        avg_stats['batches_trained'] = len(epoch_losses['total_loss'])
        avg_stats['time_taken'] = elapsed

        if verbose:
            print(f"Epoch completed in {elapsed:.1f}s")
            print(f"  Avg total loss: {avg_stats['avg_total_loss']:.4f}")
            print(f"  Avg policy loss: {avg_stats['avg_policy_loss']:.4f}")
            print(f"  Avg value loss: {avg_stats['avg_value_loss']:.4f}")
            print(f"  Policy accuracy: {avg_stats['avg_policy_accuracy']:.2%}")
            print(f"  Value accuracy: {avg_stats['avg_value_accuracy']:.2%}")

        return avg_stats

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        additional_info: Optional[Dict] = None
    ) -> None:
        """
        Save training checkpoint.

        Saves model weights, optimizer state, and training metrics.
        This allows resuming training from where you left off.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            additional_info: Optional dict with extra info to save

        Example:
            >>> trainer.save_checkpoint(
            >>>     'checkpoints/model_epoch_10.pt',
            >>>     epoch=10,
            >>>     additional_info={'iteration': 5}
            >>> )
        """
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'train_steps': self.metrics['train_steps']
        }

        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)

        # Save
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint.

        Restores model weights, optimizer state, and training metrics.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint info (epoch, etc.)

        Example:
            >>> info = trainer.load_checkpoint('checkpoints/model_epoch_10.pt')
            >>> print(f"Resumed from epoch {info['epoch']}")
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore metrics
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']

        return checkpoint

    def get_metrics_summary(self) -> Dict:
        """
        Get summary of training metrics.

        Returns:
            Dictionary with recent and overall metrics

        Example:
            >>> summary = trainer.get_metrics_summary()
            >>> print(f"Train steps: {summary['train_steps']}")
            >>> print(f"Recent avg loss: {summary['recent_avg_loss']:.4f}")
        """
        if self.metrics['train_steps'] == 0:
            return {'train_steps': 0}

        # Compute averages over last 100 steps
        recent_window = 100
        recent_losses = self.metrics['total_loss_history'][-recent_window:]

        summary = {
            'train_steps': self.metrics['train_steps'],
            'recent_avg_loss': np.mean(recent_losses) if recent_losses else 0,
            'overall_avg_loss': np.mean(self.metrics['total_loss_history']),
            'best_loss': np.min(self.metrics['total_loss_history']) if self.metrics['total_loss_history'] else float('inf')
        }

        return summary


def create_trainer(model, config: Optional[Dict] = None) -> ChessTrainer:
    """
    Factory function to create trainer from configuration.

    Args:
        model: ChessNet neural network
        config: Dictionary with trainer parameters:
               - 'learning_rate': Learning rate (default: 0.001)
               - 'weight_decay': L2 regularization (default: 1e-4)
               - 'policy_weight': Policy loss weight (default: 1.0)
               - 'value_weight': Value loss weight (default: 1.0)
               - 'device': 'cpu' or 'cuda' (default: 'cpu')

    Returns:
        ChessTrainer instance

    Example:
        >>> config = {'learning_rate': 0.001, 'device': 'cuda'}
        >>> trainer = create_trainer(model, config)
    """
    if config is None:
        config = {}

    return ChessTrainer(
        model=model,
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4),
        policy_weight=config.get('policy_weight', 1.0),
        value_weight=config.get('value_weight', 1.0),
        device=config.get('device', 'cpu')
    )


if __name__ == "__main__":
    """Demo: test trainer with dummy data"""
    print("Trainer Demo")
    print("=" * 60)

    # Import dependencies
    from src.models.chess_net import ChessNet
    from src.training.replay_buffer import ReplayBuffer
    from src.training.self_play import TrainingExample
    import numpy as np

    print("\nInitializing model and trainer...")
    model = ChessNet()
    trainer = ChessTrainer(model, learning_rate=0.001)

    print("Creating dummy training data...")
    buffer = ReplayBuffer(max_size=1000)

    # Create dummy examples
    dummy_examples = []
    for i in range(200):
        example = TrainingExample(
            board_tensor=np.random.rand(18, 8, 8).astype(np.float32),
            mcts_policy=np.random.rand(4672).astype(np.float32),
            value=np.random.choice([-1.0, 0.0, 1.0]),
            move_number=i % 50 + 1
        )
        # Normalize policy
        example.mcts_policy = example.mcts_policy / example.mcts_policy.sum()
        dummy_examples.append(example)

    buffer.add_games(dummy_examples)
    print(f"Replay buffer size: {len(buffer)}")

    print("\nTraining for one epoch...")
    stats = trainer.train_epoch(
        buffer,
        batch_size=16,
        num_batches=10,
        verbose=True
    )

    print("\nTraining metrics summary:")
    summary = trainer.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nTrainer is ready for use in training loop.")
