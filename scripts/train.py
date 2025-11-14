#!/usr/bin/env python3
"""
Main Training Script for Chess AI

This script implements the complete AlphaZero-style training loop:
1. Self-play: Generate games using current neural network + MCTS
2. Training: Update neural network on self-play data
3. Iteration: Repeat for many iterations

The training cycle:
- Each iteration:
  1. Generate N self-play games with current network
  2. Add all training examples to replay buffer
  3. Train network on M batches sampled from replay buffer
  4. Save checkpoint
  5. Evaluate (optional)

Over many iterations, the network improves:
- Initially: Random/weak play
- After training: MCTS distills good policy into network
- Network gets better → MCTS searches better → generates better data → repeat

This creates a self-improvement loop where the AI teaches itself to play
by learning from its own games.

Configuration:
- Edit parameters in this script or use command-line arguments
- See config.yaml for hyperparameter defaults
- Checkpoints saved to checkpoints/ directory
- Logs saved to logs/ directory

Usage:
    python scripts/train.py --iterations 20 --games-per-iter 50
    python scripts/train.py --resume checkpoints/model_iter_5.pt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
import time
from datetime import datetime
import json

# Import all training components
from src.game.encoder import BoardEncoder
from src.game.decoder import MoveDecoder
from src.models.chess_net import ChessNet
from src.training.self_play import SelfPlayWorker
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer import ChessTrainer


class TrainingConfig:
    """
    Configuration for training run.

    This class holds all hyperparameters for the training process.
    Adjust these values to control training behavior.
    """

    def __init__(self):
        # Training iterations
        self.num_iterations = 20  # Number of train-play-update cycles
        self.games_per_iteration = 50  # Self-play games per iteration
        self.training_batches_per_iteration = 100  # Training batches per iteration
        self.batch_size = 64  # Batch size for training

        # MCTS configuration for self-play
        self.mcts_simulations = 50  # MCTS simulations per move during self-play
        self.mcts_c_puct = 1.5  # Exploration constant
        self.temperature_threshold = 30  # Move number to reduce temperature
        self.high_temperature = 1.0  # Temperature for first N moves (exploration)
        self.low_temperature = 0.1  # Temperature after N moves (exploitation)

        # Neural network training
        self.learning_rate = 0.001  # Learning rate for Adam optimizer
        self.weight_decay = 1e-4  # L2 regularization strength
        self.policy_weight = 1.0  # Weight for policy loss
        self.value_weight = 1.0  # Weight for value loss

        # Replay buffer
        self.replay_buffer_size = 50000  # Maximum training examples to store

        # Checkpointing
        self.checkpoint_dir = "checkpoints"  # Directory to save checkpoints
        self.save_every = 1  # Save checkpoint every N iterations
        self.keep_checkpoints = 5  # Keep last N checkpoints (delete older)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Logging
        self.log_dir = "logs"
        self.verbose = True

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d):
        """Create config from dictionary"""
        config = cls()
        for k, v in d.items():
            setattr(config, k, v)
        return config


def setup_training(config: TrainingConfig, resume_from: str = None):
    """
    Setup training components: model, encoder, decoder, buffer, trainer.

    Args:
        config: TrainingConfig with hyperparameters
        resume_from: Optional path to checkpoint to resume from

    Returns:
        Tuple of (model, encoder, decoder, replay_buffer, trainer, start_iteration)
    """
    print("=" * 70)
    print(" " * 20 + "CHESS AI TRAINING")
    print("=" * 70)
    print(f"\nDevice: {config.device}")

    # Create components
    print("\nInitializing components...")
    encoder = BoardEncoder()
    decoder = MoveDecoder()
    model = ChessNet()
    print(f"✓ Model created ({model.count_parameters():,} parameters)")

    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    print(f"✓ Replay buffer created (max size: {config.replay_buffer_size:,})")

    # Create trainer
    trainer = ChessTrainer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        policy_weight=config.policy_weight,
        value_weight=config.value_weight,
        device=config.device
    )
    print(f"✓ Trainer created (lr={config.learning_rate})")

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = trainer.load_checkpoint(resume_from)
        start_iteration = checkpoint.get('iteration', 0) + 1
        print(f"✓ Resumed from iteration {checkpoint.get('iteration', 0)}")

    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    return model, encoder, decoder, replay_buffer, trainer, start_iteration


def run_iteration(
    iteration: int,
    config: TrainingConfig,
    model,
    encoder,
    decoder,
    replay_buffer,
    trainer
):
    """
    Run one training iteration: self-play + training.

    One iteration consists of:
    1. Generate self-play games with current network
    2. Add training examples to replay buffer
    3. Train network on batches from replay buffer
    4. Log metrics

    Args:
        iteration: Current iteration number (0-indexed)
        config: Training configuration
        model: Neural network
        encoder: BoardEncoder
        decoder: MoveDecoder
        replay_buffer: ReplayBuffer for storing data
        trainer: ChessTrainer

    Returns:
        Dictionary with iteration statistics
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration + 1}/{config.num_iterations}")
    print(f"{'='*70}")

    iteration_start = time.time()

    # PHASE 1: SELF-PLAY
    # Generate games using current network + MCTS
    print(f"\n[1/2] Self-Play Generation")
    print(f"  Generating {config.games_per_iteration} games...")
    print(f"  MCTS simulations per move: {config.mcts_simulations}")

    # Create self-play worker with current network
    mcts_config = {
        'num_simulations': config.mcts_simulations,
        'c_puct': config.mcts_c_puct,
        'temperature_threshold': config.temperature_threshold,
        'high_temperature': config.high_temperature,
        'low_temperature': config.low_temperature
    }

    worker = SelfPlayWorker(encoder, decoder, model, mcts_config=mcts_config)

    # Generate games
    selfplay_start = time.time()
    examples, game_infos = worker.generate_games(
        num_games=config.games_per_iteration,
        verbose=False  # Reduce verbosity
    )
    selfplay_time = time.time() - selfplay_start

    # Add to replay buffer
    replay_buffer.add_games(examples, game_infos)

    # Self-play statistics
    num_examples = len(examples)
    avg_moves = sum(info['num_moves'] for info in game_infos) / len(game_infos)
    white_wins = sum(1 for info in game_infos if '1-0' in info['result'])
    black_wins = sum(1 for info in game_infos if '0-1' in info['result'])
    draws = len(game_infos) - white_wins - black_wins

    print(f"\n  Self-play completed in {selfplay_time:.1f}s")
    print(f"  Training examples generated: {num_examples:,}")
    print(f"  Average game length: {avg_moves:.1f} moves")
    print(f"  Results: {white_wins}W-{draws}D-{black_wins}L")
    print(f"  Replay buffer: {len(replay_buffer):,}/{config.replay_buffer_size:,}")

    # PHASE 2: TRAINING
    # Train network on data from replay buffer
    print(f"\n[2/2] Neural Network Training")
    print(f"  Training on {config.training_batches_per_iteration} batches...")

    training_start = time.time()
    epoch_stats = trainer.train_epoch(
        replay_buffer,
        batch_size=config.batch_size,
        num_batches=config.training_batches_per_iteration,
        verbose=False
    )
    training_time = time.time() - training_start

    print(f"\n  Training completed in {training_time:.1f}s")
    if epoch_stats:
        print(f"  Average total loss: {epoch_stats['avg_total_loss']:.4f}")
        print(f"  Average policy loss: {epoch_stats['avg_policy_loss']:.4f}")
        print(f"  Average value loss: {epoch_stats['avg_value_loss']:.4f}")
        print(f"  Policy accuracy: {epoch_stats['avg_policy_accuracy']:.2%}")
        print(f"  Value accuracy: {epoch_stats['avg_value_accuracy']:.2%}")

    iteration_time = time.time() - iteration_start

    # Compile iteration statistics
    iteration_stats = {
        'iteration': iteration,
        'selfplay_time': selfplay_time,
        'training_time': training_time,
        'total_time': iteration_time,
        'num_examples': num_examples,
        'avg_game_length': avg_moves,
        'white_wins': white_wins,
        'draws': draws,
        'black_wins': black_wins,
        'buffer_size': len(replay_buffer),
        **epoch_stats
    }

    return iteration_stats


def save_iteration_checkpoint(
    iteration: int,
    config: TrainingConfig,
    trainer: ChessTrainer,
    iteration_stats: dict
):
    """
    Save checkpoint after iteration.

    Args:
        iteration: Current iteration number
        config: Training configuration
        trainer: ChessTrainer with model
        iteration_stats: Statistics from this iteration
    """
    checkpoint_path = f"{config.checkpoint_dir}/model_iter_{iteration + 1}.pt"

    trainer.save_checkpoint(
        checkpoint_path,
        epoch=iteration,
        additional_info={
            'iteration': iteration,
            'config': config.to_dict(),
            'iteration_stats': iteration_stats
        }
    )

    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    # Clean up old checkpoints if needed
    if config.keep_checkpoints > 0:
        checkpoints = sorted(Path(config.checkpoint_dir).glob("model_iter_*.pt"))
        if len(checkpoints) > config.keep_checkpoints:
            for old_checkpoint in checkpoints[:-config.keep_checkpoints]:
                old_checkpoint.unlink()
                print(f"  Removed old checkpoint: {old_checkpoint.name}")


def save_training_log(config: TrainingConfig, all_iteration_stats: list):
    """
    Save training log with all iteration statistics.

    Args:
        config: Training configuration
        all_iteration_stats: List of iteration statistics dicts
    """
    log_path = f"{config.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    log_data = {
        'config': config.to_dict(),
        'iterations': all_iteration_stats
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"✓ Training log saved: {log_path}")


def main():
    """
    Main training loop.

    Runs the complete training process for num_iterations iterations.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Chess AI')
    parser.add_argument('--iterations', type=int, default=20, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=50, help='Self-play games per iteration')
    parser.add_argument('--batches', type=int, default=100, help='Training batches per iteration')
    parser.add_argument('--simulations', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig()

    # Override with command-line arguments
    if args.iterations:
        config.num_iterations = args.iterations
    if args.games:
        config.games_per_iteration = args.games
    if args.batches:
        config.training_batches_per_iteration = args.batches
    if args.simulations:
        config.mcts_simulations = args.simulations
    if args.device:
        config.device = args.device

    # Setup training
    model, encoder, decoder, replay_buffer, trainer, start_iteration = setup_training(
        config, resume_from=args.resume
    )

    print(f"\nTraining configuration:")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Games per iteration: {config.games_per_iteration}")
    print(f"  Training batches per iteration: {config.training_batches_per_iteration}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  MCTS simulations: {config.mcts_simulations}")

    # Main training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")

    all_iteration_stats = []
    training_start = time.time()

    try:
        for iteration in range(start_iteration, config.num_iterations):
            # Run one iteration
            iteration_stats = run_iteration(
                iteration, config, model, encoder, decoder, replay_buffer, trainer
            )

            all_iteration_stats.append(iteration_stats)

            # Save checkpoint
            if (iteration + 1) % config.save_every == 0:
                save_iteration_checkpoint(iteration, config, trainer, iteration_stats)

            # Print progress summary
            elapsed = time.time() - training_start
            avg_iter_time = elapsed / (iteration - start_iteration + 1)
            remaining_iters = config.num_iterations - iteration - 1
            eta = remaining_iters * avg_iter_time

            print(f"\nProgress: {iteration + 1}/{config.num_iterations} iterations")
            print(f"  Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        save_iteration_checkpoint(iteration, config, trainer, iteration_stats)

    # Training complete
    total_time = time.time() - training_start

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Iterations completed: {len(all_iteration_stats)}")
    print(f"Total training examples generated: {sum(s['num_examples'] for s in all_iteration_stats):,}")

    # Save final checkpoint
    save_iteration_checkpoint(
        len(all_iteration_stats) - 1 + start_iteration,
        config, trainer,
        all_iteration_stats[-1]
    )

    # Save training log
    save_training_log(config, all_iteration_stats)

    print("\n✓ Training finished successfully!")
    print(f"  Model checkpoints: {config.checkpoint_dir}/")
    print(f"  Training logs: {config.log_dir}/")


if __name__ == "__main__":
    main()
