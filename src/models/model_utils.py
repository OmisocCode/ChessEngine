"""
Model Utilities - Save/Load functionality for ChessNet

This module provides utilities for saving and loading neural network models,
including checkpoints with training state.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


def save_model(
    model: torch.nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model to disk with optional training state.

    This function saves:
    - Model state dict (weights and biases)
    - Model configuration (architecture parameters)
    - Optional: optimizer state
    - Optional: training metadata (epoch, loss, etc.)

    Args:
        model: ChessNet model to save
        filepath: Path where to save the model (.pth or .pt)
        optimizer: Optional optimizer to save state from
        epoch: Optional epoch number
        metadata: Optional dictionary with additional info

    Example:
        >>> model = ChessNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> save_model(
        >>>     model,
        >>>     'checkpoints/model_epoch10.pth',
        >>>     optimizer=optimizer,
        >>>     epoch=10,
        >>>     metadata={'loss': 0.523, 'accuracy': 0.78}
        >>> )
    """
    # Ensure directory exists
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    # Prepare checkpoint dictionary
    checkpoint = {
        # Model architecture and weights
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),

        # Timestamp
        'timestamp': datetime.now().isoformat(),
    }

    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer).__name__

    # Add epoch if provided
    if epoch is not None:
        checkpoint['epoch'] = epoch

    # Add metadata if provided
    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Save to disk
    torch.save(checkpoint, filepath)

    print(f"✓ Model saved to {filepath}")
    if epoch is not None:
        print(f"  Epoch: {epoch}")
    if metadata:
        print(f"  Metadata: {metadata}")


def load_model(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model from disk.

    This function can:
    1. Load weights into an existing model instance
    2. Return checkpoint data to create a new model
    3. Optionally load optimizer state

    Args:
        filepath: Path to the saved model file
        model: Optional model instance to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model onto ('cpu', 'cuda', etc.)

    Returns:
        Dictionary containing:
        - 'model_config': Architecture configuration
        - 'epoch': Training epoch (if saved)
        - 'metadata': Additional metadata (if saved)
        - 'checkpoint': Full checkpoint dictionary

    Example:
        >>> # Load into existing model
        >>> model = ChessNet()
        >>> load_model('checkpoints/best_model.pth', model=model)
        >>>
        >>> # Load and create new model
        >>> checkpoint = load_model('checkpoints/best_model.pth')
        >>> config = checkpoint['model_config']
        >>> model = ChessNet(**config)
        >>> model.load_state_dict(checkpoint['checkpoint']['model_state_dict'])
    """
    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights if model provided
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"✓ Model weights loaded from {filepath}")

    # Load optimizer state if optimizer provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Optimizer state loaded")

    # Prepare return dictionary
    result = {
        'model_config': checkpoint.get('model_config', {}),
        'checkpoint': checkpoint
    }

    # Add optional fields
    if 'epoch' in checkpoint:
        result['epoch'] = checkpoint['epoch']
        print(f"  Epoch: {checkpoint['epoch']}")

    if 'metadata' in checkpoint:
        result['metadata'] = checkpoint['metadata']
        print(f"  Metadata: {checkpoint['metadata']}")

    if 'timestamp' in checkpoint:
        result['timestamp'] = checkpoint['timestamp']
        print(f"  Saved: {checkpoint['timestamp']}")

    return result


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str = 'data/checkpoints',
    filename: Optional[str] = None
) -> str:
    """
    Save a training checkpoint with automatic naming.

    Convenience function that creates a checkpoint file with metadata
    about the current training state.

    Args:
        model: Model to save
        optimizer: Optimizer to save state from
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints (default: data/checkpoints)
        filename: Optional custom filename (default: checkpoint_epoch{epoch}.pth)

    Returns:
        Path to the saved checkpoint file

    Example:
        >>> for epoch in range(num_epochs):
        >>>     # Training loop...
        >>>     loss = train_one_epoch(model, optimizer)
        >>>
        >>>     # Save checkpoint every 5 epochs
        >>>     if (epoch + 1) % 5 == 0:
        >>>         save_checkpoint(model, optimizer, epoch, loss)
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f"checkpoint_epoch{epoch:03d}.pth"

    filepath = os.path.join(checkpoint_dir, filename)

    # Save with metadata
    metadata = {
        'loss': float(loss),
        'epoch': epoch,
    }

    save_model(
        model=model,
        filepath=filepath,
        optimizer=optimizer,
        epoch=epoch,
        metadata=metadata
    )

    return filepath


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Convenience function to resume training from a saved checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load onto

    Returns:
        Dictionary with checkpoint information including epoch and loss

    Example:
        >>> model = ChessNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> # Resume training
        >>> checkpoint = load_checkpoint(
        >>>     'data/checkpoints/checkpoint_epoch10.pth',
        >>>     model,
        >>>     optimizer
        >>> )
        >>> start_epoch = checkpoint['epoch'] + 1
        >>> print(f"Resuming from epoch {start_epoch}")
    """
    return load_model(filepath, model, optimizer, device)


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and parameter count.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with:
        - 'num_parameters': Total trainable parameters
        - 'size_mb': Estimated size in megabytes (float32)
        - 'size_bytes': Size in bytes

    Example:
        >>> model = ChessNet()
        >>> info = get_model_size(model)
        >>> print(f"Parameters: {info['num_parameters']:,}")
        >>> print(f"Size: {info['size_mb']:.2f} MB")
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size (assuming float32 = 4 bytes per parameter)
    size_bytes = num_params * 4
    size_mb = size_bytes / (1024 * 1024)

    return {
        'num_parameters': num_params,
        'size_bytes': size_bytes,
        'size_mb': size_mb
    }


def export_to_onnx(
    model: torch.nn.Module,
    filepath: str,
    input_shape: tuple = (1, 18, 8, 8)
) -> None:
    """
    Export model to ONNX format for deployment.

    ONNX (Open Neural Network Exchange) is a format that allows
    models to be used in different frameworks and platforms.

    Args:
        model: Model to export
        filepath: Path where to save ONNX file (.onnx)
        input_shape: Shape of input tensor (default: (1, 18, 8, 8))

    Example:
        >>> model = ChessNet()
        >>> export_to_onnx(model, 'models/chess_net.onnx')
    """
    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['board_input'],
        output_names=['policy_output', 'value_output'],
        dynamic_axes={
            'board_input': {0: 'batch_size'},
            'policy_output': {0: 'batch_size'},
            'value_output': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to ONNX: {filepath}")


def list_checkpoints(checkpoint_dir: str = 'data/checkpoints') -> list:
    """
    List all checkpoint files in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        List of checkpoint filenames sorted by modification time (newest first)

    Example:
        >>> checkpoints = list_checkpoints()
        >>> for ckpt in checkpoints:
        >>>     print(ckpt)
    """
    if not os.path.exists(checkpoint_dir):
        return []

    # Find all .pth and .pt files
    checkpoint_files = []
    for ext in ['.pth', '.pt']:
        checkpoint_files.extend(Path(checkpoint_dir).glob(f'*{ext}'))

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return [f.name for f in checkpoint_files]


def find_best_checkpoint(
    checkpoint_dir: str = 'data/checkpoints',
    metric: str = 'loss',
    minimize: bool = True
) -> Optional[str]:
    """
    Find the best checkpoint based on a metric.

    Args:
        checkpoint_dir: Directory with checkpoints
        metric: Metric name to compare (e.g., 'loss', 'accuracy')
        minimize: If True, find minimum; if False, find maximum

    Returns:
        Filename of best checkpoint, or None if no checkpoints found

    Example:
        >>> # Find checkpoint with lowest loss
        >>> best = find_best_checkpoint(metric='loss', minimize=True)
        >>> if best:
        >>>     model = ChessNet()
        >>>     load_checkpoint(f'data/checkpoints/{best}', model)
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        return None

    best_value = float('inf') if minimize else float('-inf')
    best_checkpoint = None

    for ckpt in checkpoints:
        filepath = os.path.join(checkpoint_dir, ckpt)
        try:
            checkpoint_data = torch.load(filepath, map_location='cpu')

            if 'metadata' in checkpoint_data and metric in checkpoint_data['metadata']:
                value = checkpoint_data['metadata'][metric]

                if minimize and value < best_value:
                    best_value = value
                    best_checkpoint = ckpt
                elif not minimize and value > best_value:
                    best_value = value
                    best_checkpoint = ckpt

        except Exception as e:
            print(f"Warning: Could not load {ckpt}: {e}")
            continue

    return best_checkpoint


if __name__ == "__main__":
    """Demo: model save/load functionality"""
    print("Model Utilities Demo")
    print("=" * 60)

    # This demo requires torch and the ChessNet class
    try:
        from chess_net import ChessNet

        # Create a model
        print("\n1. Creating model...")
        model = ChessNet()

        # Get model info
        info = get_model_size(model)
        print(f"   Parameters: {info['num_parameters']:,}")
        print(f"   Size: {info['size_mb']:.2f} MB")

        # Save model
        print("\n2. Saving model...")
        os.makedirs('data/checkpoints', exist_ok=True)
        save_model(
            model,
            'data/checkpoints/demo_model.pth',
            metadata={'demo': True, 'loss': 0.5}
        )

        # Load model
        print("\n3. Loading model...")
        loaded_model = ChessNet()
        checkpoint = load_model('data/checkpoints/demo_model.pth', model=loaded_model)

        # List checkpoints
        print("\n4. Available checkpoints:")
        checkpoints = list_checkpoints('data/checkpoints')
        for ckpt in checkpoints:
            print(f"   - {ckpt}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")

    except ImportError as e:
        print(f"\nSkipping demo: {e}")
        print("Run this from the src/models directory or ensure chess_net.py is available")
