"""
Chess Neural Network - AlphaZero-style architecture

This module implements a convolutional neural network for chess position evaluation.
The network has a dual-head architecture:
- Policy head: predicts move probabilities (4672 possible moves)
- Value head: evaluates position quality (-1 to +1)

The architecture is inspired by AlphaZero but simplified for educational purposes
and feasibility on consumer hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Convolutional block with Batch Normalization and ReLU activation.

    This is the basic building block of the network. Each block consists of:
    1. 2D Convolution (preserves spatial dimensions with padding)
    2. Batch Normalization (stabilizes training)
    3. ReLU activation (introduces non-linearity)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kernel_size: Size of the convolution kernel (typically 3x3)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ConvBlock, self).__init__()

        # Calculate padding to preserve spatial dimensions
        # For kernel_size=3, padding=1 keeps the 8x8 board size unchanged
        padding = kernel_size // 2

        # Convolutional layer: applies learned filters to extract features
        # - in_channels: number of feature maps from previous layer
        # - out_channels: number of feature maps this layer produces
        # - kernel_size: size of the sliding window (3x3 is standard)
        # - padding: adds zeros around edges to maintain size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False  # Bias is redundant with BatchNorm
        )

        # Batch Normalization: normalizes activations for stable training
        # This helps the network train faster and more reliably
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape (batch, in_channels, 8, 8)

        Returns:
            Output tensor of shape (batch, out_channels, 8, 8)
        """
        # Apply convolution
        x = self.conv(x)

        # Apply batch normalization
        x = self.bn(x)

        # Apply ReLU activation: max(0, x)
        # This introduces non-linearity, allowing the network to learn complex patterns
        x = F.relu(x)

        return x


class PolicyHead(nn.Module):
    """
    Policy head: predicts move probabilities.

    This head takes the shared convolutional features and outputs a probability
    distribution over all possible moves (4672 total).

    Architecture:
    1. 1x1 convolution to reduce channels
    2. Batch normalization + ReLU
    3. Flatten to vector
    4. Output: raw logits (before softmax) for each possible move

    Args:
        in_channels: Number of input feature maps from the body
        num_moves: Total number of possible moves (4672)
    """

    def __init__(self, in_channels: int, num_moves: int = 4672):
        super(PolicyHead, self).__init__()

        # 1x1 convolution: reduces feature maps while preserving spatial information
        # This is more efficient than using fully connected layers directly
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=73,  # 73 move planes per square
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(73)

        # Final fully connected layer that outputs move logits
        # Input size: 73 planes × 8 × 8 squares = 4672
        # Output size: 4672 (one logit per possible move)
        self.fc = nn.Linear(73 * 8 * 8, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.

        Args:
            x: Feature tensor from body, shape (batch, in_channels, 8, 8)

        Returns:
            Policy logits, shape (batch, 4672)
            Note: These are raw scores, not probabilities. Apply softmax for probabilities.
        """
        # Apply 1x1 convolution to get 73 feature planes
        # Shape: (batch, 73, 8, 8)
        x = self.conv(x)

        # Batch normalize and activate
        x = self.bn(x)
        x = F.relu(x)

        # Flatten spatial dimensions: (batch, 73, 8, 8) → (batch, 73*8*8)
        # This converts the 3D feature maps into a 1D vector
        x = x.view(x.size(0), -1)  # -1 automatically calculates the size

        # Apply final linear layer to get move logits
        # Shape: (batch, 4672)
        policy_logits = self.fc(x)

        return policy_logits


class ValueHead(nn.Module):
    """
    Value head: evaluates the position quality.

    This head predicts a single value in range [-1, 1]:
    - +1: winning position for the side to move
    - 0: equal/drawn position
    - -1: losing position for the side to move

    Architecture:
    1. 1x1 convolution to reduce channels
    2. Batch normalization + ReLU
    3. Flatten
    4. Hidden fully connected layer
    5. Output layer with tanh activation

    Args:
        in_channels: Number of input feature maps from the body
        hidden_size: Size of the hidden layer (default: 32)
    """

    def __init__(self, in_channels: int, hidden_size: int = 32):
        super(ValueHead, self).__init__()

        # 1x1 convolution to reduce feature maps to a single plane
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,  # Just one feature map for value
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(1)

        # Hidden layer: processes the flattened features
        # Input: 1 × 8 × 8 = 64 values
        self.fc1 = nn.Linear(1 * 8 * 8, hidden_size)

        # Output layer: produces final position evaluation
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.

        Args:
            x: Feature tensor from body, shape (batch, in_channels, 8, 8)

        Returns:
            Position values, shape (batch, 1)
            Values are in range [-1, 1] due to tanh activation.
        """
        # Apply 1x1 convolution to get single feature plane
        # Shape: (batch, 1, 8, 8)
        x = self.conv(x)

        # Batch normalize and activate
        x = self.bn(x)
        x = F.relu(x)

        # Flatten: (batch, 1, 8, 8) → (batch, 64)
        x = x.view(x.size(0), -1)

        # Apply hidden layer with ReLU
        # Shape: (batch, hidden_size)
        x = F.relu(self.fc1(x))

        # Apply output layer
        # Shape: (batch, 1)
        x = self.fc2(x)

        # Apply tanh to bound output to [-1, 1]
        # tanh squashes any value to the range [-1, 1]
        value = torch.tanh(x)

        return value


class ChessNet(nn.Module):
    """
    Complete Chess Neural Network with dual-head architecture.

    This network takes a chess board position as input and outputs:
    1. Policy: probability distribution over moves
    2. Value: position evaluation score

    Architecture Overview:
    ┌─────────────────────────────────────┐
    │ Input: (batch, 18, 8, 8)           │
    │ 18 feature planes × 8×8 board      │
    └──────────┬──────────────────────────┘
               │
    ┌──────────▼──────────────────────────┐
    │ Convolutional Body                  │
    │ - Conv Block 1: 18 → 64 filters    │
    │ - Conv Block 2: 64 → 64 filters    │
    │ - Conv Block 3: 64 → 128 filters   │
    │ Shared feature extraction           │
    └──────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    ┌────▼─────┐ ┌──▼──────┐
    │ Policy   │ │ Value   │
    │ Head     │ │ Head    │
    │ → 4672   │ │ → 1     │
    └──────────┘ └─────────┘

    Args:
        input_planes: Number of input feature planes (default: 18)
        conv_filters: List of filter sizes for conv blocks (default: [64, 64, 128])
        policy_output: Number of policy outputs (default: 4672)
        value_hidden: Hidden layer size for value head (default: 32)
    """

    def __init__(
        self,
        input_planes: int = 18,
        conv_filters: list = None,
        policy_output: int = 4672,
        value_hidden: int = 32
    ):
        super(ChessNet, self).__init__()

        # Default filter configuration if not specified
        if conv_filters is None:
            conv_filters = [64, 64, 128]

        # Store configuration for later reference
        self.input_planes = input_planes
        self.conv_filters = conv_filters
        self.policy_output = policy_output
        self.value_hidden = value_hidden

        # Build the convolutional body (shared feature extractor)
        # This processes the input board and extracts high-level features
        conv_blocks = []

        # First conv block: input_planes → first filter size
        in_channels = input_planes
        for out_channels in conv_filters:
            conv_blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        # Convert list to sequential module
        self.conv_body = nn.Sequential(*conv_blocks)

        # Build the two heads
        # Both heads receive the same features from conv_body
        final_channels = conv_filters[-1]  # Last filter size (128)

        # Policy head: predicts which move to play
        self.policy_head = PolicyHead(final_channels, policy_output)

        # Value head: evaluates how good the position is
        self.value_head = ValueHead(final_channels, value_hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor of shape (batch, 18, 8, 8)
               18 feature planes encoding the chess position

        Returns:
            Tuple of (policy_logits, value):
            - policy_logits: shape (batch, 4672) - raw move scores
            - value: shape (batch, 1) - position evaluation in [-1, 1]

        Example:
            >>> net = ChessNet()
            >>> input_tensor = torch.randn(32, 18, 8, 8)  # Batch of 32 positions
            >>> policy, value = net(input_tensor)
            >>> policy.shape
            torch.Size([32, 4672])
            >>> value.shape
            torch.Size([32, 1])
        """
        # Pass through convolutional body
        # This extracts features from the board position
        # Shape: (batch, 18, 8, 8) → (batch, 128, 8, 8)
        features = self.conv_body(x)

        # Pass features through both heads in parallel
        # Policy head: which move to play
        policy_logits = self.policy_head(features)

        # Value head: how good is this position
        value = self.value_head(features)

        return policy_logits, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the network (inference mode).

        This is a convenience method that sets the network to evaluation mode
        and disables gradient computation for faster inference.

        Args:
            x: Input tensor of shape (batch, 18, 8, 8) or (18, 8, 8)

        Returns:
            Tuple of (policy_probs, value):
            - policy_probs: shape (batch, 4672) - move probabilities (after softmax)
            - value: shape (batch, 1) - position evaluation in [-1, 1]
        """
        # Set to evaluation mode (disables dropout, batchnorm uses running stats)
        self.eval()

        # Disable gradient computation for efficiency
        with torch.no_grad():
            # Add batch dimension if needed
            if x.dim() == 3:  # Single position (18, 8, 8)
                x = x.unsqueeze(0)  # Add batch dim: (1, 18, 8, 8)

            # Forward pass
            policy_logits, value = self.forward(x)

            # Apply softmax to policy logits to get probabilities
            # Softmax converts raw scores to probability distribution
            policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value

    def count_parameters(self) -> int:
        """
        Count total trainable parameters in the network.

        Returns:
            Total number of trainable parameters

        Example:
            >>> net = ChessNet()
            >>> net.count_parameters()
            ~500000
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """
        Get network configuration as dictionary.

        Returns:
            Dictionary with network architecture parameters
        """
        return {
            'input_planes': self.input_planes,
            'conv_filters': self.conv_filters,
            'policy_output': self.policy_output,
            'value_hidden': self.value_hidden
        }


def create_chess_net(config: dict = None) -> ChessNet:
    """
    Factory function to create a ChessNet with optional custom configuration.

    Args:
        config: Dictionary with network parameters (optional)
                If None, uses default configuration

    Returns:
        Initialized ChessNet instance

    Example:
        >>> # Default network
        >>> net = create_chess_net()
        >>>
        >>> # Custom network
        >>> config = {
        >>>     'input_planes': 18,
        >>>     'conv_filters': [128, 128, 256],
        >>>     'policy_output': 4672,
        >>>     'value_hidden': 64
        >>> }
        >>> net = create_chess_net(config)
    """
    if config is None:
        # Default configuration
        config = {
            'input_planes': 18,
            'conv_filters': [64, 64, 128],
            'policy_output': 4672,
            'value_hidden': 32
        }

    return ChessNet(**config)


if __name__ == "__main__":
    """Demo: create network and test forward pass"""
    print("Chess Neural Network Demo")
    print("=" * 60)

    # Create network with default configuration
    net = create_chess_net()

    print("\nNetwork Configuration:")
    print(f"  Input planes: {net.input_planes}")
    print(f"  Conv filters: {net.conv_filters}")
    print(f"  Policy output: {net.policy_output}")
    print(f"  Value hidden: {net.value_hidden}")

    # Count parameters
    param_count = net.count_parameters()
    print(f"\nTotal trainable parameters: {param_count:,}")
    print(f"Estimated size: {param_count * 4 / 1024 / 1024:.2f} MB (float32)")

    # Test forward pass with random input
    print("\nTesting forward pass...")
    batch_size = 8
    dummy_input = torch.randn(batch_size, 18, 8, 8)

    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    policy_logits, value = net(dummy_input)

    print(f"Policy output shape: {policy_logits.shape}")
    print(f"Value output shape: {value.shape}")

    # Test prediction (with softmax)
    print("\nTesting prediction (inference mode)...")
    policy_probs, value_pred = net.predict(dummy_input)

    print(f"Policy probabilities shape: {policy_probs.shape}")
    print(f"Policy probs sum to 1: {torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size))}")
    print(f"Value range: [{value_pred.min():.3f}, {value_pred.max():.3f}]")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
