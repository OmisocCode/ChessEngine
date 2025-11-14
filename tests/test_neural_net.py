"""
Test Suite for Chess Neural Network

Tests the ChessNet architecture, forward pass, and model utilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import tempfile

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Skipping neural network tests.")

if TORCH_AVAILABLE:
    from src.models.chess_net import ChessNet, ConvBlock, PolicyHead, ValueHead, create_chess_net
    from src.models.model_utils import (
        save_model, load_model, save_checkpoint, load_checkpoint,
        get_model_size, list_checkpoints
    )


class TestConvBlock:
    """Test convolutional block"""

    def test_conv_block_creation(self):
        """Test creating a conv block"""
        if not TORCH_AVAILABLE:
            return

        block = ConvBlock(in_channels=18, out_channels=64)

        assert block.conv.in_channels == 18
        assert block.conv.out_channels == 64
        assert block.conv.kernel_size == (3, 3)

    def test_conv_block_forward(self):
        """Test forward pass through conv block"""
        if not TORCH_AVAILABLE:
            return

        block = ConvBlock(in_channels=18, out_channels=64)

        # Test input
        x = torch.randn(4, 18, 8, 8)
        output = block(x)

        # Check output shape
        assert output.shape == (4, 64, 8, 8)

    def test_conv_block_preserves_spatial_dims(self):
        """Test that spatial dimensions are preserved"""
        if not TORCH_AVAILABLE:
            return

        block = ConvBlock(in_channels=32, out_channels=64)

        x = torch.randn(1, 32, 8, 8)
        output = block(x)

        # Spatial dimensions should remain 8x8
        assert output.shape[2:] == (8, 8)


class TestPolicyHead:
    """Test policy head"""

    def test_policy_head_creation(self):
        """Test creating policy head"""
        if not TORCH_AVAILABLE:
            return

        head = PolicyHead(in_channels=128, num_moves=4672)

        assert head.conv.in_channels == 128
        assert head.conv.out_channels == 73

    def test_policy_head_forward(self):
        """Test forward pass through policy head"""
        if not TORCH_AVAILABLE:
            return

        head = PolicyHead(in_channels=128)

        x = torch.randn(4, 128, 8, 8)
        output = head(x)

        # Should output policy logits
        assert output.shape == (4, 4672)

    def test_policy_head_output_range(self):
        """Test that policy head outputs are unbounded (logits)"""
        if not TORCH_AVAILABLE:
            return

        head = PolicyHead(in_channels=128)

        x = torch.randn(8, 128, 8, 8)
        output = head(x)

        # Logits can be any value (not bounded)
        assert output.shape == (8, 4672)
        assert torch.isfinite(output).all()


class TestValueHead:
    """Test value head"""

    def test_value_head_creation(self):
        """Test creating value head"""
        if not TORCH_AVAILABLE:
            return

        head = ValueHead(in_channels=128, hidden_size=32)

        assert head.conv.in_channels == 128
        assert head.fc1.in_features == 64  # 1 * 8 * 8
        assert head.fc1.out_features == 32

    def test_value_head_forward(self):
        """Test forward pass through value head"""
        if not TORCH_AVAILABLE:
            return

        head = ValueHead(in_channels=128)

        x = torch.randn(4, 128, 8, 8)
        output = head(x)

        # Should output single value per position
        assert output.shape == (4, 1)

    def test_value_head_output_range(self):
        """Test that value head outputs are in [-1, 1]"""
        if not TORCH_AVAILABLE:
            return

        head = ValueHead(in_channels=128)

        x = torch.randn(16, 128, 8, 8)
        output = head(x)

        # Tanh bounds output to [-1, 1]
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)


class TestChessNet:
    """Test complete ChessNet model"""

    def test_chessnet_creation(self):
        """Test creating ChessNet"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        assert net.input_planes == 18
        assert net.conv_filters == [64, 64, 128]
        assert net.policy_output == 4672

    def test_chessnet_forward(self):
        """Test forward pass through complete network"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        # Batch of 8 positions
        x = torch.randn(8, 18, 8, 8)
        policy_logits, value = net(x)

        # Check output shapes
        assert policy_logits.shape == (8, 4672)
        assert value.shape == (8, 1)

    def test_chessnet_predict(self):
        """Test prediction with softmax"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        x = torch.randn(4, 18, 8, 8)
        policy_probs, value = net.predict(x)

        # Check shapes
        assert policy_probs.shape == (4, 4672)
        assert value.shape == (4, 1)

        # Check probabilities sum to 1
        prob_sums = policy_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4))

    def test_chessnet_single_position(self):
        """Test prediction with single position (no batch dim)"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        # Single position without batch dimension
        x = torch.randn(18, 8, 8)
        policy_probs, value = net.predict(x)

        # Should add batch dim automatically
        assert policy_probs.shape == (1, 4672)
        assert value.shape == (1, 1)

    def test_chessnet_count_parameters(self):
        """Test parameter counting"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        param_count = net.count_parameters()

        # Should have approximately 500K parameters
        assert 400_000 < param_count < 600_000
        print(f"  Total parameters: {param_count:,}")

    def test_chessnet_get_config(self):
        """Test getting configuration"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet(
            input_planes=18,
            conv_filters=[32, 64],
            policy_output=4672,
            value_hidden=16
        )

        config = net.get_config()

        assert config['input_planes'] == 18
        assert config['conv_filters'] == [32, 64]
        assert config['policy_output'] == 4672
        assert config['value_hidden'] == 16

    def test_chessnet_custom_architecture(self):
        """Test creating network with custom architecture"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet(
            input_planes=18,
            conv_filters=[128, 128, 256],
            value_hidden=64
        )

        x = torch.randn(2, 18, 8, 8)
        policy, value = net(x)

        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)


class TestCreateChessNet:
    """Test factory function"""

    def test_create_default_net(self):
        """Test creating network with defaults"""
        if not TORCH_AVAILABLE:
            return

        net = create_chess_net()

        assert isinstance(net, ChessNet)
        assert net.input_planes == 18
        assert net.conv_filters == [64, 64, 128]

    def test_create_custom_net(self):
        """Test creating network with custom config"""
        if not TORCH_AVAILABLE:
            return

        config = {
            'input_planes': 18,
            'conv_filters': [32, 32, 64],
            'policy_output': 4672,
            'value_hidden': 16
        }

        net = create_chess_net(config)

        assert net.conv_filters == [32, 32, 64]
        assert net.value_hidden == 16


class TestModelSaveLoad:
    """Test model save/load functionality"""

    def test_save_model(self):
        """Test saving model to disk"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pth')

            save_model(net, filepath)

            assert os.path.exists(filepath)

    def test_load_model(self):
        """Test loading model from disk"""
        if not TORCH_AVAILABLE:
            return

        # Save a model
        net1 = ChessNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pth')

            save_model(net1, filepath)

            # Load into new model
            net2 = ChessNet()
            checkpoint = load_model(filepath, model=net2)

            assert 'model_config' in checkpoint
            assert checkpoint['model_config']['input_planes'] == 18

    def test_save_load_weights_match(self):
        """Test that saved and loaded weights match"""
        if not TORCH_AVAILABLE:
            return

        net1 = ChessNet()

        # Make predictions with original model
        x = torch.randn(4, 18, 8, 8)
        policy1, value1 = net1.predict(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pth')

            # Save
            save_model(net1, filepath)

            # Load into new model
            net2 = ChessNet()
            load_model(filepath, model=net2)

            # Make predictions with loaded model
            policy2, value2 = net2.predict(x)

            # Predictions should match
            assert torch.allclose(policy1, policy2)
            assert torch.allclose(value1, value2)

    def test_save_with_metadata(self):
        """Test saving with metadata"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pth')

            metadata = {'loss': 0.5, 'accuracy': 0.8, 'epoch': 10}
            save_model(net, filepath, metadata=metadata)

            # Load and check metadata
            checkpoint = load_model(filepath)

            assert 'metadata' in checkpoint
            assert checkpoint['metadata']['loss'] == 0.5
            assert checkpoint['metadata']['epoch'] == 10

    def test_save_with_optimizer(self):
        """Test saving with optimizer state"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        optimizer = torch.optim.Adam(net.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pth')

            save_model(net, filepath, optimizer=optimizer)

            checkpoint = torch.load(filepath)

            assert 'optimizer_state_dict' in checkpoint
            assert 'optimizer_type' in checkpoint


class TestCheckpoints:
    """Test checkpoint functionality"""

    def test_save_checkpoint(self):
        """Test saving training checkpoint"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        optimizer = torch.optim.Adam(net.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_checkpoint(
                net,
                optimizer,
                epoch=5,
                loss=0.42,
                checkpoint_dir=tmpdir
            )

            assert os.path.exists(filepath)
            assert 'checkpoint_epoch005.pth' in filepath

    def test_load_checkpoint(self):
        """Test loading training checkpoint"""
        if not TORCH_AVAILABLE:
            return

        net1 = ChessNet()
        optimizer1 = torch.optim.Adam(net1.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            filepath = save_checkpoint(
                net1,
                optimizer1,
                epoch=10,
                loss=0.35,
                checkpoint_dir=tmpdir
            )

            # Load checkpoint
            net2 = ChessNet()
            optimizer2 = torch.optim.Adam(net2.parameters())

            checkpoint = load_checkpoint(filepath, net2, optimizer2)

            assert checkpoint['epoch'] == 10
            assert checkpoint['metadata']['loss'] == 0.35

    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        optimizer = torch.optim.Adam(net.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple checkpoints
            for epoch in [1, 2, 3]:
                save_checkpoint(net, optimizer, epoch, 0.5, tmpdir)

            # List checkpoints
            checkpoints = list_checkpoints(tmpdir)

            assert len(checkpoints) == 3
            assert any('epoch001' in c for c in checkpoints)
            assert any('epoch002' in c for c in checkpoints)
            assert any('epoch003' in c for c in checkpoints)


class TestModelSize:
    """Test model size utilities"""

    def test_get_model_size(self):
        """Test getting model size info"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        size_info = get_model_size(net)

        assert 'num_parameters' in size_info
        assert 'size_mb' in size_info
        assert 'size_bytes' in size_info

        # Check reasonable values
        assert 400_000 < size_info['num_parameters'] < 600_000
        assert 1 < size_info['size_mb'] < 5

    def test_model_size_consistency(self):
        """Test that size calculation is consistent"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        size_info = get_model_size(net)
        param_count = net.count_parameters()

        assert size_info['num_parameters'] == param_count


class TestGradientFlow:
    """Test that gradients flow properly"""

    def test_backward_pass(self):
        """Test backward pass computes gradients"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()

        x = torch.randn(4, 18, 8, 8, requires_grad=True)
        policy, value = net(x)

        # Compute dummy loss
        loss = policy.sum() + value.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in net.parameters())
        assert has_gradients

    def test_optimizer_step(self):
        """Test optimizer can update weights"""
        if not TORCH_AVAILABLE:
            return

        net = ChessNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        # Get initial weights
        initial_weights = [p.clone() for p in net.parameters()]

        # Forward and backward
        x = torch.randn(4, 18, 8, 8)
        policy, value = net(x)
        loss = policy.sum() + value.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that weights changed
        final_weights = list(net.parameters())

        weights_changed = any(
            not torch.equal(init, final)
            for init, final in zip(initial_weights, final_weights)
        )

        assert weights_changed


if __name__ == "__main__":
    """Run tests with pytest or manually"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install: pip install torch")
        sys.exit(1)

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
