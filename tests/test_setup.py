"""
Test Suite for ChessEngine Setup Verification
Tests environment, dependencies, and project structure
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_python_version():
    """Verify Python version is 3.8+"""
    assert sys.version_info >= (3, 8), f"Python 3.8+ required, found {sys.version_info}"
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def test_import_torch():
    """Test PyTorch import and version"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        assert torch.__version__ >= "2.0.0", f"PyTorch 2.0+ required, found {torch.__version__}"
    except ImportError as e:
        raise AssertionError(f"Failed to import PyTorch: {e}")


def test_cuda_availability():
    """Check CUDA availability (optional, not required)"""
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - training will use CPU")


def test_import_chess():
    """Test python-chess library import"""
    try:
        import chess
        print(f"✓ python-chess imported successfully")

        # Test basic functionality
        board = chess.Board()
        assert board.is_valid(), "Failed to create valid chess board"
        assert len(list(board.legal_moves)) == 20, "Initial position should have 20 legal moves"
        print(f"  Initial legal moves: {len(list(board.legal_moves))}")
    except ImportError as e:
        raise AssertionError(f"Failed to import python-chess: {e}")


def test_import_numpy():
    """Test NumPy import"""
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        raise AssertionError(f"Failed to import NumPy: {e}")


def test_import_yaml():
    """Test PyYAML import"""
    try:
        import yaml
        print(f"✓ PyYAML imported successfully")
    except ImportError as e:
        raise AssertionError(f"Failed to import PyYAML: {e}")


def test_config_file_exists():
    """Verify config.yaml exists and is valid"""
    import yaml

    config_path = project_root / "config.yaml"
    assert config_path.exists(), f"config.yaml not found at {config_path}"
    print(f"✓ config.yaml found at {config_path}")

    # Try loading config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Verify required sections
    required_sections = ['model', 'mcts', 'training', 'evaluation', 'system', 'paths']
    for section in required_sections:
        assert section in config, f"Missing section '{section}' in config.yaml"

    print(f"✓ config.yaml is valid with all required sections")


def test_project_structure():
    """Verify all required directories exist"""
    required_dirs = [
        'src',
        'src/models',
        'src/mcts',
        'src/game',
        'src/training',
        'src/evaluation',
        'src/utils',
        'data',
        'data/games',
        'data/training_data',
        'data/checkpoints',
        'data/logs',
        'notebooks',
        'scripts',
        'tests'
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists() and full_path.is_dir(), f"Directory {dir_path} not found"

    print(f"✓ All {len(required_dirs)} required directories exist")


def test_init_files():
    """Verify __init__.py files exist in src modules"""
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/mcts/__init__.py',
        'src/game/__init__.py',
        'src/training/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py'
    ]

    for init_file in init_files:
        full_path = project_root / init_file
        assert full_path.exists(), f"Missing {init_file}"

    print(f"✓ All {len(init_files)} __init__.py files exist")


def test_project_import():
    """Test importing the main project package"""
    try:
        import src
        print(f"✓ Project package 'src' imported successfully")
        print(f"  Version: {src.__version__}")
    except ImportError as e:
        raise AssertionError(f"Failed to import project package: {e}")


def test_torch_tensor_creation():
    """Test basic PyTorch operations"""
    import torch

    # Create a simple tensor
    x = torch.randn(18, 8, 8)
    assert x.shape == (18, 8, 8), f"Expected shape (18, 8, 8), got {x.shape}"

    # Test device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    print(f"✓ PyTorch tensor operations working")
    print(f"  Test tensor shape: {x.shape}")
    print(f"  Device: {device}")


def test_chess_move_generation():
    """Test chess move generation and board manipulation"""
    import chess

    board = chess.Board()

    # Test move
    move = chess.Move.from_uci("e2e4")
    board.push(move)
    assert board.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    # Test legal moves
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) == 20, f"Expected 20 legal moves, got {len(legal_moves)}"

    print(f"✓ Chess board operations working")
    print(f"  Legal moves after e2e4: {len(legal_moves)}")


if __name__ == "__main__":
    """Run all tests manually"""
    print("=" * 60)
    print("ChessEngine Setup Verification")
    print("=" * 60)
    print()

    tests = [
        ("Python Version", test_python_version),
        ("PyTorch Import", test_import_torch),
        ("CUDA Availability", test_cuda_availability),
        ("python-chess Import", test_import_chess),
        ("NumPy Import", test_import_numpy),
        ("PyYAML Import", test_import_yaml),
        ("Config File", test_config_file_exists),
        ("Project Structure", test_project_structure),
        ("Init Files", test_init_files),
        ("Project Import", test_project_import),
        ("PyTorch Operations", test_torch_tensor_creation),
        ("Chess Operations", test_chess_move_generation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
