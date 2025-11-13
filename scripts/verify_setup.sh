#!/bin/bash
# Quick setup verification script

echo "========================================"
echo "ChessEngine - Quick Setup Check"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python..."
python --version || { echo "✗ Python not found"; exit 1; }
echo ""

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment detected"
    echo "  Recommended: python -m venv venv && source venv/bin/activate"
fi
echo ""

# Quick import test
echo "Testing core imports..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "✗ PyTorch not installed"
python -c "import chess; print('✓ python-chess')" 2>/dev/null || echo "✗ python-chess not installed"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "✗ NumPy not installed"
python -c "import yaml; print('✓ PyYAML')" 2>/dev/null || echo "✓ PyYAML"
echo ""

# Check project structure
echo "Checking project structure..."
if [ -f "config.yaml" ] && [ -f "requirements.txt" ] && [ -d "src" ] && [ -d "tests" ]; then
    echo "✓ Project structure correct"
else
    echo "✗ Missing core files/directories"
fi
echo ""

# Suggest next steps
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
if ! python -c "import torch" 2>/dev/null; then
    echo "1. Install dependencies:"
    echo "   pip install -r requirements.txt"
    echo ""
fi
echo "2. Run full test suite:"
echo "   python tests/test_setup.py"
echo ""
echo "3. Start STEP 2 - Board Encoder"
echo ""
