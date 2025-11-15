#!/usr/bin/env python3
"""
ChessEngine GUI Launcher

Quick launcher for ChessEngine GUI application.

Usage:
    python run_gui.py

Or make executable and run:
    chmod +x run_gui.py
    ./run_gui.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run GUI
from GUI.Gui_scheletro import main

if __name__ == "__main__":
    print("="*60)
    print("Starting ChessEngine GUI...")
    print("="*60)
    print()
    print("Welcome to ChessEngine AI!")
    print("Loading graphical interface...")
    print()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGUI closed by user.")
    except Exception as e:
        print(f"\nError starting GUI: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Tkinter is installed:")
        print("     python -c 'import tkinter; print(\"OK\")'")
        print("  2. On Linux, install with:")
        print("     sudo apt-get install python3-tk")
        print("  3. Check that all dependencies are installed:")
        print("     pip install -r requirements.txt")
        sys.exit(1)
