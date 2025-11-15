"""
ChessEngine GUI - Statistics Module

Modulo per visualizzare e analizzare le statistiche di training
e le performance dei modelli.

Features:
- Carica e visualizza checkpoint disponibili
- Mostra metriche di training (loss, accuracy)
- Valutazione vs random player
- Test su puzzle tattici
- Confronto tra modelli
- Grafici di progresso

Usage:
    Accessibile da GUI principale
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import sys
import json
import threading

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StatisticsFrame(tk.Frame):
    """
    Frame per il modulo Statistics.

    Mostra statistiche e permette di valutare i modelli.
    """

    def __init__(self, parent, controller):
        """
        Inizializza Statistics frame.

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent, bg='white')
        self.controller = controller
        self.grid(row=0, column=0, sticky="nsew")

        # State
        self.checkpoints = []
        self.selected_checkpoint = None
        self.evaluation_running = False

        # Create UI
        self.create_ui()

        # Load checkpoints
        self.refresh_checkpoints()

    def create_ui(self):
        """Crea UI per statistics"""
        # Header
        header_frame = tk.Frame(self, bg='white')
        header_frame.pack(fill='x', padx=20, pady=20)

        header = tk.Label(
            header_frame,
            text="Model Statistics & Evaluation",
            font=('Helvetica', 18, 'bold'),
            bg='white',
            fg='#2C3E50'
        )
        header.pack(side='left')

        # Refresh button
        ttk.Button(
            header_frame,
            text="🔄 Refresh",
            command=self.refresh_checkpoints
        ).pack(side='right')

        # Main container
        main_container = tk.Frame(self, bg='white')
        main_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Left panel: Checkpoint list
        left_panel = tk.Frame(main_container, bg='white')
        left_panel.pack(side='left', fill='both', padx=(0, 10))

        self.create_checkpoint_list(left_panel)

        # Right panel: Evaluation options
        right_panel = tk.Frame(main_container, bg='white')
        right_panel.pack(side='left', fill='both', expand=True)

        self.create_evaluation_panel(right_panel)

    def create_checkpoint_list(self, parent):
        """Crea lista checkpoint"""
        # Header
        tk.Label(
            parent,
            text="Available Checkpoints",
            font=('Helvetica', 14, 'bold'),
            bg='white',
            fg='#2C3E50'
        ).pack(anchor='w', pady=(0, 10))

        # Listbox frame
        listbox_frame = tk.Frame(parent, bg='white')
        listbox_frame.pack(fill='both', expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side='right', fill='y')

        # Listbox
        self.checkpoint_listbox = tk.Listbox(
            listbox_frame,
            font=('Courier', 10),
            yscrollcommand=scrollbar.set,
            width=30,
            height=20
        )
        self.checkpoint_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.checkpoint_listbox.yview)

        # Bind selection
        self.checkpoint_listbox.bind('<<ListboxSelect>>', self.on_checkpoint_select)

        # Info label
        self.checkpoint_info_label = tk.Label(
            parent,
            text="No checkpoint selected",
            font=('Helvetica', 9),
            bg='white',
            fg='#7F8C8D',
            justify='left'
        )
        self.checkpoint_info_label.pack(anchor='w', pady=(10, 0))

    def create_evaluation_panel(self, parent):
        """Crea pannello valutazione"""
        # Tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)

        # Tab 1: Quick Stats
        stats_tab = tk.Frame(notebook, bg='white')
        notebook.add(stats_tab, text="Quick Stats")
        self.create_stats_tab(stats_tab)

        # Tab 2: Evaluation vs Random
        eval_random_tab = tk.Frame(notebook, bg='white')
        notebook.add(eval_random_tab, text="Vs Random")
        self.create_eval_random_tab(eval_random_tab)

        # Tab 3: Puzzle Testing
        puzzle_tab = tk.Frame(notebook, bg='white')
        notebook.add(puzzle_tab, text="Puzzle Test")
        self.create_puzzle_tab(puzzle_tab)

        # Tab 4: Model Comparison
        compare_tab = tk.Frame(notebook, bg='white')
        notebook.add(compare_tab, text="Compare Models")
        self.create_compare_tab(compare_tab)

    def create_stats_tab(self, parent):
        """Crea tab statistiche veloci"""
        # Info text
        self.stats_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=('Courier', 10),
            height=20
        )
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Load stats button
        ttk.Button(
            parent,
            text="Load Checkpoint Info",
            command=self.load_checkpoint_stats
        ).pack(pady=10)

    def create_eval_random_tab(self, parent):
        """Crea tab valutazione vs random"""
        # Description
        desc = tk.Label(
            parent,
            text="Test selected model against random player",
            font=('Helvetica', 10),
            bg='white',
            fg='#7F8C8D'
        )
        desc.pack(pady=10)

        # Config frame
        config_frame = tk.LabelFrame(
            parent,
            text="Configuration",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        config_frame.pack(fill='x', padx=10, pady=10)

        # Number of games
        games_frame = tk.Frame(config_frame, bg='white')
        games_frame.pack(fill='x', pady=5)

        tk.Label(
            games_frame,
            text="Number of games:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.eval_random_games = tk.StringVar(value="100")
        ttk.Entry(
            games_frame,
            textvariable=self.eval_random_games,
            width=10
        ).pack(side='left', padx=5)

        tk.Label(
            games_frame,
            text="(even number, 50% as white)",
            font=('Helvetica', 9),
            bg='white',
            fg='#95A5A6'
        ).pack(side='left')

        # MCTS simulations
        sims_frame = tk.Frame(config_frame, bg='white')
        sims_frame.pack(fill='x', pady=5)

        tk.Label(
            sims_frame,
            text="MCTS simulations:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.eval_random_sims = tk.StringVar(value="50")
        ttk.Entry(
            sims_frame,
            textvariable=self.eval_random_sims,
            width=10
        ).pack(side='left', padx=5)

        tk.Label(
            sims_frame,
            text="(higher = stronger but slower)",
            font=('Helvetica', 9),
            bg='white',
            fg='#95A5A6'
        ).pack(side='left')

        # Run button
        self.eval_random_button = ttk.Button(
            config_frame,
            text="▶ Run Evaluation",
            command=self.run_eval_random
        )
        self.eval_random_button.pack(pady=10)

        # Results
        results_frame = tk.LabelFrame(
            parent,
            text="Results",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.eval_random_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            height=10
        )
        self.eval_random_text.pack(fill='both', expand=True)

    def create_puzzle_tab(self, parent):
        """Crea tab test puzzle"""
        # Description
        desc = tk.Label(
            parent,
            text="Test tactical puzzle solving ability",
            font=('Helvetica', 10),
            bg='white',
            fg='#7F8C8D'
        )
        desc.pack(pady=10)

        # Config frame
        config_frame = tk.LabelFrame(
            parent,
            text="Configuration",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        config_frame.pack(fill='x', padx=10, pady=10)

        # Puzzle set
        set_frame = tk.Frame(config_frame, bg='white')
        set_frame.pack(fill='x', pady=5)

        tk.Label(
            set_frame,
            text="Puzzle set:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.puzzle_set = tk.StringVar(value="mate_in_2")
        ttk.Combobox(
            set_frame,
            textvariable=self.puzzle_set,
            values=['mate_in_2', 'all'],
            state='readonly',
            width=15
        ).pack(side='left', padx=5)

        # MCTS simulations
        sims_frame = tk.Frame(config_frame, bg='white')
        sims_frame.pack(fill='x', pady=5)

        tk.Label(
            sims_frame,
            text="MCTS simulations:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.puzzle_sims = tk.StringVar(value="100")
        ttk.Entry(
            sims_frame,
            textvariable=self.puzzle_sims,
            width=10
        ).pack(side='left', padx=5)

        # Run button
        self.puzzle_button = ttk.Button(
            config_frame,
            text="▶ Run Puzzle Test",
            command=self.run_puzzle_test
        )
        self.puzzle_button.pack(pady=10)

        # Results
        results_frame = tk.LabelFrame(
            parent,
            text="Results",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.puzzle_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            height=10
        )
        self.puzzle_text.pack(fill='both', expand=True)

    def create_compare_tab(self, parent):
        """Crea tab confronto modelli"""
        # Description
        desc = tk.Label(
            parent,
            text="Compare two models head-to-head",
            font=('Helvetica', 10),
            bg='white',
            fg='#7F8C8D'
        )
        desc.pack(pady=10)

        # Config frame
        config_frame = tk.LabelFrame(
            parent,
            text="Configuration",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        config_frame.pack(fill='x', padx=10, pady=10)

        # Model 2 selection
        model2_frame = tk.Frame(config_frame, bg='white')
        model2_frame.pack(fill='x', pady=5)

        tk.Label(
            model2_frame,
            text="Compare with:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.compare_model2 = tk.StringVar(value="")
        self.compare_model2_combo = ttk.Combobox(
            model2_frame,
            textvariable=self.compare_model2,
            state='readonly',
            width=30
        )
        self.compare_model2_combo.pack(side='left', padx=5)

        # Games
        games_frame = tk.Frame(config_frame, bg='white')
        games_frame.pack(fill='x', pady=5)

        tk.Label(
            games_frame,
            text="Number of games:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.compare_games = tk.StringVar(value="50")
        ttk.Entry(
            games_frame,
            textvariable=self.compare_games,
            width=10
        ).pack(side='left', padx=5)

        # MCTS simulations
        sims_frame = tk.Frame(config_frame, bg='white')
        sims_frame.pack(fill='x', pady=5)

        tk.Label(
            sims_frame,
            text="MCTS simulations:",
            font=('Helvetica', 10),
            bg='white',
            width=20,
            anchor='w'
        ).pack(side='left')

        self.compare_sims = tk.StringVar(value="50")
        ttk.Entry(
            sims_frame,
            textvariable=self.compare_sims,
            width=10
        ).pack(side='left', padx=5)

        # Run button
        self.compare_button = ttk.Button(
            config_frame,
            text="▶ Run Comparison",
            command=self.run_comparison
        )
        self.compare_button.pack(pady=10)

        # Results
        results_frame = tk.LabelFrame(
            parent,
            text="Results",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.compare_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            height=10
        )
        self.compare_text.pack(fill='both', expand=True)

    def refresh_checkpoints(self):
        """Aggiorna lista checkpoint"""
        checkpoint_dir = Path("checkpoints")

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
            self.checkpoints = []
            self.checkpoint_listbox.delete(0, tk.END)
            self.checkpoint_listbox.insert(0, "No checkpoints found")
            return

        # Find checkpoint files
        checkpoint_files = sorted(
            checkpoint_dir.glob("model_iter_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        self.checkpoints = checkpoint_files

        # Update listbox
        self.checkpoint_listbox.delete(0, tk.END)

        if not checkpoint_files:
            self.checkpoint_listbox.insert(0, "No checkpoints found")
            self.checkpoint_listbox.insert(1, "")
            self.checkpoint_listbox.insert(2, "Train a model first!")
        else:
            for ckpt in checkpoint_files:
                # Extract iteration number
                iter_num = ckpt.stem.split('_')[-1]
                # Get file size
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                self.checkpoint_listbox.insert(
                    tk.END,
                    f"Iteration {iter_num:>3s} ({size_mb:.1f} MB)"
                )

        # Update compare combo
        if hasattr(self, 'compare_model2_combo'):
            checkpoint_names = [
                f"Iteration {ckpt.stem.split('_')[-1]}"
                for ckpt in checkpoint_files
            ]
            self.compare_model2_combo['values'] = checkpoint_names

    def on_checkpoint_select(self, event):
        """Gestisci selezione checkpoint"""
        selection = self.checkpoint_listbox.curselection()

        if not selection or not self.checkpoints:
            return

        idx = selection[0]
        if idx >= len(self.checkpoints):
            return

        self.selected_checkpoint = self.checkpoints[idx]

        # Update info label
        iter_num = self.selected_checkpoint.stem.split('_')[-1]
        self.checkpoint_info_label.config(
            text=f"Selected: Iteration {iter_num}\n"
                 f"Path: {self.selected_checkpoint}"
        )

    def load_checkpoint_stats(self):
        """Carica statistiche da checkpoint"""
        if not self.selected_checkpoint:
            messagebox.showwarning("Warning", "Please select a checkpoint first")
            return

        try:
            import torch

            checkpoint = torch.load(self.selected_checkpoint, map_location='cpu')

            # Format stats
            stats = "="*60 + "\n"
            stats += "CHECKPOINT INFORMATION\n"
            stats += "="*60 + "\n\n"

            stats += f"File: {self.selected_checkpoint.name}\n"
            stats += f"Iteration: {checkpoint.get('iteration', 'N/A')}\n"
            stats += f"Training steps: {checkpoint.get('train_steps', 'N/A')}\n\n"

            # Config
            if 'config' in checkpoint:
                stats += "Configuration:\n"
                stats += "-"*60 + "\n"
                config = checkpoint['config']
                for key, value in config.items():
                    stats += f"  {key}: {value}\n"
                stats += "\n"

            # Metrics
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                stats += "Training Metrics:\n"
                stats += "-"*60 + "\n"

                if 'total_loss_history' in metrics and metrics['total_loss_history']:
                    recent_losses = metrics['total_loss_history'][-10:]
                    avg_recent = sum(recent_losses) / len(recent_losses)
                    stats += f"  Recent avg loss (last 10): {avg_recent:.4f}\n"
                    stats += f"  Best loss: {min(metrics['total_loss_history']):.4f}\n"

                if 'policy_accuracy_history' in metrics and metrics['policy_accuracy_history']:
                    recent_acc = metrics['policy_accuracy_history'][-10:]
                    avg_acc = sum(recent_acc) / len(recent_acc)
                    stats += f"  Recent policy accuracy: {avg_acc:.2%}\n"

            # Iteration stats
            if 'iteration_stats' in checkpoint:
                iter_stats = checkpoint['iteration_stats']
                stats += "\nLast Iteration Stats:\n"
                stats += "-"*60 + "\n"
                for key, value in iter_stats.items():
                    if isinstance(value, float):
                        stats += f"  {key}: {value:.4f}\n"
                    else:
                        stats += f"  {key}: {value}\n"

            # Display
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', stats)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint:\n{e}")

    def run_eval_random(self):
        """Run evaluation vs random"""
        if not self.selected_checkpoint:
            messagebox.showwarning("Warning", "Please select a checkpoint first")
            return

        if self.evaluation_running:
            messagebox.showwarning("Warning", "Evaluation already in progress")
            return

        try:
            num_games = int(self.eval_random_games.get())
            mcts_sims = int(self.eval_random_sims.get())

            if num_games % 2 != 0:
                messagebox.showwarning("Warning", "Number of games must be even")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return

        # Disable button
        self.eval_random_button.config(state='disabled')
        self.evaluation_running = True

        # Clear results
        self.eval_random_text.delete('1.0', tk.END)
        self.eval_random_text.insert('1.0', "Running evaluation...\n\n")

        # Run in thread
        thread = threading.Thread(
            target=self._run_eval_random_thread,
            args=(num_games, mcts_sims),
            daemon=True
        )
        thread.start()

    def _run_eval_random_thread(self, num_games, mcts_sims):
        """Thread per eval vs random"""
        try:
            import torch
            from src.game.encoder import BoardEncoder
            from src.game.decoder import MoveDecoder
            from src.models.chess_net import ChessNet
            from src.evaluation.evaluator import ModelEvaluator, estimate_elo

            # Load model
            encoder = BoardEncoder()
            decoder = MoveDecoder()
            model = ChessNet()

            checkpoint = torch.load(self.selected_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Create evaluator
            evaluator = ModelEvaluator(encoder, decoder)

            # Log
            self._append_eval_random(f"Model loaded: {self.selected_checkpoint.name}\n")
            self._append_eval_random(f"Games: {num_games}, MCTS sims: {mcts_sims}\n\n")
            self._append_eval_random("Playing games...\n")

            # Run evaluation
            results = evaluator.evaluate_vs_random(
                model=model,
                num_games=num_games,
                mcts_simulations=mcts_sims,
                verbose=False
            )

            # Format results
            output = "="*60 + "\n"
            output += "EVALUATION RESULTS\n"
            output += "="*60 + "\n\n"

            output += f"Total games: {results['total_games']}\n"
            output += f"Wins: {results['wins']}\n"
            output += f"Draws: {results['draws']}\n"
            output += f"Losses: {results['losses']}\n"
            output += f"Win rate: {results['win_rate']:.1%}\n\n"

            # ELO estimate
            if results['win_rate'] > 0:
                elo = estimate_elo(results['win_rate'], opponent_elo=800)
                output += f"Estimated ELO: ~{elo}\n\n"

            output += f"Average moves: {results['avg_moves']:.1f}\n"
            output += f"Average time per game: {results['avg_time']:.1f}s\n"

            self._append_eval_random(output)

        except Exception as e:
            self._append_eval_random(f"\nERROR: {e}\n")
            import traceback
            self._append_eval_random(traceback.format_exc())

        finally:
            self.evaluation_running = False
            self.controller.after(0, lambda: self.eval_random_button.config(state='normal'))

    def _append_eval_random(self, text):
        """Append text to eval random output"""
        self.controller.after(0, lambda: self.eval_random_text.insert(tk.END, text))
        self.controller.after(0, lambda: self.eval_random_text.see(tk.END))

    def run_puzzle_test(self):
        """Run puzzle test"""
        if not self.selected_checkpoint:
            messagebox.showwarning("Warning", "Please select a checkpoint first")
            return

        if self.evaluation_running:
            messagebox.showwarning("Warning", "Evaluation already in progress")
            return

        try:
            puzzle_set = self.puzzle_set.get()
            mcts_sims = int(self.puzzle_sims.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return

        self.puzzle_button.config(state='disabled')
        self.evaluation_running = True

        self.puzzle_text.delete('1.0', tk.END)
        self.puzzle_text.insert('1.0', "Running puzzle test...\n\n")

        thread = threading.Thread(
            target=self._run_puzzle_test_thread,
            args=(puzzle_set, mcts_sims),
            daemon=True
        )
        thread.start()

    def _run_puzzle_test_thread(self, puzzle_set, mcts_sims):
        """Thread per puzzle test"""
        try:
            import torch
            from src.game.encoder import BoardEncoder
            from src.game.decoder import MoveDecoder
            from src.models.chess_net import ChessNet
            from src.evaluation.puzzles import PuzzleTester, get_builtin_puzzle_set

            # Load model
            encoder = BoardEncoder()
            decoder = MoveDecoder()
            model = ChessNet()

            checkpoint = torch.load(self.selected_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Load puzzles
            puzzles = get_builtin_puzzle_set(puzzle_set)

            self._append_puzzle(f"Model: {self.selected_checkpoint.name}\n")
            self._append_puzzle(f"Puzzle set: {puzzle_set} ({len(puzzles)} puzzles)\n")
            self._append_puzzle(f"MCTS sims: {mcts_sims}\n\n")

            # Create tester
            tester = PuzzleTester(encoder, decoder)

            # Run tests
            results = tester.test_puzzles(
                puzzles=puzzles,
                model=model,
                mcts_simulations=mcts_sims,
                verbose=False
            )

            # Format results
            output = "="*60 + "\n"
            output += "PUZZLE TEST RESULTS\n"
            output += "="*60 + "\n\n"

            output += f"Total puzzles: {results['total']}\n"
            output += f"Solved: {results['solved']}\n"
            output += f"Failed: {results['failed']}\n"
            output += f"Accuracy: {results['accuracy']:.1%}\n\n"

            # By category
            if results['by_category']:
                output += "By category:\n"
                for cat, stats in results['by_category'].items():
                    output += f"  {cat}: {stats['solved']}/{stats['total']} ({stats['accuracy']:.1%})\n"
                output += "\n"

            # By difficulty
            if results['by_difficulty']:
                output += "By difficulty:\n"
                for diff, stats in results['by_difficulty'].items():
                    output += f"  {diff}: {stats['solved']}/{stats['total']} ({stats['accuracy']:.1%})\n"

            self._append_puzzle(output)

        except Exception as e:
            self._append_puzzle(f"\nERROR: {e}\n")
            import traceback
            self._append_puzzle(traceback.format_exc())

        finally:
            self.evaluation_running = False
            self.controller.after(0, lambda: self.puzzle_button.config(state='normal'))

    def _append_puzzle(self, text):
        """Append text to puzzle output"""
        self.controller.after(0, lambda: self.puzzle_text.insert(tk.END, text))
        self.controller.after(0, lambda: self.puzzle_text.see(tk.END))

    def run_comparison(self):
        """Run model comparison"""
        if not self.selected_checkpoint:
            messagebox.showwarning("Warning", "Please select first model")
            return

        model2_name = self.compare_model2.get()
        if not model2_name:
            messagebox.showwarning("Warning", "Please select second model")
            return

        if self.evaluation_running:
            messagebox.showwarning("Warning", "Evaluation already in progress")
            return

        # Find model 2 checkpoint
        iter_num = model2_name.split()[-1]
        model2_path = Path(f"checkpoints/model_iter_{iter_num}.pt")

        if not model2_path.exists():
            messagebox.showerror("Error", "Second model not found")
            return

        try:
            num_games = int(self.compare_games.get())
            mcts_sims = int(self.compare_sims.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return

        self.compare_button.config(state='disabled')
        self.evaluation_running = True

        self.compare_text.delete('1.0', tk.END)
        self.compare_text.insert('1.0', "Running comparison...\n\n")

        thread = threading.Thread(
            target=self._run_comparison_thread,
            args=(model2_path, num_games, mcts_sims),
            daemon=True
        )
        thread.start()

    def _run_comparison_thread(self, model2_path, num_games, mcts_sims):
        """Thread per confronto"""
        try:
            import torch
            from src.game.encoder import BoardEncoder
            from src.game.decoder import MoveDecoder
            from src.models.chess_net import ChessNet
            from src.evaluation.evaluator import ModelEvaluator

            encoder = BoardEncoder()
            decoder = MoveDecoder()

            # Load models
            model1 = ChessNet()
            checkpoint1 = torch.load(self.selected_checkpoint, map_location='cpu')
            model1.load_state_dict(checkpoint1['model_state_dict'])
            model1.eval()

            model2 = ChessNet()
            checkpoint2 = torch.load(model2_path, map_location='cpu')
            model2.load_state_dict(checkpoint2['model_state_dict'])
            model2.eval()

            model1_name = f"Iter {checkpoint1.get('iteration', '?')}"
            model2_name = f"Iter {checkpoint2.get('iteration', '?')}"

            self._append_compare(f"Model 1: {model1_name}\n")
            self._append_compare(f"Model 2: {model2_name}\n")
            self._append_compare(f"Games: {num_games}, MCTS sims: {mcts_sims}\n\n")

            # Create evaluator
            evaluator = ModelEvaluator(encoder, decoder)

            # Run comparison
            results = evaluator.compare_models(
                model1=model1,
                model2=model2,
                num_games=num_games,
                mcts_simulations=mcts_sims,
                model1_name=model1_name,
                model2_name=model2_name,
                verbose=False
            )

            # Format results
            output = "="*60 + "\n"
            output += "COMPARISON RESULTS\n"
            output += "="*60 + "\n\n"

            output += f"{model1_name}: {results['model1_wins']}W - {results['draws']}D - {results['model2_wins']}L\n"
            output += f"{model2_name}: {results['model2_wins']}W - {results['draws']}D - {results['model1_wins']}L\n\n"

            output += f"{model1_name} win rate: {results['model1_win_rate']:.1%}\n"
            output += f"{model2_name} win rate: {results['model2_win_rate']:.1%}\n\n"

            if results['model1_wins'] > results['model2_wins']:
                output += f"Winner: {model1_name}\n"
            elif results['model2_wins'] > results['model1_wins']:
                output += f"Winner: {model2_name}\n"
            else:
                output += "Result: Tie\n"

            self._append_compare(output)

        except Exception as e:
            self._append_compare(f"\nERROR: {e}\n")
            import traceback
            self._append_compare(traceback.format_exc())

        finally:
            self.evaluation_running = False
            self.controller.after(0, lambda: self.compare_button.config(state='normal'))

    def _append_compare(self, text):
        """Append text to compare output"""
        self.controller.after(0, lambda: self.compare_text.insert(tk.END, text))
        self.controller.after(0, lambda: self.compare_text.see(tk.END))

    def on_show(self):
        """Called when frame is shown"""
        self.refresh_checkpoints()
