"""
ChessEngine GUI - Training Module

Interfaccia per configurare e avviare il training della rete neurale.

Features:
- Configurazione completa parametri training
- Help tooltips per ogni parametro
- Progress bar e log output
- Start/Stop training
- Salvataggio/caricamento configurazioni

Usage:
    Accessibile da GUI principale
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import time
from pathlib import Path
import json
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TrainingFrame(tk.Frame):
    """
    Frame per il modulo Training.

    Permette di configurare tutti i parametri di training e
    avviare/monitorare il processo di training.
    """

    def __init__(self, parent, controller):
        """
        Inizializza Training frame.

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent, bg='white')
        self.controller = controller
        self.grid(row=0, column=0, sticky="nsew")

        # Training state
        self.training_thread = None
        self.training_active = False
        self.log_queue = queue.Queue()

        # Parameter definitions con help text
        self.param_definitions = self.get_parameter_definitions()

        # Parameter variables
        self.params = {}

        # Create UI
        self.create_ui()

        # Start log monitor
        self.monitor_log()

    def get_parameter_definitions(self):
        """
        Definisce tutti i parametri di training con help text.

        Returns:
            Dict con definizioni parametri
        """
        return {
            # Training iterations
            'num_iterations': {
                'label': 'Number of Iterations',
                'default': 20,
                'type': 'int',
                'min': 1,
                'max': 1000,
                'help': '''Number of training iterations to run.

Each iteration consists of:
1. Generate self-play games
2. Train network on collected data
3. Save checkpoint

Typical values: 10-50
More iterations = better model (but longer training time)

Suggested:
• Quick test: 2-5
• Standard: 20-30
• Advanced: 50-100'''
            },

            'games_per_iteration': {
                'label': 'Games per Iteration',
                'default': 50,
                'type': 'int',
                'min': 1,
                'max': 500,
                'help': '''Number of self-play games per iteration.

More games = more training data per iteration
This improves learning but increases time.

Typical values: 25-100

Suggested:
• Quick: 10-25
• Standard: 50
• Strong: 100-200'''
            },

            'training_batches': {
                'label': 'Training Batches per Iteration',
                'default': 100,
                'type': 'int',
                'min': 10,
                'max': 1000,
                'help': '''Number of training batches per iteration.

After self-play, the network trains on this many
batches sampled from the replay buffer.

More batches = more training per iteration

Typical values: 50-200

Suggested:
• Quick: 50
• Standard: 100
• Thorough: 200+'''
            },

            'batch_size': {
                'label': 'Batch Size',
                'default': 64,
                'type': 'int',
                'min': 8,
                'max': 512,
                'help': '''Number of examples per training batch.

Larger batch size:
• More stable gradients
• Better GPU utilization
• Requires more memory

Typical values: 32-128

Suggested:
• CPU: 32-64
• GPU: 64-128'''
            },

            # MCTS config
            'mcts_simulations': {
                'label': 'MCTS Simulations',
                'default': 50,
                'type': 'int',
                'min': 10,
                'max': 1000,
                'help': '''MCTS simulations per move during self-play.

More simulations:
• Stronger play
• Better training data
• MUCH slower

Typical values: 25-100

Suggested:
• Quick: 20-30
• Standard: 50
• Strong: 100-200

Note: This is the biggest time factor!
50 sims ≈ 2-3s/move, 200 sims ≈ 8-10s/move'''
            },

            'mcts_c_puct': {
                'label': 'MCTS Exploration (c_puct)',
                'default': 1.5,
                'type': 'float',
                'min': 0.1,
                'max': 5.0,
                'help': '''MCTS exploration constant.

Controls exploration vs exploitation trade-off.
Higher = more exploration

Formula: Q + c_puct * P * sqrt(N_parent) / (1 + N)

Typical values: 1.0-2.0

Suggested:
• Conservative: 1.0
• Standard: 1.5
• Exploratory: 2.0'''
            },

            'temperature_threshold': {
                'label': 'Temperature Threshold',
                'default': 30,
                'type': 'int',
                'min': 0,
                'max': 100,
                'help': '''Move number to switch from high to low temperature.

Before this move: temp = 1.0 (diverse play)
After this move: temp = 0.1 (optimal play)

This ensures training data has variety.

Typical values: 20-30

Suggested:
• Short games: 15-20
• Standard: 30
• Long games: 40+'''
            },

            # Neural network training
            'learning_rate': {
                'label': 'Learning Rate',
                'default': 0.001,
                'type': 'float',
                'min': 0.00001,
                'max': 0.1,
                'help': '''Learning rate for Adam optimizer.

Controls how fast the network learns.
Too high = unstable
Too low = slow learning

Typical values: 0.0001-0.01

Suggested:
• Conservative: 0.0001
• Standard: 0.001
• Aggressive: 0.01

Note: Adam adapts this automatically'''
            },

            'weight_decay': {
                'label': 'Weight Decay (L2 reg)',
                'default': 0.0001,
                'type': 'float',
                'min': 0.0,
                'max': 0.01,
                'help': '''L2 regularization strength.

Prevents overfitting by penalizing large weights.

Typical values: 0.0001-0.001

Suggested:
• No regularization: 0.0
• Standard: 0.0001
• Strong: 0.001'''
            },

            # Replay buffer
            'replay_buffer_size': {
                'label': 'Replay Buffer Size',
                'default': 50000,
                'type': 'int',
                'min': 1000,
                'max': 200000,
                'help': '''Maximum training examples to store.

Older data is discarded (FIFO).
Larger buffer = more diverse data

Typical values: 10,000-100,000

Suggested:
• Small: 10,000
• Standard: 50,000
• Large: 100,000

Memory usage: ~500MB for 50k examples'''
            },

            # Checkpointing
            'save_every': {
                'label': 'Save Checkpoint Every N Iterations',
                'default': 1,
                'type': 'int',
                'min': 1,
                'max': 50,
                'help': '''Save model checkpoint every N iterations.

1 = save every iteration (recommended)
5 = save every 5 iterations

Lower values = more checkpoints
(useful for finding best model)'''
            },

            'keep_checkpoints': {
                'label': 'Keep Last N Checkpoints',
                'default': 5,
                'type': 'int',
                'min': 1,
                'max': 100,
                'help': '''Number of recent checkpoints to keep.

Older checkpoints are deleted to save space.

0 = keep all (not recommended!)
5 = keep last 5

Disk usage: ~2MB per checkpoint'''
            },

            # Device
            'device': {
                'label': 'Device',
                'default': 'cpu',
                'type': 'choice',
                'choices': ['cpu', 'cuda'],
                'help': '''Computation device.

cpu: Works everywhere, slower
cuda: GPU acceleration (if available), 5-10x faster

Auto-detects CUDA availability.
If CUDA not available, falls back to CPU.'''
            },
        }

    def create_ui(self):
        """Crea UI per training"""
        # Main container con scrollbar
        main_container = tk.Frame(self, bg='white')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Header
        header = tk.Label(
            main_container,
            text="Neural Network Training",
            font=('Helvetica', 18, 'bold'),
            bg='white',
            fg='#2C3E50'
        )
        header.pack(pady=(0, 20))

        # Create notebook (tabs)
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill='both', expand=True)

        # Tab 1: Configuration
        config_tab = tk.Frame(notebook, bg='white')
        notebook.add(config_tab, text="Configuration")
        self.create_config_tab(config_tab)

        # Tab 2: Training Monitor
        monitor_tab = tk.Frame(notebook, bg='white')
        notebook.add(monitor_tab, text="Training Monitor")
        self.create_monitor_tab(monitor_tab)

    def create_config_tab(self, parent):
        """Crea tab configurazione"""
        # Configure grid weights
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Main frame with grid
        main_frame = tk.Frame(parent, bg='white')
        main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Create parameter inputs in grid layout
        self.create_parameter_inputs(main_frame)

        # Buttons frame at bottom
        button_frame = tk.Frame(parent, bg='white')
        button_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Buttons
        ttk.Button(
            button_frame,
            text="Load Config",
            command=self.load_configuration
        ).grid(row=0, column=0, padx=3)

        ttk.Button(
            button_frame,
            text="Save Config",
            command=self.save_configuration
        ).grid(row=0, column=1, padx=3)

        ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset_to_defaults
        ).grid(row=0, column=2, padx=3)

        # Start button (right side)
        self.start_button = ttk.Button(
            button_frame,
            text="▶ Start Training",
            command=self.start_training
        )
        self.start_button.grid(row=0, column=3, padx=3, sticky='e')
        button_frame.grid_columnconfigure(3, weight=1)

    def create_parameter_inputs(self, parent):
        """Crea inputs per tutti i parametri in layout a 2 colonne"""
        # Group parameters in 2 columns
        left_groups = {
            'Training Iterations': [
                'num_iterations', 'games_per_iteration',
                'training_batches', 'batch_size'
            ],
            'MCTS Configuration': [
                'mcts_simulations', 'mcts_c_puct', 'temperature_threshold'
            ],
        }

        right_groups = {
            'Neural Network': [
                'learning_rate', 'weight_decay'
            ],
            'Data & System': [
                'replay_buffer_size', 'save_every', 'keep_checkpoints', 'device'
            ]
        }

        # Left column
        left_col = tk.Frame(parent, bg='white')
        left_col.grid(row=0, column=0, sticky='nsew', padx=5)

        row_idx = 0
        for group_name, param_keys in left_groups.items():
            group_frame = tk.LabelFrame(
                left_col,
                text=group_name,
                font=('Helvetica', 11, 'bold'),
                bg='white',
                fg='#2C3E50',
                padx=10,
                pady=5
            )
            group_frame.grid(row=row_idx, column=0, sticky='ew', pady=5)
            row_idx += 1

            for param_key in param_keys:
                self.create_parameter_input(group_frame, param_key)

        # Right column
        right_col = tk.Frame(parent, bg='white')
        right_col.grid(row=0, column=1, sticky='nsew', padx=5)

        row_idx = 0
        for group_name, param_keys in right_groups.items():
            group_frame = tk.LabelFrame(
                right_col,
                text=group_name,
                font=('Helvetica', 11, 'bold'),
                bg='white',
                fg='#2C3E50',
                padx=10,
                pady=5
            )
            group_frame.grid(row=row_idx, column=0, sticky='ew', pady=5)
            row_idx += 1

            for param_key in param_keys:
                self.create_parameter_input(group_frame, param_key)

        # Configure column weights
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

    def create_parameter_input(self, parent, param_key):
        """Crea input per un singolo parametro con grid layout"""
        param_def = self.param_definitions[param_key]

        # Row frame
        row_frame = tk.Frame(parent, bg='white')
        row_frame.pack(fill='x', pady=2)

        # Label (più compatta)
        label = tk.Label(
            row_frame,
            text=param_def['label'] + ":",
            font=('Helvetica', 9),
            bg='white',
            fg='#34495E',
            anchor='w'
        )
        label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        # Input based on type
        if param_def['type'] == 'choice':
            # Dropdown
            var = tk.StringVar(value=param_def['default'])
            input_widget = ttk.Combobox(
                row_frame,
                textvariable=var,
                values=param_def['choices'],
                state='readonly',
                width=10
            )
        elif param_def['type'] in ['int', 'float']:
            # Entry
            var = tk.StringVar(value=str(param_def['default']))
            input_widget = ttk.Entry(row_frame, textvariable=var, width=10)

        input_widget.grid(row=0, column=1, padx=2)

        # Store variable
        self.params[param_key] = var

        # Help button (più piccolo)
        help_btn = tk.Button(
            row_frame,
            text="?",
            command=lambda: self.show_param_help(param_key),
            bg='#3498DB',
            fg='white',
            font=('Helvetica', 8, 'bold'),
            width=2,
            height=1,
            relief='flat',
            cursor='hand2'
        )
        help_btn.grid(row=0, column=2, padx=2)

        # Value range (più piccolo e compatto)
        if param_def['type'] in ['int', 'float']:
            range_text = f"({param_def['min']}-{param_def['max']})"
            range_label = tk.Label(
                row_frame,
                text=range_text,
                font=('Helvetica', 8),
                bg='white',
                fg='#95A5A6'
            )
            range_label.grid(row=0, column=3, padx=2)

        # Configure column weights
        row_frame.grid_columnconfigure(0, weight=1)

    def show_param_help(self, param_key):
        """Mostra help dialog per parametro"""
        param_def = self.param_definitions[param_key]

        # Create help window
        help_window = tk.Toplevel(self)
        help_window.title(f"Help: {param_def['label']}")
        help_window.geometry("500x400")
        help_window.transient(self)

        # Header
        header = tk.Label(
            help_window,
            text=param_def['label'],
            font=('Helvetica', 14, 'bold'),
            bg='#3498DB',
            fg='white',
            pady=10
        )
        header.pack(fill='x')

        # Help text
        help_text = scrolledtext.ScrolledText(
            help_window,
            wrap=tk.WORD,
            font=('Helvetica', 10),
            padx=10,
            pady=10
        )
        help_text.pack(fill='both', expand=True, padx=10, pady=10)
        help_text.insert('1.0', param_def['help'])
        help_text.config(state='disabled')

        # Close button
        ttk.Button(
            help_window,
            text="Close",
            command=help_window.destroy
        ).pack(pady=10)

    def create_monitor_tab(self, parent):
        """Crea tab per monitoring"""
        # Progress frame
        progress_frame = tk.LabelFrame(
            parent,
            text="Progress",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        progress_frame.pack(fill='x', padx=10, pady=10)

        # Progress label
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to start training...",
            font=('Helvetica', 10),
            bg='white',
            fg='#2C3E50'
        )
        self.progress_label.pack(anchor='w')

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill='x', pady=5)

        # Iteration info
        self.iteration_label = tk.Label(
            progress_frame,
            text="Iteration: 0/0",
            font=('Helvetica', 9),
            bg='white',
            fg='#7F8C8D'
        )
        self.iteration_label.pack(anchor='w')

        # Log frame
        log_frame = tk.LabelFrame(
            parent,
            text="Training Log",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Log text
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Courier', 9),
            bg='#1E1E1E',
            fg='#D4D4D4',
            height=15
        )
        self.log_text.pack(fill='both', expand=True)

        # Control buttons
        control_frame = tk.Frame(parent, bg='white')
        control_frame.pack(fill='x', padx=10, pady=10)

        self.stop_button = ttk.Button(
            control_frame,
            text="⏹ Stop Training",
            command=self.stop_training,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)

        ttk.Button(
            control_frame,
            text="Clear Log",
            command=lambda: self.log_text.delete('1.0', tk.END)
        ).pack(side='left', padx=5)

    def validate_parameters(self):
        """Valida parametri"""
        try:
            for key, var in self.params.items():
                param_def = self.param_definitions[key]
                value_str = var.get()

                if param_def['type'] == 'int':
                    value = int(value_str)
                    if value < param_def['min'] or value > param_def['max']:
                        raise ValueError(
                            f"{param_def['label']} must be between "
                            f"{param_def['min']} and {param_def['max']}"
                        )
                elif param_def['type'] == 'float':
                    value = float(value_str)
                    if value < param_def['min'] or value > param_def['max']:
                        raise ValueError(
                            f"{param_def['label']} must be between "
                            f"{param_def['min']} and {param_def['max']}"
                        )

            return True

        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return False

    def get_config_dict(self):
        """Ottieni configurazione come dictionary"""
        config = {}
        for key, var in self.params.items():
            param_def = self.param_definitions[key]
            value_str = var.get()

            if param_def['type'] == 'int':
                config[key] = int(value_str)
            elif param_def['type'] == 'float':
                config[key] = float(value_str)
            else:
                config[key] = value_str

        return config

    def start_training(self):
        """Avvia training"""
        if not self.validate_parameters():
            return

        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress!")
            return

        # Get config
        config = self.get_config_dict()

        # Confirm
        if not messagebox.askyesno(
            "Start Training",
            f"Start training with {config['num_iterations']} iterations?\n\n"
            f"This may take several hours.\n"
            f"The GUI will remain responsive."
        ):
            return

        # Update UI
        self.training_active = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar['value'] = 0
        self.log("Training started...")
        self.log(f"Configuration: {json.dumps(config, indent=2)}")

        # Start training thread
        self.training_thread = threading.Thread(
            target=self.run_training_thread,
            args=(config,),
            daemon=True
        )
        self.training_thread.start()

    def run_training_thread(self, config):
        """Run training in background thread"""
        try:
            # Import training components
            from src.game.encoder import BoardEncoder
            from src.game.decoder import MoveDecoder
            from src.models.chess_net import ChessNet
            from src.training.trainer import ChessTrainer
            from src.training.self_play import SelfPlayWorker
            from src.training.replay_buffer import ReplayBuffer
            import torch

            self.log("Initializing components...")

            # Create components
            encoder = BoardEncoder()
            decoder = MoveDecoder()
            model = ChessNet()

            replay_buffer = ReplayBuffer(max_size=config['replay_buffer_size'])

            trainer = ChessTrainer(
                model=model,
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                device=config['device']
            )

            self.log("✓ Components initialized")

            # Training loop
            for iteration in range(config['num_iterations']):
                if not self.training_active:
                    self.log("Training stopped by user")
                    break

                iter_start = time.time()
                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration + 1}/{config['num_iterations']}")
                self.log(f"{'='*60}")

                # Update progress
                progress = (iteration / config['num_iterations']) * 100
                self.update_progress(progress, f"Iteration {iteration+1}/{config['num_iterations']}")

                # Self-play
                self.log(f"[1/2] Self-Play: Generating {config['games_per_iteration']} games...")

                mcts_config = {
                    'num_simulations': config['mcts_simulations'],
                    'c_puct': config['mcts_c_puct'],
                    'temperature_threshold': config['temperature_threshold']
                }

                worker = SelfPlayWorker(encoder, decoder, model, mcts_config=mcts_config)
                examples, game_infos = worker.generate_games(
                    num_games=config['games_per_iteration'],
                    verbose=False
                )

                replay_buffer.add_games(examples, game_infos)
                self.log(f"  Generated {len(examples)} training examples")

                # Training
                self.log(f"[2/2] Training network on {config['training_batches']} batches...")

                epoch_stats = trainer.train_epoch(
                    replay_buffer,
                    batch_size=config['batch_size'],
                    num_batches=config['training_batches'],
                    verbose=False
                )

                if epoch_stats:
                    self.log(f"  Loss: {epoch_stats['avg_total_loss']:.4f}")
                    self.log(f"  Policy acc: {epoch_stats['avg_policy_accuracy']:.2%}")
                    self.log(f"  Value acc: {epoch_stats['avg_value_accuracy']:.2%}")

                # Save checkpoint
                if (iteration + 1) % config['save_every'] == 0:
                    checkpoint_path = f"checkpoints/model_iter_{iteration + 1}.pt"
                    trainer.save_checkpoint(checkpoint_path, epoch=iteration)
                    self.log(f"✓ Checkpoint saved: {checkpoint_path}")

                iter_time = time.time() - iter_start
                self.log(f"Iteration completed in {iter_time:.1f}s")

            # Training complete
            self.log(f"\n{'='*60}")
            self.log("TRAINING COMPLETE!")
            self.log(f"{'='*60}")
            self.update_progress(100, "Training complete!")

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())

        finally:
            self.training_active = False
            self.controller.after(0, self.training_finished)

    def stop_training(self):
        """Stop training"""
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            self.training_active = False
            self.log("Stopping training...")

    def training_finished(self):
        """Called when training finishes"""
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def log(self, message):
        """Aggiungi messaggio al log"""
        self.log_queue.put(message)

    def monitor_log(self):
        """Monitor log queue e aggiorna UI"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        # Schedule next check
        self.after(100, self.monitor_log)

    def update_progress(self, value, text):
        """Aggiorna progress bar"""
        def update():
            self.progress_bar['value'] = value
            self.progress_label.config(text=text)

        self.controller.after(0, update)

    def load_configuration(self):
        """Carica configurazione da file"""
        filepath = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            for key, value in config.items():
                if key in self.params:
                    self.params[key].set(str(value))

            messagebox.showinfo("Success", "Configuration loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def save_configuration(self):
        """Salva configurazione su file"""
        if not self.validate_parameters():
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            config = self.get_config_dict()

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo("Success", "Configuration saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def reset_to_defaults(self):
        """Reset tutti i parametri ai valori default"""
        if messagebox.askyesno("Reset", "Reset all parameters to default values?"):
            for key, var in self.params.items():
                default = self.param_definitions[key]['default']
                var.set(str(default))

    def on_show(self):
        """Called when frame is shown"""
        pass
