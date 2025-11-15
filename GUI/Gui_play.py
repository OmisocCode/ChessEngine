"""
ChessEngine GUI - Play Module

Interfaccia per giocare contro i modelli addestrati.

Features:
- Scacchiera grafica con pezzi Unicode
- Selezione modello e configurazione AI
- Selezione colore (White/Black)
- Move history
- Game status
- New game / Undo move
- Export PGN

Usage:
    Accessibile da GUI principale
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import sys
import chess
import chess.pgn
from datetime import datetime
import threading

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PlayFrame(tk.Frame):
    """
    Frame per giocare contro l'AI.

    Mostra scacchiera grafica e gestisce il gioco.
    """

    # Unicode chess pieces
    PIECE_SYMBOLS = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }

    def __init__(self, parent, controller):
        """
        Inizializza Play frame.

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent, bg='white')
        self.controller = controller
        self.grid(row=0, column=0, sticky="nsew")

        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.human_color = chess.WHITE
        self.ai_model = None
        self.ai_evaluator = None
        self.ai_mcts = None
        self.game_active = False
        self.ai_thinking = False

        # UI elements
        self.square_buttons = {}
        self.move_history = []

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Crea UI per play module"""
        # Header
        header_frame = tk.Frame(self, bg='white')
        header_frame.pack(fill='x', padx=20, pady=20)

        header = tk.Label(
            header_frame,
            text="Play vs AI",
            font=('Helvetica', 18, 'bold'),
            bg='white',
            fg='#2C3E50'
        )
        header.pack(side='left')

        # Main container
        main_container = tk.Frame(self, bg='white')
        main_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Left panel: Game setup and controls
        left_panel = tk.Frame(main_container, bg='white', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)

        self.create_setup_panel(left_panel)

        # Center: Board
        board_panel = tk.Frame(main_container, bg='white')
        board_panel.pack(side='left', fill='both', expand=False)

        self.create_board(board_panel)

        # Right panel: Move history and status
        right_panel = tk.Frame(main_container, bg='white', width=250)
        right_panel.pack(side='left', fill='y', padx=(20, 0))
        right_panel.pack_propagate(False)

        self.create_info_panel(right_panel)

    def create_setup_panel(self, parent):
        """Crea pannello setup"""
        # Model selection
        model_frame = tk.LabelFrame(
            parent,
            text="AI Configuration",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        model_frame.pack(fill='x', pady=(0, 10))

        # Model selector
        tk.Label(
            model_frame,
            text="Select Model:",
            font=('Helvetica', 10),
            bg='white'
        ).pack(anchor='w')

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state='readonly'
        )
        self.model_combo.pack(fill='x', pady=5)

        # Refresh models button
        ttk.Button(
            model_frame,
            text="🔄 Refresh Models",
            command=self.refresh_models
        ).pack(fill='x', pady=5)

        # MCTS simulations
        tk.Label(
            model_frame,
            text="AI Strength (MCTS sims):",
            font=('Helvetica', 10),
            bg='white'
        ).pack(anchor='w', pady=(10, 0))

        self.mcts_sims = tk.IntVar(value=50)
        ttk.Scale(
            model_frame,
            from_=10,
            to=200,
            variable=self.mcts_sims,
            orient='horizontal'
        ).pack(fill='x')

        self.mcts_label = tk.Label(
            model_frame,
            text="50 simulations",
            font=('Helvetica', 9),
            bg='white',
            fg='#7F8C8D'
        )
        self.mcts_label.pack()

        self.mcts_sims.trace('w', self.update_mcts_label)

        # Game settings
        game_frame = tk.LabelFrame(
            parent,
            text="Game Settings",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        game_frame.pack(fill='x', pady=(0, 10))

        # Color selection
        tk.Label(
            game_frame,
            text="Play as:",
            font=('Helvetica', 10),
            bg='white'
        ).pack(anchor='w')

        self.color_var = tk.StringVar(value="White")
        ttk.Radiobutton(
            game_frame,
            text="White",
            variable=self.color_var,
            value="White"
        ).pack(anchor='w')
        ttk.Radiobutton(
            game_frame,
            text="Black",
            variable=self.color_var,
            value="Black"
        ).pack(anchor='w')

        # Controls
        control_frame = tk.LabelFrame(
            parent,
            text="Game Controls",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        control_frame.pack(fill='x', pady=(0, 10))

        self.new_game_button = ttk.Button(
            control_frame,
            text="🎮 New Game",
            command=self.new_game
        )
        self.new_game_button.pack(fill='x', pady=2)

        self.undo_button = ttk.Button(
            control_frame,
            text="↶ Undo Move",
            command=self.undo_move,
            state='disabled'
        )
        self.undo_button.pack(fill='x', pady=2)

        ttk.Button(
            control_frame,
            text="💾 Export PGN",
            command=self.export_pgn
        ).pack(fill='x', pady=2)

        # Status
        self.status_label = tk.Label(
            parent,
            text="Click 'New Game' to start",
            font=('Helvetica', 10, 'bold'),
            bg='#3498DB',
            fg='white',
            pady=10
        )
        self.status_label.pack(fill='x', pady=(10, 0))

    def create_board(self, parent):
        """Crea scacchiera grafica"""
        # Board frame
        board_frame = tk.Frame(parent, bg='#2C3E50', padx=10, pady=10)
        board_frame.pack()

        # Coordinate labels
        coord_frame = tk.Frame(board_frame, bg='#2C3E50')
        coord_frame.grid(row=0, column=0)

        # Top file labels (a-h)
        file_label_top = tk.Frame(coord_frame, bg='#2C3E50')
        file_label_top.grid(row=0, column=1)
        for file_idx, file_name in enumerate('abcdefgh'):
            tk.Label(
                file_label_top,
                text=file_name,
                font=('Helvetica', 10, 'bold'),
                bg='#2C3E50',
                fg='white',
                width=5
            ).grid(row=0, column=file_idx)

        # Left rank labels (8-1)
        rank_label_left = tk.Frame(coord_frame, bg='#2C3E50')
        rank_label_left.grid(row=1, column=0)
        for rank_idx, rank_name in enumerate('87654321'):
            tk.Label(
                rank_label_left,
                text=rank_name,
                font=('Helvetica', 10, 'bold'),
                bg='#2C3E50',
                fg='white',
                height=2
            ).grid(row=rank_idx, column=0)

        # Board squares
        board_squares = tk.Frame(coord_frame, bg='black', padx=2, pady=2)
        board_squares.grid(row=1, column=1)

        # Create squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Flip rank for display

                # Square color
                is_light = (rank + file) % 2 == 0
                bg_color = '#F0D9B5' if is_light else '#B58863'

                # Create button
                btn = tk.Button(
                    board_squares,
                    text='',
                    font=('Arial Unicode MS', 32),
                    width=2,
                    height=1,
                    bg=bg_color,
                    activebackground=bg_color,
                    relief='flat',
                    cursor='hand2',
                    command=lambda sq=square: self.on_square_click(sq)
                )
                btn.grid(row=rank, column=file, padx=1, pady=1)

                self.square_buttons[square] = btn

        # Bottom file labels
        file_label_bottom = tk.Frame(coord_frame, bg='#2C3E50')
        file_label_bottom.grid(row=2, column=1)
        for file_idx, file_name in enumerate('abcdefgh'):
            tk.Label(
                file_label_bottom,
                text=file_name,
                font=('Helvetica', 10, 'bold'),
                bg='#2C3E50',
                fg='white',
                width=5
            ).grid(row=0, column=file_idx)

        # Right rank labels
        rank_label_right = tk.Frame(coord_frame, bg='#2C3E50')
        rank_label_right.grid(row=1, column=2)
        for rank_idx, rank_name in enumerate('87654321'):
            tk.Label(
                rank_label_right,
                text=rank_name,
                font=('Helvetica', 10, 'bold'),
                bg='#2C3E50',
                fg='white',
                height=2
            ).grid(row=rank_idx, column=0)

        # Update board display
        self.update_board_display()

    def create_info_panel(self, parent):
        """Crea pannello info"""
        # Move history
        history_frame = tk.LabelFrame(
            parent,
            text="Move History",
            font=('Helvetica', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        history_frame.pack(fill='both', expand=True)

        # Scrollable text
        self.history_text = tk.Text(
            history_frame,
            font=('Courier', 9),
            width=25,
            height=30,
            wrap=tk.WORD
        )
        history_text_scroll = ttk.Scrollbar(
            history_frame,
            command=self.history_text.yview
        )
        self.history_text.config(yscrollcommand=history_text_scroll.set)

        self.history_text.pack(side='left', fill='both', expand=True)
        history_text_scroll.pack(side='right', fill='y')

    def refresh_models(self):
        """Refresh lista modelli"""
        checkpoint_dir = Path("checkpoints")

        if not checkpoint_dir.exists():
            self.model_combo['values'] = ["No models found"]
            self.model_var.set("No models found")
            return

        checkpoint_files = sorted(
            checkpoint_dir.glob("model_iter_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True  # Most recent first
        )

        if not checkpoint_files:
            self.model_combo['values'] = ["No models found"]
            self.model_var.set("No models found")
            return

        model_names = [f"Iteration {p.stem.split('_')[-1]}" for p in checkpoint_files]
        self.model_combo['values'] = model_names
        self.model_var.set(model_names[0])  # Select most recent

    def update_mcts_label(self, *args):
        """Update MCTS label"""
        sims = self.mcts_sims.get()
        self.mcts_label.config(text=f"{sims} simulations")

    def new_game(self):
        """Avvia nuova partita"""
        # Check model selected
        if not self.model_var.get() or self.model_var.get() == "No models found":
            messagebox.showwarning("Warning", "Please select a model first")
            return

        # Confirm if game active
        if self.game_active:
            if not messagebox.askyesno("New Game", "Start a new game? Current game will be lost."):
                return

        # Load model
        try:
            self.load_ai_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            return

        # Reset game state
        self.board = chess.Board()
        self.selected_square = None
        self.move_history = []
        self.game_active = True

        # Set human color
        self.human_color = chess.WHITE if self.color_var.get() == "White" else chess.BLACK

        # Update UI
        self.update_board_display()
        self.update_move_history()
        self.undo_button.config(state='normal')
        self.update_status("Game started - Your turn!" if self.board.turn == self.human_color else "AI thinking...")

        # If AI plays first (human is black)
        if self.board.turn != self.human_color:
            self.ai_move()

    def load_ai_model(self):
        """Carica modello AI"""
        import torch
        from src.game.encoder import BoardEncoder
        from src.game.decoder import MoveDecoder
        from src.models.chess_net import ChessNet
        from src.mcts.mcts import MCTS
        from src.mcts.evaluator import NeuralNetworkEvaluator

        # Find checkpoint
        model_name = self.model_var.get()
        iter_num = model_name.split()[-1]
        checkpoint_path = Path(f"checkpoints/model_iter_{iter_num}.pt")

        # Create model
        encoder = BoardEncoder()
        decoder = MoveDecoder()
        model = ChessNet()

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create evaluator and MCTS
        self.ai_evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
        self.ai_mcts = MCTS(num_simulations=self.mcts_sims.get(), c_puct=1.5)

        self.update_status("Model loaded successfully")

    def on_square_click(self, square):
        """Gestisci click su casella"""
        if not self.game_active or self.ai_thinking:
            return

        # Not human's turn
        if self.board.turn != self.human_color:
            return

        # Game over
        if self.board.is_game_over():
            return

        # First click: select piece
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.highlight_square(square, '#F4D03F')  # Yellow highlight
                self.highlight_legal_moves(square)
        else:
            # Second click: try to move
            move = chess.Move(self.selected_square, square)

            # Check for promotion
            if self.is_promotion_move(self.selected_square, square):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                # Make move
                self.make_move(move)

                # AI's turn
                if not self.board.is_game_over():
                    self.ai_move()

            # Clear selection
            self.clear_highlights()
            self.selected_square = None

    def is_promotion_move(self, from_square, to_square):
        """Check se è una mossa di promozione"""
        piece = self.board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                return True
        return False

    def make_move(self, move):
        """Esegui mossa"""
        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append((move, san))

        self.update_board_display()
        self.update_move_history()

        # Check game over
        if self.board.is_game_over():
            self.game_over()

    def ai_move(self):
        """AI fa una mossa"""
        if not self.game_active or self.board.is_game_over():
            return

        self.ai_thinking = True
        self.update_status("AI thinking...")

        # Run in thread
        thread = threading.Thread(target=self._ai_move_thread, daemon=True)
        thread.start()

    def _ai_move_thread(self):
        """Thread per AI move"""
        try:
            # MCTS search
            root = self.ai_mcts.search(self.board, self.ai_evaluator)

            # Select move
            policy_dict = root.get_policy_distribution(temperature=0.1)
            best_move = max(policy_dict.items(), key=lambda x: x[1])[0]

            # Make move (on main thread)
            self.controller.after(0, lambda: self.make_move(best_move))
            self.controller.after(0, lambda: self.update_status("Your turn"))

        except Exception as e:
            self.controller.after(0, lambda: messagebox.showerror("Error", f"AI error:\n{e}"))

        finally:
            self.ai_thinking = False

    def undo_move(self):
        """Undo ultima mossa"""
        if not self.move_history or self.ai_thinking:
            return

        # Undo AI move
        if self.board.turn == self.human_color:
            self.board.pop()
            self.move_history.pop()

        # Undo human move
        if self.move_history:
            self.board.pop()
            self.move_history.pop()

        self.update_board_display()
        self.update_move_history()
        self.update_status("Move undone - Your turn")

    def game_over(self):
        """Gestisci fine partita"""
        self.game_active = False
        self.undo_button.config(state='disabled')

        result = self.board.result()
        outcome = self.board.outcome()

        if outcome.winner == self.human_color:
            msg = f"You win! ({result})"
        elif outcome.winner is None:
            msg = f"Draw! ({result})"
        else:
            msg = f"AI wins! ({result})"

        self.update_status(f"Game Over - {msg}")
        messagebox.showinfo("Game Over", msg)

    def update_board_display(self):
        """Aggiorna visualizzazione scacchiera"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)

            # Get piece symbol
            if piece:
                symbol = self.PIECE_SYMBOLS.get(piece.symbol(), '')
            else:
                symbol = ''

            # Update button
            btn = self.square_buttons[square]
            btn.config(text=symbol)

    def highlight_square(self, square, color):
        """Evidenzia casella"""
        self.square_buttons[square].config(bg=color)

    def highlight_legal_moves(self, from_square):
        """Evidenzia mosse legali"""
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                self.highlight_square(move.to_square, '#ABEBC6')  # Green

    def clear_highlights(self):
        """Rimuovi evidenziazioni"""
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                is_light = (rank + file) % 2 == 0
                bg_color = '#F0D9B5' if is_light else '#B58863'
                self.square_buttons[square].config(bg=bg_color)

    def update_move_history(self):
        """Aggiorna history"""
        self.history_text.delete('1.0', tk.END)

        for idx, (move, san) in enumerate(self.move_history):
            move_num = (idx // 2) + 1
            if idx % 2 == 0:
                self.history_text.insert(tk.END, f"{move_num}. {san} ")
            else:
                self.history_text.insert(tk.END, f"{san}\n")

        self.history_text.see(tk.END)

    def update_status(self, text):
        """Aggiorna status"""
        self.status_label.config(text=text)

    def export_pgn(self):
        """Esporta partita in PGN"""
        if not self.move_history:
            messagebox.showwarning("Warning", "No game to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export PGN",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            # Create PGN game
            game = chess.pgn.Game()

            # Headers
            game.headers["Event"] = "ChessEngine GUI Game"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = "Human" if self.human_color == chess.WHITE else "AI"
            game.headers["Black"] = "AI" if self.human_color == chess.WHITE else "Human"
            game.headers["Result"] = self.board.result() if self.board.is_game_over() else "*"

            # Add moves
            node = game
            board_copy = chess.Board()
            for move, _ in self.move_history:
                node = node.add_variation(move)
                board_copy.push(move)

            # Save
            with open(filepath, 'w') as f:
                print(game, file=f)

            messagebox.showinfo("Success", "Game exported successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")

    def on_show(self):
        """Called when frame is shown"""
        self.refresh_models()
