"""
ChessEngine GUI - Main Application Window

Questa è la finestra principale dell'applicazione GUI per ChessEngine.
Gestisce la navigazione tra i diversi moduli:
- Training: Configura e avvia training
- Statistiche: Visualizza progressi e metriche
- Play: Gioca contro i modelli salvati

La GUI usa Tkinter (built-in Python, cross-platform).

Usage:
    python GUI/Gui_scheletro.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ChessEngineApp(tk.Tk):
    """
    Main application window per ChessEngine GUI.

    Gestisce:
    - Window layout principale
    - Navigation tra moduli
    - Frame switching
    - Menu bar

    Attributes:
        frames: Dict di frame per ogni modulo
        current_frame: Frame attualmente visualizzato
    """

    def __init__(self):
        """Inizializza main window"""
        super().__init__()

        # Window config
        self.title("ChessEngine AI - Training & Evaluation GUI")
        self.geometry("1200x800")

        # Centra la finestra
        self.center_window()

        # Style configuration
        self.configure_styles()

        # Container per frames
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Dictionary per contenere i frames
        self.frames = {}
        self.current_frame_name = None

        # Create menu bar
        self.create_menu_bar()

        # Create navigation sidebar
        self.create_sidebar()

        # Create welcome page
        self.create_welcome_frame()

        # Show welcome page
        self.show_frame("Welcome")

    def center_window(self):
        """Centra la finestra sullo schermo"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def configure_styles(self):
        """Configura stili ttk"""
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme

        # Custom styles
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Helvetica', 12))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Sidebar.TButton', padding=10, font=('Helvetica', 11))

    def create_menu_bar(self):
        """Crea menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Home", command=lambda: self.show_frame("Welcome"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Modules menu
        modules_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modules", menu=modules_menu)
        modules_menu.add_command(label="Training", command=self.show_training)
        modules_menu.add_command(label="Statistics", command=self.show_statistics)
        modules_menu.add_command(label="Play vs AI", command=self.show_play)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_docs)

    def create_sidebar(self):
        """Crea sidebar con navigazione"""
        sidebar = tk.Frame(self, bg='#2C3E50', width=200)
        sidebar.pack(side="left", fill="y")

        # Logo/Title
        title_label = tk.Label(
            sidebar,
            text="ChessEngine AI",
            bg='#2C3E50',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            pady=20
        )
        title_label.pack()

        # Separator
        tk.Frame(sidebar, height=2, bg='#34495E').pack(fill='x', padx=10)

        # Navigation buttons
        buttons = [
            ("🏠 Home", lambda: self.show_frame("Welcome")),
            ("🎓 Training", self.show_training),
            ("📊 Statistics", self.show_statistics),
            ("♟️  Play vs AI", self.show_play),
        ]

        for text, command in buttons:
            btn = tk.Button(
                sidebar,
                text=text,
                command=command,
                bg='#34495E',
                fg='white',
                font=('Helvetica', 11),
                relief='flat',
                pady=15,
                cursor='hand2',
                activebackground='#1ABC9C',
                activeforeground='white'
            )
            btn.pack(fill='x', padx=5, pady=2)

            # Hover effects
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#1ABC9C'))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg='#34495E'))

    def create_welcome_frame(self):
        """Crea welcome page"""
        frame = tk.Frame(self.container, bg='white')
        frame.grid(row=0, column=0, sticky="nsew")

        # Welcome content
        welcome_container = tk.Frame(frame, bg='white')
        welcome_container.place(relx=0.5, rely=0.5, anchor='center')

        # Title
        title = tk.Label(
            welcome_container,
            text="Welcome to ChessEngine AI",
            font=('Helvetica', 24, 'bold'),
            bg='white',
            fg='#2C3E50'
        )
        title.pack(pady=20)

        # Subtitle
        subtitle = tk.Label(
            welcome_container,
            text="Educational Chess AI using AlphaZero Principles",
            font=('Helvetica', 14),
            bg='white',
            fg='#7F8C8D'
        )
        subtitle.pack()

        # Description
        desc_frame = tk.Frame(welcome_container, bg='white')
        desc_frame.pack(pady=30)

        descriptions = [
            ("🎓", "Training", "Configure and run neural network training"),
            ("📊", "Statistics", "Analyze model performance and progress"),
            ("♟️", "Play", "Test your skills against trained AI"),
        ]

        for emoji, title_text, desc_text in descriptions:
            module_frame = tk.Frame(desc_frame, bg='white')
            module_frame.pack(pady=10)

            tk.Label(
                module_frame,
                text=f"{emoji} {title_text}",
                font=('Helvetica', 14, 'bold'),
                bg='white',
                fg='#2C3E50'
            ).pack(anchor='w')

            tk.Label(
                module_frame,
                text=desc_text,
                font=('Helvetica', 11),
                bg='white',
                fg='#7F8C8D'
            ).pack(anchor='w', padx=30)

        # Quick start buttons
        button_frame = tk.Frame(welcome_container, bg='white')
        button_frame.pack(pady=30)

        ttk.Button(
            button_frame,
            text="Start Training",
            command=self.show_training,
            style='Sidebar.TButton'
        ).pack(side='left', padx=10)

        ttk.Button(
            button_frame,
            text="View Statistics",
            command=self.show_statistics,
            style='Sidebar.TButton'
        ).pack(side='left', padx=10)

        ttk.Button(
            button_frame,
            text="Play vs AI",
            command=self.show_play,
            style='Sidebar.TButton'
        ).pack(side='left', padx=10)

        # Store frame
        self.frames["Welcome"] = frame

    def show_training(self):
        """Mostra modulo Training"""
        if "Training" not in self.frames:
            # Lazy loading - carica modulo solo quando necessario
            try:
                from GUI.Gui_train import TrainingFrame
                frame = TrainingFrame(self.container, self)
                self.frames["Training"] = frame
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Training module:\n{e}")
                return

        self.show_frame("Training")

    def show_statistics(self):
        """Mostra modulo Statistics"""
        if "Statistics" not in self.frames:
            try:
                from GUI.Gui_statistiche import StatisticsFrame
                frame = StatisticsFrame(self.container, self)
                self.frames["Statistics"] = frame
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Statistics module:\n{e}")
                return

        self.show_frame("Statistics")

    def show_play(self):
        """Mostra modulo Play"""
        if "Play" not in self.frames:
            try:
                from GUI.Gui_play import PlayFrame
                frame = PlayFrame(self.container, self)
                self.frames["Play"] = frame
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Play module:\n{e}")
                return

        self.show_frame("Play")

    def show_frame(self, frame_name):
        """
        Mostra il frame specificato.

        Args:
            frame_name: Nome del frame da mostrare
        """
        if frame_name not in self.frames:
            messagebox.showerror("Error", f"Frame '{frame_name}' not found")
            return

        # Hide current frame
        if self.current_frame_name and self.current_frame_name in self.frames:
            self.frames[self.current_frame_name].grid_forget()

        # Show new frame
        frame = self.frames[frame_name]
        frame.grid(row=0, column=0, sticky="nsew")
        self.current_frame_name = frame_name

        # Call on_show if exists (per refresh data)
        if hasattr(frame, 'on_show'):
            frame.on_show()

    def show_about(self):
        """Mostra dialog About"""
        about_text = """
ChessEngine AI
Version 1.0

An educational chess AI project implementing
AlphaZero-style reinforcement learning.

Features:
• Neural network training with self-play
• MCTS (Monte Carlo Tree Search)
• Tactical puzzle evaluation
• Model comparison tools

Developed as an educational project to understand
deep reinforcement learning in game playing.
        """
        messagebox.showinfo("About ChessEngine AI", about_text)

    def show_docs(self):
        """Mostra informazioni su documentazione"""
        docs_text = """
Documentation is available in the project directory:

• README.md - Project overview
• STEP6_TRAINING.md - Training system guide
• STEP7_EVALUATION.md - Evaluation system guide
• VIEW_GAMES.md - How to view games in PGN

For online help, visit:
https://github.com/OmisocCode/ChessEngine
        """
        messagebox.showinfo("Documentation", docs_text)


def main():
    """Main entry point"""
    app = ChessEngineApp()
    app.mainloop()


if __name__ == "__main__":
    main()
