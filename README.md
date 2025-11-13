# ChessEngine
Chess AI - Educational Project
📋 Descrizione
Progetto educativo per costruire un'intelligenza artificiale che impara a giocare a scacchi tramite self-play reinforcement learning. L'obiettivo è comprendere i principi di AlphaZero in scala ridotta, praticabile su hardware consumer.
🎯 Obiettivi del Progetto
Obiettivo Principale
Creare un'AI che giochi a scacchi a livello principiante (~800-1000 ELO) e sia capace di:

✅ Giocare mosse legali al 100%
✅ Riconoscere pattern tattici semplici
✅ Trovare scacco matto in 2-3 mosse in posizioni elementari
✅ Applicare principi base (controllo centro, sviluppo pezzi)

Obiettivi Secondari (Didattici)

Comprendere il funzionamento di Monte Carlo Tree Search (MCTS)
Implementare una rete neurale con dual-head (policy + value)
Gestire un training loop di reinforcement learning
Valutare performance di un modello di gioco

🏗️ Struttura del Progetto
chess_ai_educational/
│
├── README.md
├── requirements.txt
├── config.yaml                 # Configurazione centralizzata
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chess_net.py        # Rete neurale principale
│   │   └── model_utils.py      # Helper per save/load modelli
│   │
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── node.py             # Nodo dell'albero MCTS
│   │   ├── mcts.py             # Algoritmo Monte Carlo Tree Search
│   │   └── evaluator.py        # Bridge tra MCTS e Neural Net
│   │
│   ├── game/
│   │   ├── __init__.py
│   │   ├── encoder.py          # Encoding scacchiera → tensor
│   │   ├── decoder.py          # Decoding output rete → mossa
│   │   └── game_state.py       # Wrapper per python-chess
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── self_play.py        # Generatore partite self-play
│   │   ├── trainer.py          # Training loop
│   │   └── replay_buffer.py    # Storage esperienze
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Valutazione vs baseline
│   │   └── puzzles.py          # Test su puzzle tattici
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging utilities
│       └── visualization.py    # Plot statistiche
│
├── data/
│   ├── games/                  # Partite generate (.pgn)
│   ├── training_data/          # Dataset training (.npz)
│   ├── checkpoints/            # Modelli salvati (.pth)
│   └── logs/                   # Training logs
│
├── notebooks/
│   ├── 01_board_encoding.ipynb
│   ├── 02_neural_net_test.ipynb
│   ├── 03_mcts_exploration.ipynb
│   └── 04_training_analysis.ipynb
│
├── scripts/
│   ├── train.py                # Script principale training
│   ├── play_vs_ai.py           # Gioca contro l'AI
│   ├── run_puzzles.py          # Test su puzzle tattici
│   └── compare_models.py       # Confronta checkpoints
│
└── tests/
    ├── test_encoder.py
    ├── test_mcts.py
    └── test_neural_net.py
🧠 Architettura Rete Neurale
Overview
Rete convoluzionale semplificata con architettura dual-head:
Input (Scacchiera)
    ↓
[Representation: 18 planes × 8×8]
    ↓
Conv Block 1: 64 filters
    ↓
Conv Block 2: 64 filters
    ↓
Conv Block 3: 128 filters
    ↓
    ├──→ Policy Head → 4672 logits (probabilità mosse)
    └──→ Value Head → 1 output (valutazione posizione)
Dettagli Tecnici
Input Layer

Shape: (batch, 18, 8, 8)
18 piani feature (semplificati rispetto ad AlphaZero):

12 piani: posizioni pezzi (P,N,B,R,Q,K × 2 colori)
1 piano: ripetizioni (detect draw)
1 piano: colore del turno
2 piani: diritti di arrocco (bianco/nero)
1 piano: en passant
1 piano: halfmove clock (regola 50 mosse)



Convolutional Body
python# Block 1
Conv2D(in=18, out=64, kernel=3, padding=1)
BatchNorm2D(64)
ReLU()

# Block 2
Conv2D(in=64, out=64, kernel=3, padding=1)
BatchNorm2D(64)
ReLU()

# Block 3
Conv2D(in=64, out=128, kernel=3, padding=1)
BatchNorm2D(128)
ReLU()
Policy Head
pythonConv2D(in=128, out=73, kernel=1)  # 73 = tipi di mossa
Flatten() → 73 × 8 × 8 = 4672
# Rappresenta: 64 posizioni × 73 direzioni possibili
Value Head
pythonConv2D(in=128, out=1, kernel=1)
Flatten() → 64 valori
Dense(64 → 32)
ReLU()
Dense(32 → 1)
Tanh()  # Output: [-1, 1]
Dimensionamento

Parametri totali: ~500K (training su CPU fattibile)
VRAM richiesta: ~500MB
Inference time: ~10-50ms su CPU

📚 Librerie e Dipendenze
Core Dependencies
txt# Deep Learning
torch>=2.0.0                # Neural network framework
torchvision>=0.15.0         # Utilities

# Chess Logic
python-chess>=1.999         # Gestione scacchiera e regole

# Data & Numeric
numpy>=1.24.0               # Array operations
pandas>=2.0.0               # Data analysis

# Visualization
matplotlib>=3.7.0           # Plotting
seaborn>=0.12.0             # Statistical plots

# Utilities
pyyaml>=6.0                 # Config files
tqdm>=4.65.0                # Progress bars
Optional (Development)
txt# Testing
pytest>=7.3.0

# Notebook
jupyter>=1.0.0
ipywidgets>=8.0.0

# Logging avanzato
tensorboard>=2.13.0         # Solo se hai spazio/tempo

# Code quality
black>=23.0.0               # Formatter
pylint>=2.17.0              # Linter
Installazione
bash# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
🚀 Step di Implementazione
Phase 0: Setup (1-2 giorni)

 Setup repository e virtual environment
 Installazione dipendenze
 Struttura cartelle
 Config file base

Phase 1: Game Logic (3-5 giorni)

 encoder.py: Conversione chess.Board → tensor 18×8×8
 decoder.py: Conversione output rete → chess.Move
 game_state.py: Wrapper per gestione partita
 Test unitari per encoding/decoding
 Notebook esplorativo 01_board_encoding.ipynb

Phase 2: Neural Network (3-5 giorni)

 chess_net.py: Implementazione architettura
 Forward pass test
 Save/load modello
 Test con input random
 Notebook 02_neural_net_test.ipynb

Phase 3: MCTS (5-7 giorni)

 node.py: Struttura nodo albero
 mcts.py: Algoritmo base (selection, expansion, backprop)
 evaluator.py: Integrazione rete in MCTS
 Test MCTS su posizioni semplici
 Notebook 03_mcts_exploration.ipynb

Phase 4: Self-Play (3-5 giorni)

 self_play.py: Generazione partite
 replay_buffer.py: Storage training data
 Genera prime 100 partite random
 Salvataggio dati in formato .npz
 Validazione formato dati

Phase 5: Training Loop (5-7 giorni)

 trainer.py: Training loop completo
 Loss function (policy + value)
 Optimizer setup (Adam o SGD)
 Logging metriche
 Checkpoint saving
 Script train.py eseguibile

Phase 6: Evaluation (3-5 giorni)

 evaluator.py: Valutazione vs random player
 puzzles.py: Test su puzzle mate-in-2
 Script play_vs_ai.py per giocare manualmente
 Script run_puzzles.py per benchmark
 Notebook 04_training_analysis.ipynb

Phase 7: Iterazioni & Tuning (ongoing)

 Training per 10-20 iterazioni
 Tuning iperparametri (learning rate, MCTS sims)
 Analisi convergenza
 Confronto tra checkpoint
 Documentazione risultati

⚙️ Configurazione Training
Iperparametri Consigliati (Hardware Limitato)
yaml# config.yaml

model:
  input_planes: 18
  conv_filters: [64, 64, 128]
  policy_output: 4672
  value_hidden: 32

mcts:
  num_simulations: 50          # Basso per velocità (vs 800 di AlphaZero)
  c_puct: 1.5                   # Exploration constant
  temperature: 1.0              # Sampling temperature
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

training:
  num_iterations: 20            # Numero cicli self-play → train
  games_per_iteration: 50       # Partite per iterazione
  batch_size: 64
  epochs_per_iteration: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  replay_buffer_size: 10000     # Ultimi 10k stati

evaluation:
  eval_games: 20                # Partite vs random per validazione
  puzzle_dataset: "mate_in_2"   # Puzzle semplici
Hardware Requirements

CPU: Qualsiasi Intel/AMD moderno (4+ cores raccomandati)
RAM: 8GB minimo, 16GB raccomandati
Storage: 5-10GB per dati + checkpoints
GPU: Opzionale (RTX 3050 o superiore accelera 5-10x)

Tempo Stimato

Training iteration: 1-3 ore (CPU) / 10-30 min (GPU entry-level)
Training completo (20 iter): 2-3 giorni CPU / 6-12 ore GPU

📊 Metriche di Successo
Metriche Quantitative

Move Legality: 100% mosse legali
Puzzle Accuracy: >50% su mate-in-2 semplici
Win Rate vs Random: >80%
Value Accuracy: MAE < 0.3 (su posizioni annotate)

Metriche Qualitative

Sviluppa pezzi nelle prime mosse
Non sacrifica pezzi senza motivo
Riconosce pattern di matto elementari (back rank, queen+rook)
Evita posizioni ovviamente perdenti
