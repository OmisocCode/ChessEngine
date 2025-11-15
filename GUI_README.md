# ChessEngine GUI - User Manual

Interfaccia grafica completa per ChessEngine AI.

## 🚀 Avvio

```bash
python GUI/Gui_scheletro.py
```

## 📦 Requisiti

La GUI usa **Tkinter** (incluso in Python standard), quindi **non richiede dipendenze extra**.

## 🎨 Moduli

### 1. 🏠 Welcome Page
- Pagina iniziale con overview del progetto
- Quick links ai moduli principali
- Documentazione integrata

### 2. 🎓 Training Module
Configura e avvia il training della rete neurale.

**Features**:
- ✅ **Configurazione completa** - Tutti i parametri con help tooltips
- ✅ **Help integrato** - Pulsante "?" per ogni parametro con spiegazione dettagliata
- ✅ **Live monitoring** - Progress bar e log in tempo reale
- ✅ **Save/Load config** - Salva configurazioni per riutilizzo
- ✅ **Stop/Resume** - Ferma e riprendi training

**Parametri configurabili**:

#### Training Iterations
- **Number of Iterations**: Cicli di train-play-update (default: 20)
- **Games per Iteration**: Partite self-play per iterazione (default: 50)
- **Training Batches**: Batches di training per iterazione (default: 100)
- **Batch Size**: Esempi per batch (default: 64)

#### MCTS Configuration
- **MCTS Simulations**: Simulazioni per mossa durante self-play (default: 50)
  - ⚠️ **Impatto maggiore sul tempo**: 50 sim ≈ 2-3s/mossa
- **MCTS c_puct**: Costante esplorazione (default: 1.5)
- **Temperature Threshold**: Mossa per switch temperatura (default: 30)

#### Neural Network
- **Learning Rate**: Learning rate Adam optimizer (default: 0.001)
- **Weight Decay**: L2 regularization (default: 0.0001)

#### Data Management
- **Replay Buffer Size**: Esempi massimi da mantenere (default: 50,000)

#### Checkpointing
- **Save Every N Iterations**: Frequenza salvataggio (default: 1)
- **Keep Last N Checkpoints**: Checkpoint da mantenere (default: 5)

#### System
- **Device**: 'cpu' o 'cuda' (auto-detect GPU)

**Uso**:
1. Configura parametri (o usa default)
2. Click "?" per help su ogni parametro
3. Click "▶ Start Training"
4. Monitora progress nella tab "Training Monitor"
5. Checkpoints salvati automaticamente in `checkpoints/`

**Suggerimenti**:
- **Test veloce**: iterations=2, games=10, simulations=20 (~10 min)
- **Training standard**: iterations=20, games=50, simulations=50 (~10-15 ore CPU)
- **Training forte**: iterations=50, games=100, simulations=100 (~30+ ore CPU o ~10 ore GPU)

---

### 3. 📊 Statistics Module
Visualizza statistiche e valuta performance dei modelli.

**Features**:
- ✅ **Checkpoint browser** - Lista tutti i checkpoint disponibili
- ✅ **Quick stats** - Info rapide su checkpoint selezionato
- ✅ **Eval vs Random** - Win rate contro random player + stima ELO
- ✅ **Puzzle test** - Test su puzzle tattici mate-in-2
- ✅ **Model comparison** - Confronto head-to-head tra modelli

#### Tab: Quick Stats
Mostra informazioni sul checkpoint selezionato:
- Numero iterazione
- Configurazione training
- Loss history (recente e migliore)
- Accuracies (policy e value)

**Uso**:
1. Seleziona checkpoint dalla lista
2. Click "Load Checkpoint Info"

#### Tab: Vs Random
Valuta modello contro random player.

**Parametri**:
- **Number of games**: Partite da giocare (default: 100, deve essere pari)
- **MCTS simulations**: Forza AI (default: 50)

**Output**:
- Win-Draw-Loss record
- Win rate percentuale
- **Stima ELO** (approssimativa vs random 800 ELO)
- Avg moves e tempo

**Interpretazione**:
- Win rate >80% → ELO ~1000+ → ✅ Obiettivo raggiunto!
- Win rate 70-80% → ELO ~900-1000
- Win rate <50% → Needs more training

**Uso**:
1. Seleziona checkpoint
2. Configura parametri
3. Click "▶ Run Evaluation"
4. Attendi risultati (~10-30 min per 100 games)

#### Tab: Puzzle Test
Test su puzzle tattici (mate-in-2).

**Parametri**:
- **Puzzle set**: 'mate_in_2' (10 puzzle) o 'all'
- **MCTS simulations**: Accuratezza (default: 100, più alto = più accurato)

**Output**:
- Total puzzles / Solved / Failed
- **Accuracy percentuale**
- Stats per categoria e difficoltà

**Interpretazione**:
- Accuracy >70% → Buone capacità tattiche
- Accuracy 50-70% → Principiante con conoscenze
- Accuracy >90% → Livello forte

**Uso**:
1. Seleziona checkpoint
2. Click "▶ Run Puzzle Test"
3. Attendi (~5-10 min per 10 puzzle)

#### Tab: Compare Models
Confronto diretto tra due checkpoint.

**Parametri**:
- **Compare with**: Seleziona secondo modello
- **Number of games**: Partite (default: 50)
- **MCTS simulations**: Forza per entrambi (default: 50)

**Output**:
- W-D-L per ogni modello
- Win rate percentuale
- Vincitore

**Uso**:
1. Seleziona primo checkpoint dalla lista
2. Seleziona secondo checkpoint da dropdown
3. Click "▶ Run Comparison"
4. Attendi (~30-60 min per 50 games)

---

### 4. ♟️ Play Module
Gioca contro i modelli addestrati.

**Features**:
- ✅ **Scacchiera grafica** - Unicode pieces, coordinate labels
- ✅ **Model selection** - Scegli quale checkpoint usare
- ✅ **AI strength** - Slider per simulazioni MCTS (10-200)
- ✅ **Color choice** - Gioca come White o Black
- ✅ **Move history** - Lista mosse in notazione SAN
- ✅ **Undo move** - Annulla ultima mossa
- ✅ **Export PGN** - Salva partita

**Controlli**:

#### Setup
1. **Select Model**: Scegli checkpoint (default: più recente)
   - Click "🔄 Refresh Models" per aggiornare lista

2. **AI Strength**: Slider MCTS simulations
   - 10-50: Veloce ma debole
   - 50-100: Bilanciato
   - 100-200: Forte ma lento (2-10s per mossa)

3. **Play as**: Scegli colore (White/Black)

#### Game
1. Click "🎮 New Game" per iniziare

2. **Fare una mossa**:
   - Click sul pezzo da muovere (evidenziato giallo)
   - Le mosse legali sono evidenziate in verde
   - Click sulla casella destinazione
   - Promozioni automatiche a Donna

3. **Undo Move**: Click "↶ Undo Move" per annullare
   - Annulla sia la tua mossa che quella dell'AI

4. **Fine partita**: Notifica automatica (checkmate/draw/stalemate)

#### Export
Click "💾 Export PGN" per salvare partita:
- Formato standard PGN
- Importabile su lichess.org/paste, chess.com/analysis
- Visualizzabile con qualsiasi software scacchi

**Tips**:
- **Modello untrained**: Gioca legale ma random/debole
- **Modello trained (20+ iter)**: Dovrebbe giocare decentemente
- **Aumenta simulazioni** per AI più forte (ma più lento)
- **Controlla move history** per analizzare partita

---

## 📁 File Structure

```
ChessEngine/
├── GUI/
│   ├── __init__.py               # Package init
│   ├── Gui_scheletro.py          # Main window
│   ├── Gui_train.py              # Training module
│   ├── Gui_statistiche.py        # Statistics module
│   └── Gui_play.py               # Play module
├── checkpoints/                   # Model checkpoints (auto-created)
├── logs/                         # Training logs (auto-created)
└── GUI_README.md                 # This file
```

---

## 🎯 Workflow Tipico

### 1. First Time Setup
```bash
# Avvia GUI
python GUI/Gui_scheletro.py

# Vai a Training module
# Configura parametri (o usa default)
# Start training
```

### 2. Durante Training
- Monitora progress in tab "Training Monitor"
- Checkpoints salvati automaticamente
- Puoi fermare e riprendere

### 3. Dopo Training
**Valutazione**:
```
Statistics → Seleziona checkpoint → Vs Random
# Target: win rate >80%

Statistics → Puzzle Test
# Target: accuracy >50%
```

**Gioca**:
```
Play → Select Model → New Game
# Testa skills contro AI!
```

### 4. Iterazioni Successive
- Confronta nuovi checkpoint con vecchi (Compare Models)
- Identifica miglior checkpoint
- Continua training se necessario

---

## ⚙️ Configurazioni Consigliate

### Test Veloce (per verificare funzionamento)
```
Training:
  Iterations: 2
  Games: 10
  Simulations: 20
Tempo: ~10-15 minuti
```

### Training Standard
```
Training:
  Iterations: 20
  Games: 50
  Simulations: 50
Tempo: ~10-15 ore CPU, ~3-5 ore GPU
Risultato atteso: Win rate 70-85%, Puzzle 60-70%
```

### Training Forte
```
Training:
  Iterations: 50
  Games: 100
  Simulations: 100
Tempo: ~30+ ore CPU, ~10-15 ore GPU
Risultato atteso: Win rate 85-90%+, Puzzle 75-85%
```

---

## 🐛 Troubleshooting

### "No models found"
- Devi fare training prima!
- Vai a Training module e avvia training
- Checkpoints appariranno in `checkpoints/`

### "Training too slow"
- Riduci MCTS simulations (20-30 invece di 50)
- Riduci games per iteration (25 invece di 50)
- Usa GPU se disponibile (device='cuda')

### "GUI non si avvia"
- Verifica Python installato con Tkinter:
  ```bash
  python -c "import tkinter; print('OK')"
  ```
- Su Linux: `sudo apt-get install python3-tk`

### "Model evaluation error"
- Verifica checkpoint esista
- Riavvia GUI
- Prova con checkpoint diverso

### "AI moves too slow in Play"
- Riduci slider MCTS simulations (10-30)
- Usa modello più recente (dovrebbe essere più veloce)

---

## 📚 Documentazione Aggiuntiva

Per dettagli tecnici, consulta:
- `STEP6_TRAINING.md` - Training system
- `STEP7_EVALUATION.md` - Evaluation system
- `VIEW_GAMES.md` - How to view PGN games
- `README.md` - Project overview

---

## 🎉 Tips & Tricks

### Salva Configurazioni Favorite
Training module → Configure → "Save Configuration"
- Salva config in JSON
- Ricarica con "Load Configuration"

### Confronto Progressi
Statistics → Compare Models
- Confronta iter_5 vs iter_10 vs iter_20
- Vedi miglioramento nel tempo

### Analisi Partite
Play → Export PGN
- Carica su lichess.org/paste
- Analizza con engine Stockfish
- Trova errori dell'AI

### Multi-Model Testing
Statistics → Vs Random
- Testa tutti i checkpoint
- Trova quello con miglior performance
- Non sempre l'ultimo è il migliore!

---

✅ **Buon divertimento con ChessEngine GUI!**

Per domande o bug, consulta la documentazione del progetto.
