# STEP 7: Sistema di Valutazione - Completato ✓

## Obiettivo
Implementare un sistema completo per valutare le performance dell'AI attraverso:
1. Match contro baseline (random player)
2. Test su puzzle tattici (mate-in-2, mate-in-3)
3. Confronto tra checkpoint diversi
4. Stima ELO rating

## File Creati

### 1. Model Evaluator (`src/evaluation/evaluator.py`) - 650+ righe

**Classi principali**:

#### GameResult
**Risultato di una singola partita**:
- `white_player`, `black_player`: Nomi giocatori
- `result`: Risultato ('1-0', '0-1', '1/2-1/2')
- `num_moves`: Numero mosse
- `time_taken`: Tempo totale
- `termination`: Motivo fine partita ('checkmate', 'stalemate', etc.)

**Metodi**:
- `winner()`: Ritorna nome vincitore o None se patta
- `to_dict()`: Converte a dictionary per salvataggio

#### ModelEvaluator
**Sistema completo di valutazione modelli**:

**Metodi core**:

##### `play_game(white_fn, black_fn, ...)`
Gioca una singola partita tra due policy functions.

**Parametri**:
- `white_player_fn`: Function (board) → move per il bianco
- `black_player_fn`: Function (board) → move per il nero
- `white_name`, `black_name`: Nomi per logging
- `max_moves`: Max mosse prima di dichiarare patta (default: 500)
- `verbose`: Stampa mosse durante partita

**Returns**: `GameResult` object

**Uso interno**: Questo è il metodo base usato da tutti gli altri.

##### `create_model_policy(model, mcts_sims, temperature)`
Crea una policy function da un neural network model.

**Parametri**:
- `model`: ChessNet model
- `mcts_simulations`: Numero simulazioni MCTS (default: 50)
- `temperature`: Temperature per sampling (default: 0.1 = quasi greedy)

**Returns**: Policy function (board) → move

**Implementazione**:
```python
# Crea evaluator e MCTS
nn_evaluator = NeuralNetworkEvaluator(encoder, model, decoder)
mcts = MCTS(num_simulations=mcts_sims)

def policy_fn(board):
    root = mcts.search(board, nn_evaluator)
    policy_dict = root.get_policy_distribution(temperature)

    # Greedy o sampling based on temperature
    if temperature < 0.5:
        return max(policy_dict.items(), key=lambda x: x[1])[0]
    else:
        # Sample da distribuzione
        return np.random.choice(moves, p=probs)

return policy_fn
```

##### `create_random_policy()`
Crea una policy function che gioca mosse casuali.

**Returns**: Policy function che seleziona move random

##### `evaluate_vs_random(model, num_games, mcts_sims, verbose)`
**Valuta modello contro random player**.

**Parametri**:
- `model`: ChessNet model da valutare
- `num_games`: Numero partite (deve essere pari)
- `mcts_simulations`: Simulazioni MCTS per il modello
- `verbose`: Stampa progresso

**Process**:
1. Crea policy per modello (MCTS + NN)
2. Crea policy random
3. Gioca `num_games` partite (metà come bianco, metà come nero)
4. Conta W-D-L
5. Calcola statistiche

**Returns** dictionary:
```python
{
    'total_games': 100,
    'wins': 85,
    'draws': 10,
    'losses': 5,
    'win_rate': 0.85,
    'avg_moves': 45.3,
    'avg_time': 2.1,  # secondi per partita
    'termination_counts': {
        'checkmate': 70,
        'stalemate': 10,
        ...
    },
    'games': [GameResult, ...]  # Lista completa partite
}
```

##### `compare_models(model1, model2, num_games, ...)`
**Confronta due modelli**.

**Parametri**:
- `model1`, `model2`: ChessNet models
- `num_games`: Numero partite
- `mcts_simulations`: Simulazioni per entrambi
- `model1_name`, `model2_name`: Nomi per display

**Process**:
1. Crea policy per entrambi i modelli
2. Gioca `num_games` partite (alternando colori)
3. Conta vittorie per ciascuno

**Returns** dictionary:
```python
{
    'model1_name': 'Iteration 5',
    'model2_name': 'Iteration 20',
    'total_games': 50,
    'model1_wins': 15,
    'model2_wins': 30,
    'draws': 5,
    'model1_win_rate': 0.30,
    'model2_win_rate': 0.60,
    'games': [GameResult, ...]
}
```

##### `save_results(results, filepath)`
Salva risultati in file JSON.

#### Funzione: `estimate_elo(win_rate, opponent_elo)`
**Stima approssimativa ELO rating**.

**Formula**:
```
ELO_diff = -400 * log10((1 - win_rate) / win_rate)
ELO = opponent_ELO + ELO_diff
```

**Esempio**:
```python
# Win rate 85% vs random (ELO ~800)
elo = estimate_elo(win_rate=0.85, opponent_elo=800)
# Risultato: ~1000 ELO
```

**Note**: Stima molto approssimativa. Per ELO accurato servono match calibrati.

---

### 2. Puzzle System (`src/evaluation/puzzles.py`) - 550+ righe

**Classi principali**:

#### Puzzle
**Rappresenta un singolo puzzle tattico**:

**Attributes**:
- `puzzle_id`: ID univoco (es: 'mate2_001')
- `fen`: Posizione iniziale in FEN notation
- `solution`: Lista mosse soluzione in UCI notation (es: ['e2e4', 'e7e5'])
- `category`: Categoria ('mate_in_2', 'mate_in_3', 'tactical_win')
- `difficulty`: Difficoltà ('easy', 'medium', 'hard')
- `description`: Descrizione opzionale

**Metodi**:
- `get_board()`: Crea chess.Board dalla FEN
- `is_solution_move(move, index)`: Verifica se mossa è corretta
- `to_dict()`, `from_dict()`: Serializzazione

**Esempio creazione puzzle**:
```python
puzzle = Puzzle(
    puzzle_id='mate2_001',
    fen='6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1',
    solution=['f1f8'],  # Rf8# (matto!)
    category='mate_in_2',
    difficulty='easy',
    description='Back rank mate with rook'
)
```

#### Built-in Puzzle Sets

##### MATE_IN_2_PUZZLES (10 puzzle)
Set di 10 puzzle mate-in-2 integrati:

1. **mate2_001**: Back rank mate con torre
2. **mate2_002**: Scholar's mate pattern
3. **mate2_003**: Back rank mate setup
4. **mate2_004**: Queen mate in corner
5. **mate2_005**: Two rooks mate
6. **mate2_006**: Bishop and queen mate
7. **mate2_007**: Greek gift sacrifice
8. **mate2_008**: Simple rook mate
9. **mate2_009**: Queen back rank mate
10. **mate2_010**: Anastasia's mate pattern

**Difficoltà**:
- Easy: 7 puzzle
- Medium: 3 puzzle

#### PuzzleTester
**Sistema di testing puzzle**:

##### `test_single_puzzle(puzzle, model, mcts_sims, verbose)`
Testa modello su un singolo puzzle.

**Process**:
1. Carica posizione da FEN
2. Esegue MCTS search
3. Ottiene mossa migliore (greedy, temp=0)
4. Confronta con soluzione corretta
5. Ritorna se risolto o meno

**Returns** dictionary:
```python
{
    'puzzle_id': 'mate2_001',
    'solved': True,
    'ai_move': 'f1f8',
    'correct_move': 'f1f8',
    'matches': True,
    'category': 'mate_in_2',
    'difficulty': 'easy'
}
```

**Note**: Verifica solo la PRIMA mossa della soluzione. Per puzzle multi-move servirebbero verifiche iterative.

##### `test_puzzles(puzzles, model, mcts_sims, verbose)`
Testa modello su lista di puzzle.

**Returns** dictionary con statistiche:
```python
{
    'total': 10,
    'solved': 7,
    'failed': 3,
    'accuracy': 0.70,

    # Per categoria
    'by_category': {
        'mate_in_2': {
            'total': 10,
            'solved': 7,
            'accuracy': 0.70
        }
    },

    # Per difficoltà
    'by_difficulty': {
        'easy': {
            'total': 7,
            'solved': 6,
            'accuracy': 0.857
        },
        'medium': {
            'total': 3,
            'solved': 1,
            'accuracy': 0.333
        }
    },

    # Lista risultati individuali
    'results': [dict, dict, ...]
}
```

##### `load_puzzles_from_file(filepath)`
Carica puzzle da file JSON.

**Formato file JSON**:
```json
[
    {
        "id": "custom_001",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
        "solution": ["f1f8"],
        "category": "mate_in_2",
        "difficulty": "easy",
        "description": "Back rank mate"
    },
    ...
]
```

##### `save_results(results, filepath)`
Salva risultati test in JSON.

#### Funzione: `get_builtin_puzzle_set(name)`
Ottieni set puzzle integrato.

**Parametri**:
- `name`: 'mate_in_2' o 'all'

**Returns**: Lista di Puzzle objects

---

### 3. Script: Run Puzzles (`scripts/run_puzzles.py`)

**Script eseguibile per testare modello su puzzle**.

**Usage**:
```bash
# Test con modello trained
python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt

# Più simulazioni (più accurato)
python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --simulations 200

# Test veloce (primi N puzzle)
python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --limit 5

# Salva risultati
python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt --output results.json
```

**Arguments**:
- `--model`: Path a checkpoint (se non specificato, usa untrained)
- `--puzzles`: Quale set ('mate_in_2', 'all')
- `--simulations`: MCTS simulations (default: 100)
- `--limit`: Testa solo primi N puzzle
- `--output`: Salva risultati JSON
- `--device`: 'cpu' o 'cuda'
- `--verbose`: Mostra dettagli per ogni puzzle

**Output esempio**:
```
======================================================================
                        PUZZLE TESTING
======================================================================

Initializing components...
Loading model from: checkpoints/model_iter_20.pt
✓ Model loaded (iteration 20)

Loading puzzle set: mate_in_2
✓ Loaded 10 puzzles

MCTS simulations per puzzle: 100
Starting tests...

[1/10] Testing mate2_001... ✓ SOLVED
[2/10] Testing mate2_002... ✓ SOLVED
[3/10] Testing mate2_003... ✗ FAILED (AI: e1e7, Correct: e1e8)
...

======================================================================
RESULTS
======================================================================
Total puzzles: 10
Solved: 7
Failed: 3
Accuracy: 70.0%

By category:
  mate_in_2: 7/10 (70.0%)

By difficulty:
  easy: 6/7 (85.7%)
  medium: 1/3 (33.3%)
======================================================================

SUMMARY
======================================================================
Overall Accuracy: 70.0%
Rating: Good! 👍

Context:
  ~90%+ : Strong club player level
  ~70%+ : Intermediate player
  ~50%+ : Beginner who knows tactics
  <30%  : Needs more training
======================================================================
```

---

### 4. Script: Compare Models (`scripts/compare_models.py`)

**Script per confrontare due modelli**.

**Usage**:
```bash
# Confronta due checkpoint
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_10.pt \
    --model2 checkpoints/model_iter_20.pt \
    --games 50

# Confronta vs random
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_20.pt \
    --random \
    --games 100

# Più simulazioni (più accurato)
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_15.pt \
    --model2 checkpoints/model_iter_20.pt \
    --games 30 \
    --simulations 200

# Salva risultati
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_10.pt \
    --model2 checkpoints/model_iter_20.pt \
    --games 50 \
    --output comparison.json
```

**Arguments**:
- `--model1`: Path primo modello (required)
- `--model2`: Path secondo modello
- `--random`: Confronta vs random invece di model2
- `--games`: Numero partite (deve essere pari, default: 50)
- `--simulations`: MCTS simulations (default: 50)
- `--device`: 'cpu' o 'cuda'
- `--output`: Salva risultati JSON

**Output esempio**:
```
======================================================================
                       MODEL COMPARISON
======================================================================

Initializing components...
✓ Components initialized

Loading Model 1...
Loading: checkpoints/model_iter_10.pt
  ✓ Loaded (iteration 10)

Loading Model 2...
Loading: checkpoints/model_iter_20.pt
  ✓ Loaded (iteration 20)

======================================================================
MATCH SETUP
======================================================================
Competitor 1: Model 1 (iter 10)
Competitor 2: Model 2 (iter 20)
Number of games: 50
MCTS simulations: 50
======================================================================

Running head-to-head comparison...

Playing game 10/50...
Playing game 20/50...
...

======================================================================
RESULTS
======================================================================
Model 1 (iter 10): 15W - 8D - 27L (30.0%)
Model 2 (iter 20): 27W - 8D - 15L (54.0%)

Winner: Model 2 (iter 20)
======================================================================

======================================================================
ANALYSIS
======================================================================
Win margin: 12 games
Strength assessment: Moderate difference

Recommendation:
  → Use Model 2 (iter 20)

Confidence (decisive games): 64.3%
======================================================================
```

---

## Pipeline Completo di Valutazione

### 1. Durante il Training

**Ogni N iterazioni, valuta progresso**:
```bash
# Dopo iter 5
python scripts/run_puzzles.py --model checkpoints/model_iter_5.pt

# Dopo iter 10
python scripts/run_puzzles.py --model checkpoints/model_iter_10.pt

# Confronta miglioramento
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_5.pt \
    --model2 checkpoints/model_iter_10.pt \
    --games 30
```

### 2. Fine Training

**Valutazione completa**:
```bash
# 1. Test vs random (stima ELO)
python -c "
from src.evaluation.evaluator import ModelEvaluator, estimate_elo
from src.game.encoder import BoardEncoder
from src.game.decoder import MoveDecoder
from src.models.chess_net import ChessNet
import torch

encoder = BoardEncoder()
decoder = MoveDecoder()
model = ChessNet()
checkpoint = torch.load('checkpoints/model_iter_20.pt')
model.load_state_dict(checkpoint['model_state_dict'])

evaluator = ModelEvaluator(encoder, decoder)
results = evaluator.evaluate_vs_random(model, num_games=100, mcts_simulations=50)

print(f'Win rate: {results[\"win_rate\"]:.1%}')
elo = estimate_elo(results['win_rate'], opponent_elo=800)
print(f'Estimated ELO: ~{elo}')
"

# 2. Test puzzle tattici
python scripts/run_puzzles.py \
    --model checkpoints/model_iter_20.pt \
    --simulations 100

# 3. Confronta con versioni precedenti
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_1.pt \
    --model2 checkpoints/model_iter_20.pt \
    --games 50
```

### 3. Selezione Best Checkpoint

**Trova il checkpoint migliore**:
```bash
# Script bash per testare tutti i checkpoint
for checkpoint in checkpoints/model_iter_*.pt; do
    echo "Testing $checkpoint"
    python scripts/run_puzzles.py --model $checkpoint --limit 10 | grep "Accuracy"
done
```

---

## Come Interpretare i Risultati

### Win Rate vs Random

| Win Rate | Livello Stimato | ELO Approssimativo |
|----------|-----------------|-------------------|
| 95%+     | Forte           | ~1100+            |
| 85-95%   | Intermedio      | ~1000-1100        |
| 70-85%   | Principiante+   | ~900-1000         |
| 50-70%   | Principiante    | ~800-900          |
| <50%     | Molto debole    | <800              |

### Puzzle Accuracy

| Accuracy | Livello Tattico |
|----------|----------------|
| 90%+     | Forte club player |
| 70-90%   | Intermedio |
| 50-70%   | Principiante con conoscenze tattiche |
| 30-50%   | Principiante |
| <30%     | Needs training |

### Model Comparison

**Confidence levels**:
- Confidence >70%: Differenza significativa
- Confidence 55-70%: Differenza moderata
- Confidence <55%: Serve più games per confermare

**Win margin**:
- Margin >40%: Uno molto più forte
- Margin 20-40%: Differenza moderata
- Margin <20%: Quasi pari

---

## Esempi d'Uso

### Esempio 1: Valutazione Post-Training

```bash
# 1. Hai finito training (20 iterazioni)
# 2. Vuoi sapere quanto è forte il modello

# Test vs random (100 partite)
python -m src.evaluation.evaluator  # Usa il demo integrato

# O più diretto:
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_20.pt \
    --random \
    --games 100

# Output:
# Win rate: 82%
# Estimated ELO: ~1050
```

### Esempio 2: Test Puzzle

```bash
# Testa capacità tattiche
python scripts/run_puzzles.py \
    --model checkpoints/model_iter_20.pt \
    --simulations 100

# Output:
# Accuracy: 65% (6.5/10 puzzle risolti)
# Rating: Decent, but needs improvement
```

### Esempio 3: Trova Miglior Checkpoint

```bash
# Confronta iter_15 vs iter_20
python scripts/compare_models.py \
    --model1 checkpoints/model_iter_15.pt \
    --model2 checkpoints/model_iter_20.pt \
    --games 50

# Output:
# Model 1: 20W - 5D - 25L
# Model 2: 25W - 5D - 20L
# Winner: Model 2
# → Usa iter_20
```

### Esempio 4: Progressive Testing

```bash
# Durante training, ogni 5 iterazioni
# (Aggiungere al loop di training o eseguire manualmente)

for iter in 5 10 15 20; do
    echo "=== Iteration $iter ==="
    python scripts/run_puzzles.py \
        --model checkpoints/model_iter_$iter.pt \
        --limit 5 \
        --simulations 50
done

# Output mostra trend di miglioramento:
# Iter 5:  accuracy 20%
# Iter 10: accuracy 40%
# Iter 15: accuracy 55%
# Iter 20: accuracy 70%  ← Miglioramento!
```

---

## File di Output

### JSON Results Format

**evaluate_vs_random output**:
```json
{
  "total_games": 100,
  "wins": 85,
  "draws": 10,
  "losses": 5,
  "win_rate": 0.85,
  "avg_moves": 42.3,
  "avg_time": 2.1,
  "termination_counts": {
    "checkmate": 70,
    "stalemate": 8,
    "insufficient_material": 2,
    "max_moves": 20
  },
  "games": [
    {
      "white": "Model",
      "black": "Random",
      "result": "1-0",
      "num_moves": 38,
      "time_taken": 1.9,
      "termination": "checkmate"
    },
    ...
  ]
}
```

**puzzle test output**:
```json
{
  "total": 10,
  "solved": 7,
  "failed": 3,
  "accuracy": 0.7,
  "by_category": {
    "mate_in_2": {
      "total": 10,
      "solved": 7,
      "accuracy": 0.7
    }
  },
  "by_difficulty": {
    "easy": {"total": 7, "solved": 6, "accuracy": 0.857},
    "medium": {"total": 3, "solved": 1, "accuracy": 0.333}
  },
  "results": [
    {
      "puzzle_id": "mate2_001",
      "solved": true,
      "ai_move": "f1f8",
      "correct_move": "f1f8",
      "matches": true,
      "category": "mate_in_2",
      "difficulty": "easy"
    },
    ...
  ]
}
```

---

## Troubleshooting

### "Accuracy troppo bassa (<20%)"
- Modello probabilmente not trained o undertrained
- Esegui più iterazioni di training
- Aumenta MCTS simulations per test (es: 200)

### "Comparison non conclusivo (50-50)"
- Modelli troppo simili in forza
- Aumenta numero games (es: 100)
- Aumenta MCTS simulations

### "Test puzzle troppo lento"
- Riduci `--simulations` (es: 50 invece di 100)
- Usa `--limit` per testare solo primi N puzzle
- Usa GPU (`--device cuda`)

### "Want more puzzles"
- Aggiungi puzzle custom in file JSON
- Carica con `tester.load_puzzles_from_file('my_puzzles.json')`
- Oppure estendi MATE_IN_2_PUZZLES in `puzzles.py`

---

## Estensioni Future

### Già Implementato ✓
- Evaluation vs random player
- Puzzle testing (mate-in-2)
- Model comparison
- ELO estimation
- Detailed statistics

### Possibili Aggiunte
- **Più puzzle sets**: mate-in-3, tactical wins, endgame puzzles
- **Opening book evaluation**: Test su posizioni di apertura standard
- **Endgame tablebase check**: Verifica gioco finale vs tablebase
- **Time control testing**: Test con limiti di tempo
- **Visualizzazione avanzata**: Plot learning curves, heatmaps
- **Web interface**: Dashboard per visualizzare risultati
- **Lichess/Chess.com integration**: Gioca su piattaforme online

---

## Metriche di Successo del Progetto

Obiettivi originali del README:

| Metrica | Target | Come Testare |
|---------|--------|--------------|
| Move Legality | 100% | Verificato in play_vs_ai.py |
| Puzzle Accuracy | >50% mate-in-2 | `run_puzzles.py` |
| Win Rate vs Random | >80% | `compare_models.py --random` |
| Value Accuracy | MAE <0.3 | (Implementabile con position annotations) |

**Test Complete Success**:
```bash
# 1. Win rate >80%
python scripts/compare_models.py --model1 checkpoints/model_iter_20.pt --random --games 100
# Output: win_rate >= 0.80

# 2. Puzzle accuracy >50%
python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt
# Output: accuracy >= 0.50

# 3. ELO estimate
# Se win_rate 80%+ → ELO ~1000+ → SUCCESSO! 🎉
```

---

✅ **STEP 7 Completato!**

Il sistema di valutazione è completo e permette di misurare quantitativamente
i progressi dell'AI durante e dopo il training.

**Prossimi step**:
1. **Train model**: `python scripts/train.py --iterations 20`
2. **Evaluate**: `python scripts/run_puzzles.py --model checkpoints/model_iter_20.pt`
3. **Test vs random**: `python scripts/compare_models.py --model1 checkpoints/model_iter_20.pt --random --games 100`
4. **Celebrate** se raggiungi gli obiettivi! 🏆
