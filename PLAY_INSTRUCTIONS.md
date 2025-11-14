# Come Giocare contro l'AI

## Installazione Dipendenze

Prima di giocare, installa le dipendenze necessarie:

```bash
# Dipendenze minime (senza rete neurale)
pip install python-chess numpy

# Con rete neurale (raccomandato)
pip install python-chess numpy torch
```

## Avviare il Gioco

```bash
# Dalla directory principale del progetto
python scripts/play_vs_ai.py

# Oppure
./scripts/play_vs_ai.py
```

## Opzioni di Gioco

### 1. Scegli il Colore
- **White**: Giochi per primo
- **Black**: L'AI gioca per prima

### 2. Scegli la Forza dell'AI

| Livello | Simulazioni | Tempo per Mossa | Descrizione |
|---------|-------------|-----------------|-------------|
| Weak | 10 | ~0.2s | Principiante |
| Normal | 50 | ~1s | Intermedio |
| Strong | 100 | ~2s | Avanzato |
| Very Strong | 200 | ~4s | Esperto |
| Custom | 10-800 | Variabile | Personalizza |

**Raccomandazione**: Inizia con **Normal (50)** per un buon equilibrio velocità/forza.

### 3. Scegli l'Evaluator

| Evaluator | Descrizione | Requisiti |
|-----------|-------------|-----------|
| Neural Network | Rete neurale (non addestrata) | PyTorch |
| Random | Mosse casuali uniformi | Nessuno |
| Heuristic | Valutazione materiale | Nessuno |

**Nota**: La rete neurale **NON è addestrata**, quindi giocherà in modo casuale/debole. L'evaluator **Heuristic** è il più forte al momento.

## Come Inserire le Mosse

### Formato UCI (Universal Chess Interface)

Le mosse si inseriscono in formato UCI: `da-a`

**Esempi**:
- `e2e4` - Pedone da e2 a e4
- `g1f3` - Cavallo da g1 a f3
- `e1g1` - Arrocco corto (re)
- `e7e8q` - Promozione a regina
- `e7e8n` - Promozione a cavallo

### Comandi Durante il Gioco

| Comando | Descrizione |
|---------|-------------|
| `e2e4` | Gioca una mossa |
| `legal` | Mostra tutte le mosse legali |
| `help` | Mostra aiuto |
| `quit` | Esci dal gioco |

## Schermata di Gioco

```
========================================
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜  8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟  7
6 · · · · · · · ·  6
5 · · · · · · · ·  5
4 · · · · · · · ·  4
3 · · · · · · · ·  3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙  2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖  1
  a b c d e f g h
========================================

Move 1 - White to move

Your move (e.g., 'e2e4', or 'quit' to exit):
```

## Informazioni AI Durante il Gioco

Quando l'AI pensa, vedrai:

```
AI is thinking...
Running 50 MCTS simulations...
✓ Search completed in 1.23s
Position value: 0.042 (AI's perspective)
Best move visits: 18/50
Top moves considered:
  1. e7e5 (18 visits, Q=0.045)
  2. d7d5 (12 visits, Q=0.038)
  3. g8f6 (8 visits, Q=0.035)

AI plays: e7e5
```

**Interpretazione**:
- **Position value**: Valutazione della posizione (>0 = AI in vantaggio, <0 = AI in svantaggio)
- **Best move visits**: Quante simulazioni hanno scelto questa mossa
- **Q-value**: Valore medio della mossa dalle simulazioni

## Fine Partita

Il gioco termina automaticamente quando:
- **Scacco matto**: Un giocatore vince
- **Stallo**: Pareggio
- **Materiale insufficiente**: Pareggio (es. Re vs Re)
- **Regola 50 mosse**: Pareggio
- **Tripla ripetizione**: Pareggio

Al termine vedrai le statistiche:
```
========================================
GAME OVER
========================================

White wins by checkmate!

Game statistics:
  Total moves: 42
  AI total thinking time: 52.3s
  AI average per move: 2.5s
```

## Tips per Giocare

1. **Inizia con Heuristic evaluator**: È più forte della rete non addestrata
2. **Usa 50-100 simulazioni**: Buon equilibrio velocità/forza
3. **Comando `legal`**: Utile se non sei sicuro delle mosse disponibili
4. **Temperature bassa (0.1)**: L'AI gioca più deterministicamente (default nello script)

## Troubleshooting

### Errore: "No module named 'chess'"
```bash
pip install python-chess
```

### Errore: "No module named 'torch'"
```bash
# Rete neurale non disponibile, usa Random o Heuristic evaluator
pip install torch  # Oppure usa evaluator 2 o 3
```

### L'AI è troppo lenta
- Riduci il numero di simulazioni (es. 10-20)
- Usa Random evaluator (più veloce ma più debole)

### L'AI è troppo debole
- Aumenta simulazioni (100-200)
- Usa Heuristic evaluator
- **Nota**: La rete neurale NON è addestrata, quindi è molto debole

## Prossimi Passi

Dopo aver testato il gioco, puoi:
1. **Addestrare la rete neurale** con self-play (STEP 6)
2. **Valutare su puzzle tattici** (mate in 2/3)
3. **Confrontare versioni diverse** dell'AI

---

**Buon divertimento! ♟️**
