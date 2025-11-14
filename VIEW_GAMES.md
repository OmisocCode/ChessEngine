# Come Visualizzare le Partite Self-Play

## Metodo 1: Demo Script (Consigliato)

### Genera e visualizza una singola partita
```bash
# Partita veloce (poche simulazioni)
python scripts/demo_selfplay_game.py --simulations 20

# Partita con più simulazioni (gioco migliore)
python scripts/demo_selfplay_game.py --simulations 50

# Salva partita in file PGN
python scripts/demo_selfplay_game.py --output my_game.pgn

# Usa modello addestrato
python scripts/demo_selfplay_game.py --model checkpoints/model_iter_10.pt --simulations 100
```

Lo script mostrerà:
- Mosse in tempo reale
- Tempo per ogni mossa
- Statistiche finali (risultato, mosse totali, tempo)
- **PGN completo** da copiare e visualizzare

### Cosa vedrai
```
======================================================================
DEMO SELF-PLAY GAME
======================================================================

MCTS simulations per move: 50
Using untrained network (random play)

Playing game...
(This may take a while depending on num_simulations)

Move 1: e4 (2.3s)
Move 2: e5 (2.1s)
Move 3: Nf3 (2.4s)
...

======================================================================
GAME SUMMARY
======================================================================
Result: 1-0
Total moves: 45
Total time: 98.5s
Average time per move: 2.2s

Game ended by: Checkmate

======================================================================
PGN FORMAT
======================================================================
[Event "Self-Play Training Game"]
[Site "ChessEngine AI"]
[Date "2025.11.14"]
[White "ChessEngine AI"]
[Black "ChessEngine AI"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 ...
```

## Metodo 2: Visualizza PGN Online

### Passo 1: Genera partita
```bash
python scripts/demo_selfplay_game.py --output game.pgn
```

### Passo 2: Apri il file PGN
Il file `game.pgn` contiene la partita in formato standard.

### Passo 3: Visualizza su sito web

#### Opzione A: Lichess (Consigliato)
1. Vai su **https://lichess.org/paste**
2. Copia tutto il contenuto di `game.pgn`
3. Incolla nella casella
4. Clicca "Import"
5. Vedrai la partita con scacchiera interattiva!

#### Opzione B: Chess.com
1. Vai su **https://www.chess.com/analysis**
2. Clicca "Load PGN"
3. Incolla il contenuto di `game.pgn`
4. Clicca "Load"

#### Opzione C: ChessTempo
1. Vai su **https://chesstempo.com/pgn-viewer/**
2. Incolla il PGN
3. Clicca "Show"

## Metodo 3: Salva Partite Durante Training

Se vuoi salvare automaticamente partite durante il training, modifica `scripts/train.py`:

### Modifica la funzione `run_iteration`:

Dopo la riga `examples, game_infos = worker.generate_games(...)`, aggiungi:

```python
# Salva prima partita in PGN
if iteration == 0:  # Solo prima iterazione per non riempire disco
    from src.training.save_games import save_first_game_as_pgn
    save_first_game_as_pgn(examples, game_infos[0], f"games/iter_{iteration}_game.pgn")
```

## Formato PGN Spiegato

Un file PGN contiene:

### Headers
```
[Event "Self-Play Training Game"]
[Site "ChessEngine AI"]
[Date "2025.11.14"]
[White "ChessEngine AI"]
[Black "ChessEngine AI"]
[Result "1-0"]
```

### Mosse
```
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 ...
```

- Numeri = numero mossa
- Prima mossa dopo numero = mossa Bianco
- Seconda mossa = mossa Nero
- Notazione SAN (Standard Algebraic Notation)

## Esempi d'Uso

### Confronta partite di modelli diversi
```bash
# Partita con modello untrained
python scripts/demo_selfplay_game.py --output untrained.pgn

# Partita con modello trained dopo 10 iterazioni
python scripts/demo_selfplay_game.py --model checkpoints/model_iter_10.pt --output trained_10.pgn

# Partita con modello trained dopo 20 iterazioni
python scripts/demo_selfplay_game.py --model checkpoints/model_iter_20.pt --output trained_20.pgn
```

Poi confronta le partite su lichess per vedere il miglioramento!

### Genera partita veloce per test
```bash
# Solo 10 simulazioni = molto veloce (~30 secondi)
python scripts/demo_selfplay_game.py --simulations 10 --output quick_test.pgn
```

### Genera partita "forte" con modello trained
```bash
# 200 simulazioni = gioco molto più forte ma più lento
python scripts/demo_selfplay_game.py --model checkpoints/model_iter_20.pt --simulations 200 --output strong_game.pgn
```

## Notazione Scacchistica

### Pezzi
- K = Re (King)
- Q = Donna (Queen)
- R = Torre (Rook)
- B = Alfiere (Bishop)
- N = Cavallo (Knight)
- (niente) = Pedone (Pawn)

### Mosse Esempio
- `e4` = pedone in e4
- `Nf3` = cavallo in f3
- `Bc4` = alfiere in c4
- `O-O` = arrocco corto
- `O-O-O` = arrocco lungo
- `e8=Q` = promozione a donna
- `Nxe5` = cavallo cattura in e5
- `Qh5+` = donna in h5 con scacco
- `Qxf7#` = donna cattura in f7 con scacco matto

## Troubleshooting

### "Script troppo lento"
Riduci le simulazioni:
```bash
python scripts/demo_selfplay_game.py --simulations 10
```

### "Voglio vedere molte partite"
Usa un loop bash:
```bash
for i in {1..5}; do
    python scripts/demo_selfplay_game.py --simulations 20 --output "game_$i.pgn"
done
```

Genererà 5 partite: `game_1.pgn`, `game_2.pgn`, ..., `game_5.pgn`

### "Errore importando PGN"
Assicurati di copiare **tutto** il contenuto, inclusi gli headers `[Event...]` fino all'ultima mossa.

## File PGN di Esempio

Contenuto tipico di un file PGN:

```
[Event "Self-Play Training Game"]
[Site "ChessEngine AI"]
[Date "2025.11.14"]
[White "ChessEngine AI"]
[Black "ChessEngine AI"]
[Result "1/2-1/2"]
[Annotator "ChessEngine Self-Play"]
[MCTSSimulations "50"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+
7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 11. O-O O-O
12. Rfe1 c6 13. a4 Qd6 14. Ne4 Qf4 15. Neg5 h6 16. Ne4 Qd6
17. Neg5 Qf4 18. Ne4 Qd6 1/2-1/2
```

Puoi copiare questo esempio e testarlo su lichess.org/paste!

---

✅ Ora puoi visualizzare tutte le partite giocate dall'AI!
