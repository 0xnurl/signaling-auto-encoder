# Signaling Auto-Encoder

Code to reproduce and extend the experiments in "On the spontaneous emergence of discrete and compositional signals" by Nur Lan, Emmanuel Chemla, and Shane Steinert-Threlkeld.

## Setup

`pip install -r requirements.txt`

## Running a game

```
Usage: run_simulation.py [options]

Options:
  --name                Simulation name
  --game                Game type, "extremity" or "belief". Default: "extremity"
  -m, --message         Message dimension. Default: 2
  -o, --object          Object dimension. Default: 5
  --strict              Context strictness. Default: true
  --shared              Shared or non-shared context (displacement). Default: true
  --objects             Number of objects in context. Used only for non-strict
                        contexts. Default: 10
  --batch-size          Minibatch size. Default: 128
  -b, --batches         Number of batches. Default: 5000
  -t, --trials          Number of trials (reproductions) for each game.
                        Default: 1
  -p, --processes       Number of games to run in parallel. Default: 8
```

Example to reproduce the experiments from the paper

`python run_simulation.py --name my_extremity_game --game extremity -m 2 -o 5 --strict true --shared false --batch-size 128 --batches 5000 --trials 20`


### Running a simulation grid

All parameters except `name`, `game`, `trials`, and `processes` can be passed as a comma-separated list, e.g.:

`python run_simulation.py --name my_extremity_game --game extremity -m 1,2,5 -o 2,5,7 --strict true,false --shared true,false --batch-size 128,256 --batches 1000,5000`

This will run 144 games in total for all parameter combinations (repeated `--trials` times).

Results are saved to `simulations`.