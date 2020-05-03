import dataclasses
import itertools
import json
import logging
import multiprocessing
import pathlib
import pickle
import traceback
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import dataclasses_json
import game
import utils

utils.setup_logging()

ContextSizeType = Union[int, Tuple[int, int]]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Simulation:
    name: Text
    context_size: ContextSizeType
    object_size: int
    num_functions: int
    message_sizes: Tuple[int]
    target_function: Callable
    context_generator: Callable = None
    use_context: bool = True
    shared_context: bool = True
    nature_includes_function: bool = True
    shuffle_decoder_context: bool = False

    num_trials: int = 1
    mini_batch_size: int = 64
    num_batches: int = 5000

    epoch_nums: List[int] = dataclasses.field(default_factory=list)

    # {Message size -> [Trial x {Evaluation name -> values}]}
    evaluations: Dict[int, List[Dict[Text, Any]]] = dataclasses.field(
        default_factory=dict
    )


def _get_simulation_path(simulation_name: Text, subdir: Text = "") -> pathlib.Path:
    return (
        pathlib.Path("./simulations/").joinpath(subdir).joinpath(f"{simulation_name}/")
    )


def load_simulation(simulation_name: Text, subdir: Text = "") -> Simulation:
    name_split = simulation_name.split("/")
    if len(name_split) == 2:
        subdir, simulation_name = name_split
    return Simulation.from_json(
        _get_simulation_path(simulation_name, subdir)
        .joinpath(f"{simulation_name}.json")
        .read_text()
    )


def _save_simulation(simulation: Simulation):
    # Can't serialize functions.
    simulation_copy = dataclasses.replace(
        simulation, target_function=None, context_generator=None
    )
    simulation_path = pathlib.Path(f"./simulations/{simulation_copy.name}/")
    simulation_path.mkdir(parents=True, exist_ok=True)
    simulation_path.joinpath(f"{simulation_copy.name}.json").write_text(
        simulation_copy.to_json(indent=2)
    )


def _save_games(simulation: Simulation, games: Dict[int, List[game.Game]]):
    pickle.dump(
        games, _get_simulation_path(simulation.name).joinpath("games.pickle").open("wb")
    )


def load_games(simulation_name: Text) -> Dict[int, List[game.Game]]:
    return pickle.load(
        _get_simulation_path(simulation_name).joinpath("games.pickle").open("rb")
    )


def run_simulation(
    simulation: Simulation, visualize: bool = False, base_seed: int = 1000
) -> Dict[int, List[game.Game]]:
    logging.info(f"Running simulation: {simulation}")

    # {Message size -> Trial x Game}
    games: Dict[int, List[game.Game]] = {}

    # TODO decouple simulations from message size.

    for message_size in simulation.message_sizes:
        evaluations_per_trial: List[Dict[Text, Any]] = []
        game_per_trial: List[game.Game] = []

        for trial in range(simulation.num_trials):
            current_game: game.Game = game.Game(
                context_size=simulation.context_size,
                object_size=simulation.object_size,
                message_size=message_size,
                num_functions=simulation.num_functions,
                use_context=simulation.use_context,
                shared_context=simulation.shared_context,
                shuffle_decoder_context=simulation.shuffle_decoder_context,
                nature_includes_function=simulation.nature_includes_function,
                target_function=simulation.target_function,
                context_generator=simulation.context_generator,
                seed=base_seed + trial,
            )

            try:
                current_game.play(
                    num_batches=simulation.num_batches,
                    mini_batch_size=simulation.mini_batch_size,
                )
                if visualize:
                    current_game.visualize()

                evaluation_vals = current_game.get_evaluations()
                evaluations_per_trial.append(evaluation_vals)

                game_per_trial.append(current_game)
            except Exception as e:
                logging.error(
                    f"Simulation {simulation.name} crashed:\n{traceback.format_exc()}"
                )
                raise e

        simulation.evaluations[message_size] = evaluations_per_trial
        games[message_size] = game_per_trial

    simulation.epoch_nums = games[simulation.message_sizes[0]][0].epoch_nums
    _save_simulation(simulation)
    _save_games(simulation, games)
    return games


def run_simulation_grid(
    simulation_name: Text,
    simulation_factory: Callable,
    message_sizes: Tuple[int, ...],
    num_trials: int,
    num_processes: Optional[int] = None,
    **kwargs,
):
    keys, values = zip(*kwargs.items())
    simulations_grid = list(itertools.product(*values))

    logging.info(
        f"Running {len(simulations_grid) * len(message_sizes) * num_trials} total games"
    )

    simulations = []
    for grid_values in simulations_grid:
        simulation_kwargs = {k: v for k, v in zip(keys, grid_values)}

        current_simulation_name = f"{simulation_name}__" + utils.kwargs_to_str(
            simulation_kwargs
        )

        simulation = simulation_factory(
            name=current_simulation_name,
            message_sizes=message_sizes,
            num_trials=num_trials,
            **simulation_kwargs,
        )
        simulations.append(simulation)

    simulation_grid_kwargs = {"m": message_sizes, "trials": num_trials}
    simulation_grid_kwargs.update(kwargs)
    simulation_grid_name = f"{simulation_name}__" + utils.kwargs_to_str(
        simulation_grid_kwargs
    )
    json_path = pathlib.Path(f"./simulations/{simulation_grid_name}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            [
                dataclasses.replace(
                    simulation, target_function=None, context_generator=None
                ).to_dict()
                for simulation in simulations
            ],
            indent=2,
        )
    )

    if num_processes is not None:
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(run_simulation, simulations)
    else:
        for simulation in simulations:
            run_simulation(simulation)
