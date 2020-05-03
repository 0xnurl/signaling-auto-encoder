import simulations
import extremity_game
import belief_game
from optparse import OptionParser


parser = OptionParser()

parser.add_option(
    "--name",
    default="signaling_game_simulation",
    dest="simulation_name",
    help="Simulation name",
)
parser.add_option(
    "--game",
    default="extremity",
    dest="game_type",
    help='Game type, "extremity" or "belief". Default: "extremity"',
)
parser.add_option(
    "-m",
    "--message",
    default="2",
    dest="message_sizes",
    help="Message dimension. Default: 2",
)
parser.add_option(
    "-o",
    "--object",
    default="5",
    dest="object_size",
    help="Object dimension. Default: 5",
)
parser.add_option(
    "--strict",
    default="true",
    dest="strict_context",
    help="Context strictness. Default: true",
)
parser.add_option(
    "--shared",
    default="true",
    dest="shared_context",
    help="Shared or non-shared context (displacement). Default: true",
)
parser.add_option(
    "--objects",
    default="10",
    dest="num_objects",
    help="Number of objects in context. Used only for non-strict contexts. Default: 10",
)
parser.add_option(
    "--batch-size",
    default="128",
    dest="mini_batch_size",
    help="Minibatch size. Default: 128",
)
parser.add_option(
    "-b",
    "--batches",
    default="5000",
    dest="num_batches",
    help="Number of batches. Default: 5000",
)
parser.add_option(
    "-t",
    "--trials",
    default=1,
    type="int",
    dest="num_trials",
    help="Number of trials (reproductions) for each game. Default: 1",
)
parser.add_option(
    "-p",
    "--processes",
    default=8,
    type="int",
    dest="num_processes",
    help="Number of games to run in parallel. Default: 8",
)

(options, args) = parser.parse_args()

iterable_args = (
    "message_sizes",
    "object_size",
    "strict_context",
    "shared_context",
    "num_objects",
    "mini_batch_size",
    "num_batches",
)

game_type_to_factory = {
    "extremity": extremity_game.make_extremity_game_simulation,
    "belief": belief_game.make_belief_update_simulation,
}

kwargs = {}
for option, option_val in options.__dict__.items():
    if option == "game_type":
        try:
            factory = game_type_to_factory[option_val]
            kwargs["simulation_factory"] = factory
        except KeyError:
            parser.print_help()
            parser.error("Invalid game type.")

    elif option in iterable_args:
        vals = []
        string_vals = option_val.split(",")
        for str_val in string_vals:
            str_val = str_val.strip().lower()
            if str_val == "true":
                vals.append(True)
            elif str_val == "false":
                vals.append(False)
            else:
                vals.append(int(str_val))
        kwargs[option] = tuple(vals)

    else:
        kwargs[option] = option_val


simulations.run_simulation_grid(**kwargs)
