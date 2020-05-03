import logging
from typing import Callable, Tuple
import torch
import torch.nn.functional as F
from torch import nn

import simulations
import utils


class _BeliefUpdateNetwork(nn.Module):
    def __init__(
        self,
        context_size: int,
        object_size: int,
        num_functions: int,
        hidden_sizes: Tuple[int, ...],
        use_context: bool = True,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.use_context = use_context

        if self.use_context:
            input_size = context_size + num_functions
        else:
            input_size = num_functions
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, self.hidden_sizes[0])]
        )
        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], hidden_size))
        self.hidden_layers.append(nn.Linear(self.hidden_sizes[-1], object_size))

        logging.info("Update network:")
        logging.info(f"Context size: {context_size}")
        logging.info(f"Num functions: {num_functions}")
        logging.info(f"Hidden layers:\n{self.hidden_layers}")

    def forward(self, contexts, function_selectors):
        """`function_selectors` are one-hot vectors representing functions to be applied."""
        if self.use_context:
            input = torch.cat((contexts, function_selectors), dim=-1)
        else:
            input = function_selectors

        output = F.relu(self.hidden_layers[0](input))
        for hidden_layer in self.hidden_layers[1:]:
            output = F.relu(hidden_layer(output))
        return output


def _make_update_network_function(
    context_size: int,
    object_size: int,
    num_functions: int,
    update_network_hidden_sizes: Tuple[int, ...],
    use_context: bool,
) -> Callable:
    update_network = _BeliefUpdateNetwork(
        context_size,
        object_size,
        num_functions,
        update_network_hidden_sizes,
        use_context,
    )

    def func(contexts, function_selectors):
        with torch.no_grad():
            return update_network.forward(contexts, function_selectors)

    return func


def make_belief_update_simulation(
    context_size: int,
    object_size: int,
    num_functions: int,
    message_sizes: Tuple[int, ...],
    shared_context: bool,
    use_context: bool,
    **kwargs,
) -> simulations.Simulation:
    return simulations.Simulation(
        name=f"belief_update_game_c{context_size}_o{object_size}_f{num_functions}_m{utils.join_vals(message_sizes)}_sharedcontext{int(shared_context)}",
        context_size=context_size,
        target_function=_make_update_network_function(
            context_size, object_size, num_functions, (64,), use_context
        ),
        object_size=object_size,
        num_functions=num_functions,
        shared_context=shared_context,
        message_sizes=message_sizes,
        **kwargs,
    )
