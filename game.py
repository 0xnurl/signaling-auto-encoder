import collections
import itertools
import json
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Text, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import cluster, metrics
from torch import optim

import utils


class Game(nn.Module):
    def __init__(
        self,
        context_size: int,
        object_size: int,
        message_size: int,
        num_functions: int,
        target_function: Callable,
        use_context: bool = True,
        shared_context: bool = True,
        shuffle_decoder_context: bool = False,
        nature_includes_function: bool = True,
        context_generator: Optional[Callable] = None,
        loss_every: int = 1,
        num_exemplars: int = 100,
        encoder_hidden_sizes: Tuple[int, ...] = (64, 64),
        decoder_hidden_sizes: Tuple[int, ...] = (64, 64),
        seed: int = 100,
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.context_size = context_size
        self.object_size = object_size
        self.message_size = message_size
        self.num_functions = num_functions
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.use_context = use_context
        self.shared_context = shared_context
        self.shuffle_decoder_context = shuffle_decoder_context
        self.nature_includes_function = nature_includes_function
        self.context_generator = context_generator
        self.target_function = target_function
        self.loss_every = loss_every
        self.num_exemplars = num_exemplars
        self.seed = seed

        self.criterion = nn.MSELoss()
        self.epoch_nums: List[int] = []
        self.loss_per_epoch: List[float] = []

        self.clustering_model = None
        self.cluster_label_to_func_idx: Dict[int, int] = {}

        if isinstance(self.context_size, tuple):
            self.flat_context_size = utils.reduce_prod(self.context_size)
        elif isinstance(self.context_size, int):
            self.flat_context_size = self.context_size
        else:
            raise ValueError(f"context_size must be either a tuple or int")

        encoder_input_size = self._get_encoder_input_size()
        if self.use_context:
            decoder_input_size = self.message_size + self.flat_context_size
        else:
            decoder_input_size = self.message_size

        encoder_layer_dimensions = [(encoder_input_size, self.encoder_hidden_sizes[0])]
        decoder_layer_dimensions = [(decoder_input_size, self.decoder_hidden_sizes[0])]

        for i, hidden_size in enumerate(self.encoder_hidden_sizes[1:]):
            hidden_shape = (self.encoder_hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)

        for i, hidden_size in enumerate(self.decoder_hidden_sizes[1:]):
            hidden_shape = (self.decoder_hidden_sizes[i], hidden_size)
            decoder_layer_dimensions.append(hidden_shape)

        encoder_layer_dimensions.append(
            (self.encoder_hidden_sizes[-1], self.message_size)
        )
        decoder_layer_dimensions.append(
            (self.decoder_hidden_sizes[-1], self.object_size)
        )

        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )
        self.decoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in decoder_layer_dimensions]
        )

        logging.info("Game details:")
        logging.info(f"Seed: {seed}")
        logging.info(
            f"\nContext size: {context_size}\nObject size: {object_size}\nMessage size: {message_size}\nNumber of functions: {num_functions}"
        )
        logging.info(f"Use context: {use_context}")
        logging.info(f"Encoder layers:\n{self.encoder_hidden_layers}")
        logging.info(f"Decoder layers:\n{self.decoder_hidden_layers}")

    def play(self, num_batches, mini_batch_size):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for batch_num in range(num_batches):
            function_selectors = self._generate_function_selectors(
                mini_batch_size, random=True
            )
            contexts = self._generate_contexts(mini_batch_size)
            decoder_contexts = self._get_decoder_context(mini_batch_size, contexts)

            optimizer.zero_grad()
            loss = self._loss(contexts, function_selectors, decoder_contexts)
            loss.backward()
            optimizer.step()

            if batch_num % self.loss_every == 0 or batch_num == (num_batches - 1):
                self._log_epoch_loss(batch_num, loss.item())

            if batch_num % 100 == 0:
                logging.info(
                    f"Batch {batch_num + (1 if batch_num == 0 else 0)} loss:\t{self.loss_per_epoch[-1]:.2e}"
                )

    def _encoder_forward_pass(self, context, function_selector):
        encoder_input = self._get_input(context, function_selector)
        message = encoder_input
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            message = F.relu(hidden_layer(message))
        message = self.encoder_hidden_layers[-1](message)

        return message

    def _decoder_forward_pass(self, message, context):
        if self.use_context:
            context_flattened = utils.batch_flatten(context)
            decoder_input = torch.cat((message, context_flattened), dim=1)
        else:
            decoder_input = message

        prediction = decoder_input
        for hidden_layer in self.decoder_hidden_layers[:-1]:
            prediction = F.relu(hidden_layer(prediction))
        prediction = self.decoder_hidden_layers[-1](prediction)

        return prediction

    def _forward(self, context, function_selector, decoder_context):
        message = self._encoder_forward_pass(context, function_selector)
        prediction = self._decoder_forward_pass(message, decoder_context)
        return prediction

    def _predict(self, context, function_selector, decoder_context):
        with torch.no_grad():
            return self._forward(context, function_selector, decoder_context)

    def _predict_by_message(self, message, decoder_context):
        with torch.no_grad():
            return self._decoder_forward_pass(message, decoder_context)

    def _target(self, context, function_selector):
        return self.target_function(context, function_selector)

    def _message(self, context, function_selector):
        with torch.no_grad():
            return self._encoder_forward_pass(context, function_selector)

    def _loss(self, context, function_selectors, decoder_context):
        target = self._target(decoder_context, function_selectors)
        prediction = self._forward(context, function_selectors, decoder_context)
        return self.criterion(prediction, target)

    def _get_encoder_input_size(self):
        parts = []
        if self.use_context:
            parts.append(self.flat_context_size)
        if self.nature_includes_function:
            parts.append(self.num_functions)
        else:
            parts.append(self.object_size)
        return sum(parts)

    def _get_input(self, contexts: torch.Tensor, function_selectors: torch.Tensor):
        parts = []
        if self.use_context:
            contexts_flat = utils.batch_flatten(contexts)
            parts.append(contexts_flat)
        if self.nature_includes_function:
            parts.append(function_selectors)
        else:
            objects = self._target(contexts, function_selectors)
            parts.append(objects)
        return torch.cat(parts, dim=1)

    def _generate_contexts(self, batch_size):
        if isinstance(self.context_size, int):
            context_shape = (self.context_size,)
        else:
            context_shape = self.context_size

        if self.context_generator is None:
            return torch.randn(batch_size, *context_shape)
        else:
            return self.context_generator(batch_size, context_shape)

    def _get_decoder_context(self, batch_size, encoder_context):
        if self.shared_context:
            decoder_context = encoder_context
        else:
            decoder_context = self._generate_contexts(batch_size)

        if self.shuffle_decoder_context:
            decoder_context = decoder_context[
                :, torch.randperm(decoder_context.shape[1]), :
            ]
        return decoder_context

    def _generate_function_selectors(self, batch_size, random=False):
        """Generate `batch_size` one-hot vectors of dimension `num_functions`."""
        if random:
            function_idxs = torch.randint(self.num_functions, size=(batch_size,))
        else:
            function_idxs = torch.arange(batch_size) % self.num_functions
        return torch.nn.functional.one_hot(
            function_idxs, num_classes=self.num_functions
        ).float()

    def _generate_funcs_contexts_messages(self, num_exemplars, random=False):
        batch_size = num_exemplars * self.num_functions
        encoder_contexts = self._generate_contexts(batch_size)
        decoder_contexts = self._get_decoder_context(batch_size, encoder_contexts)
        function_selectors = self._generate_function_selectors(
            batch_size, random=random
        )
        messages = self._message(encoder_contexts, function_selectors)
        return function_selectors, encoder_contexts, decoder_contexts, messages

    def _log_epoch_loss(self, epoch, loss):
        self.loss_per_epoch.append(loss)
        self.epoch_nums.append(epoch)

    def visualize(self):
        self._plot_messages_information()
        self._run_unsupervised_clustering(visualize=True)

    # Evaluations

    def get_evaluations(self) -> Dict[Text, Any]:
        self._run_unsupervised_clustering()

        evaluation_funcs = {
            "training_losses": lambda: self.loss_per_epoch,
            "object_prediction_accuracy": self._evaluate_encoder_decoder_prediction_accuracy,
            "categorical_perception_accuracies": self._evaluate_categorical_perception,
            # Unsupervised clustering
            "detected_num_clusters": self._detect_num_clusters,
            "object_prediction_by_cluster_loss": self._evaluate_object_prediction_by_cluster,
            "clusterization_f_score": self._evaluate_clusterization_f_score,
            "average_cluster_message_perception": self._evaluate_average_cluster_message_perception,
            # Compositionality
            "addition_compositionality_loss": self._evaluate_addition_compositionality,
            "analogy_compositionality_loss": self._evaluate_analogy_compositionality_network,
            "compositionality_loss": self._evaluate_compositionality_network,
        }

        evaluation_results = {
            eval_name: f() for eval_name, f in evaluation_funcs.items()
        }

        # Collect nested dict values at top level.
        keys = tuple(evaluation_results.keys())
        for k in keys:
            if isinstance(evaluation_results[k], dict):
                for nested_k, nested_val in evaluation_results[k].items():
                    evaluation_results[nested_k] = nested_val
                del evaluation_results[k]

        elements_to_predict_from_messages = (
            "functions",
            "min_max",
            "dimension",
            "sanity",
            "object_by_context",
            "object_by_decoder_context",
            "context",
            "decoder_context",
        )
        for element in elements_to_predict_from_messages:
            evaluation_results[
                f"{element}_from_messages"
            ] = self.predict_element_by_messages(element)

        logging.info(f"Evaluations:\n{json.dumps(evaluation_results, indent=1)}")

        return evaluation_results

    def _evaluate_categorical_perception(
        self, num_contexts: int = 100, num_intermediate_points: int = 50_000
    ):
        messages_per_cluster = num_contexts // 2

        encoder_contexts = self._generate_contexts(batch_size=num_contexts)
        decoder_contexts = self._get_decoder_context(
            batch_size=num_contexts, encoder_context=encoder_contexts
        )
        decoder_contexts_1 = decoder_contexts[:messages_per_cluster]
        decoder_contexts_2 = decoder_contexts[messages_per_cluster:]

        function_selectors_idxs = torch.zeros(num_contexts)

        function_selectors_idxs[messages_per_cluster:] = 1
        function_selectors = torch.nn.functional.one_hot(
            function_selectors_idxs.long(), num_classes=self.num_functions
        ).float()

        messages = self._message(encoder_contexts, function_selectors)

        messages_1 = messages[:messages_per_cluster]
        messages_2 = messages[messages_per_cluster:]

        target_objects = self._target(decoder_contexts, function_selectors)

        target_objects_1 = target_objects[:messages_per_cluster]
        target_objects_2 = target_objects[messages_per_cluster:]

        target_1_accuracies = []
        target_2_accuracies = []

        for t in np.linspace(0, 1, num_intermediate_points):
            messages_shifted = (t * messages_1) + ((1 - t) * messages_2)

            predicted_objects_1 = self._predict_by_message(
                messages_shifted, decoder_contexts[:messages_per_cluster]
            )
            predicted_objects_2 = self._predict_by_message(
                messages_shifted, decoder_contexts[messages_per_cluster:]
            )

            acc_1 = self._evaluate_object_prediction_accuracy(
                decoder_contexts_1, predicted_objects_1, target_objects_1
            )

            acc_2 = self._evaluate_object_prediction_accuracy(
                decoder_contexts_2, predicted_objects_2, target_objects_2
            )

            target_1_accuracies.append(acc_1)
            target_2_accuracies.append(acc_2)

        return {
            "categorical_perception_target_1_accuracies": target_1_accuracies,
            "categorical_perception_target_2_accuracies": target_2_accuracies,
        }

    def _detect_num_clusters(self):
        _, _, _, messages = self._generate_funcs_contexts_messages(1000)
        dbscan = cluster.DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(messages)
        num_predicted_clusters = len(set(dbscan.labels_))
        logging.info(f"Number of predicted clusters: {num_predicted_clusters}")
        return num_predicted_clusters

    @staticmethod
    def _evaluate_object_prediction_accuracy(
        contexts: torch.Tensor,
        predicted_objects: torch.Tensor,
        target_objects: torch.Tensor,
    ) -> float:
        if len(contexts.shape) != 3:
            logging.info(f"Object prediction accuracy only valid for extremity game.")
            return 0.0

        batch_size = contexts.shape[0]
        correct = 0
        for b in range(batch_size):
            context = contexts[b]
            predicted_obj = predicted_objects[b].unsqueeze(dim=0)
            mse_per_obj = utils.batch_mse(context, predicted_obj)
            closest_obj_idx = torch.argmin(mse_per_obj)
            closest_obj = context[closest_obj_idx]
            if torch.all(closest_obj == target_objects[b]):
                correct += 1

        accuracy = correct / batch_size
        logging.info(
            f"Object prediction accuracy: {correct}/{batch_size} = {accuracy:.2f}"
        )
        return accuracy

    def _evaluate_encoder_decoder_prediction_accuracy(self):
        (
            function_selectors,
            encoder_contexts,
            decoder_contexts,
            _,
        ) = self._generate_funcs_contexts_messages(self.num_exemplars, random=False)
        predicted_objects = self._predict(
            encoder_contexts, function_selectors, decoder_contexts
        )
        target_objects = self._target(decoder_contexts, function_selectors)
        return self._evaluate_object_prediction_accuracy(
            decoder_contexts, predicted_objects, target_objects
        )

    def _plot_messages_information(
        self, visualize_targets: bool = False, visualize_latent_space: bool = False
    ):
        with torch.no_grad():
            (
                func_selectors,
                encoder_contexts,
                decoder_contexts,
                messages,
            ) = self._generate_funcs_contexts_messages(self.num_exemplars, random=False)

            message_masks = []
            message_labels = []
            for func_idx in range(self.num_functions):
                message_masks.append(
                    [
                        i * self.num_functions + func_idx
                        for i in range(self.num_exemplars)
                    ]
                )
                message_labels.append(f"F{func_idx}")

            title_information_row = f"M={self.message_size}, O={self.object_size}, C={self.context_size}, F={self.num_functions}"

            utils.plot_raw(
                messages.numpy(),
                message_masks,
                message_labels,
                f"Messages\n{title_information_row}",
            )

            if visualize_targets:
                targets = self._target(encoder_contexts, func_selectors)
                utils.plot_raw(
                    targets.numpy(),
                    message_masks,
                    message_labels,
                    f"Targets\n{title_information_row}",
                )

            if visualize_latent_space:
                # Plot latent encoder space
                encoder_context_flat = utils.batch_flatten(encoder_contexts)
                encoder_input = torch.cat((encoder_context_flat, func_selectors), dim=1)

                latent_messages_level_1 = F.relu(
                    self.encoder_hidden_layers[0](encoder_input)
                )

                latent_messages_level_2 = F.relu(
                    self.encoder_hidden_layers[1](latent_messages_level_1)
                )

                utils.plot_raw(
                    latent_messages_level_1.numpy(),
                    message_masks,
                    message_labels,
                    f"Encoder latent level 1 -- ReLu(W_e1(input))",
                )

                utils.plot_raw(
                    latent_messages_level_2.numpy(),
                    message_masks,
                    message_labels,
                    f"Encoder latent level 2 -- ReLu(W_e2(ReLu(W_e1(input))))",
                )

                # Plot latent decoder space
                decoder_context_flat = utils.batch_flatten(decoder_contexts)
                decoder_input = torch.cat((messages, decoder_context_flat), dim=1)
                latent_decoder_level_1 = F.relu(
                    self.decoder_hidden_layers[0](decoder_input)
                )
                latent_decoder_level_2 = F.relu(
                    self.decoder_hidden_layers[1](latent_decoder_level_1)
                )

                utils.plot_raw(
                    latent_decoder_level_1.numpy(),
                    message_masks,
                    message_labels,
                    f"Decoder latent level 1 -- ReLu(W_d1(messages+context))",
                )

                utils.plot_raw(
                    latent_decoder_level_2.numpy(),
                    message_masks,
                    message_labels,
                    f"Decoder latent level 2 -- ReLu(W_d2(ReLu(W_d1(messages+context))))",
                )

    def predict_element_by_messages(self, element_to_predict: Text) -> float:
        logging.info(f"Predicting {element_to_predict} from messages.")

        (
            func_selectors,
            contexts,
            _,
            messages,
        ) = self._generate_funcs_contexts_messages(self.num_exemplars, random=False)
        batch_size = func_selectors.shape[0]

        train_test_ratio = 0.7
        num_train_samples = math.ceil(batch_size * train_test_ratio)

        ACCURACY_PREDICTIONS = ("functions", "min_max", "dimension", "sanity")

        if element_to_predict in ACCURACY_PREDICTIONS:
            # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
            loss_func = torch.nn.NLLLoss()
        else:
            loss_func = torch.nn.MSELoss()

        if element_to_predict == "functions":
            elements = func_selectors
        elif element_to_predict == "min_max":
            if len(contexts.shape) != 3:
                # Requires extremity game context.
                return 0.0
            elements = torch.nn.functional.one_hot(
                func_selectors.argmax(dim=1) % 2, num_classes=2
            )
        elif element_to_predict == "dimension":
            if len(contexts.shape) != 3:
                # Requires extremity game context.
                return 0.0
            num_dimensions = contexts.shape[2]
            elements = torch.nn.functional.one_hot(
                func_selectors.argmax(dim=1) // 2, num_classes=num_dimensions,
            )
        elif element_to_predict == "sanity":
            # Test prediction accuracy of random data. Should be at chance level.
            elements = torch.nn.functional.one_hot(
                torch.randint(0, 2, (batch_size,)), num_classes=2,
            )
        elif element_to_predict == "object_by_context":
            elements = self.target_function(contexts, func_selectors)
        elif element_to_predict == "object_by_decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            decoder_contexts = self._generate_contexts(batch_size)
            elements = self.target_function(decoder_contexts, func_selectors)
        elif element_to_predict == "context":
            elements = utils.batch_flatten(contexts)
        elif element_to_predict == "decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            elements = utils.batch_flatten(self._generate_contexts(batch_size))
        else:
            raise ValueError("Invalid element to predict")

        train_target, test_target = (
            elements[:num_train_samples],
            elements[num_train_samples:],
        )
        train_messages, test_messages = (
            messages[:num_train_samples],
            messages[num_train_samples:],
        )

        classifier_hidden_size = 32
        layers = [
            torch.nn.Linear(self.message_size, classifier_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_hidden_size, test_target.shape[-1]),
        ]

        if element_to_predict in ACCURACY_PREDICTIONS:
            layers.append(torch.nn.LogSoftmax(dim=1))

        model = torch.nn.Sequential(*layers)
        logging.info(f"Prediction network layers:\n{layers}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1000
        for epoch in range(num_epochs):
            y_pred = model(train_messages)
            if element_to_predict in ACCURACY_PREDICTIONS:
                current_train_target = train_target.argmax(dim=1)
            else:
                current_train_target = train_target
            loss = loss_func(y_pred, current_train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % 100 == 0:
                logging.info(
                    f"Epoch {epoch + (1 if epoch == 0 else 0)}:\t{loss.item():.2e}"
                )

        with torch.no_grad():
            test_predicted = model(test_messages)

        if element_to_predict in ACCURACY_PREDICTIONS:
            accuracy = metrics.accuracy_score(
                test_target.argmax(dim=1).numpy(), test_predicted.argmax(dim=1).numpy()
            )
            result = accuracy
        else:
            result = loss_func(test_predicted, test_target).item()
        logging.info(f"Prediction result for {element_to_predict}: {result}")
        return result

    def _evaluate_addition_compositionality(self):
        message_losses = []
        message_cluster_accuracies = []
        prediction_output_losses = []
        prediction_output_accuracies = []

        for d1, d2 in itertools.permutations(range(self.object_size), 2):
            (
                function_selectors,
                encoder_contexts,
                decoder_contexts,
                messages,
            ) = self._generate_funcs_contexts_messages(self.num_exemplars)

            function_idxs = function_selectors.argmax(dim=1)
            argmin_mask = function_idxs % 2 == 0
            argmax_mask = function_idxs % 2 == 1
            d1_mask = function_idxs // 2 == d1
            d2_mask = function_idxs // 2 == d2

            d1_argmin_messages = messages[d1_mask * argmin_mask]
            d1_argmax_messages = messages[d1_mask * argmax_mask]
            d2_argmin_messages = messages[d2_mask * argmin_mask]

            target_messages_mask = d2_mask * argmax_mask
            target_messages = messages[target_messages_mask]
            target_function_idxs = function_idxs[target_messages_mask]

            inferred_messages = (
                d1_argmax_messages - d1_argmin_messages + d2_argmin_messages
            )

            (
                messages_loss,
                message_cluster_accuracy,
                prediction_loss,
                prediction_accuracy,
            ) = self._evaluate_inferred_messages(
                target_messages,
                inferred_messages,
                encoder_contexts[target_messages_mask],
                decoder_contexts[target_messages_mask],
                target_function_idxs,
            )

            logging.info(
                f"Addition compositionality messages loss for d{d1} <-> d{d2}: {messages_loss}"
            )
            logging.info(
                f"Addition compositionality message cluster accuracy for d{d1} <-> d{d2}: {message_cluster_accuracy}"
            )

            message_losses.append(messages_loss)
            message_cluster_accuracies.append(message_cluster_accuracy)

            # Test perception quality

            predicted_output_by_inferred_messages = self._predict_by_message(
                inferred_messages, decoder_contexts[target_messages_mask]
            )
            target_output = self._target(
                decoder_contexts[target_messages_mask],
                function_selectors[target_messages_mask],
            )
            prediction_loss = torch.nn.MSELoss()(
                predicted_output_by_inferred_messages, target_output
            ).item()
            prediction_accuracy = self._evaluate_object_prediction_accuracy(
                decoder_contexts[target_messages_mask],
                predicted_output_by_inferred_messages,
                target_output,
            )

            logging.info(
                f"Addition compositionality output loss for d{d1} <-> d{d2}: {prediction_loss}"
            )
            logging.info(
                f"Addition compositionality output accuracy for d{d1} <-> d{d2}: {prediction_accuracy}"
            )
            prediction_output_losses.append(prediction_loss)
            prediction_output_accuracies.append(prediction_accuracy)

        messages_mean_loss = np.mean(message_losses)
        message_clusters_mean_acc = np.mean(message_cluster_accuracies)

        prediction_mean_loss = np.mean(prediction_output_losses)
        prediction_mean_acc = np.mean(prediction_output_accuracies)

        logging.info(
            f"Addition compositionality mean messages loss: {messages_mean_loss}"
        )
        logging.info(
            f"Addition compositionality mean message cluster accuracy: {message_clusters_mean_acc}"
        )

        logging.info(
            f"Addition compositionality mean prediction loss: {prediction_mean_loss}"
        )
        logging.info(
            f"Addition compositionality mean prediction accuracy: {prediction_mean_acc}"
        )

        return {
            "addition_compositionality_mean_message_loss": messages_mean_loss,
            "addition_compositionality_mean_message_cluster_accuracy": message_clusters_mean_acc,
            "addition_compositionality_mean_prediction_loss": prediction_mean_loss,
            "addition_compositionality_mean_prediction_accuracy": prediction_mean_acc,
        }

    def _evaluate_analogy_compositionality_network(self):
        message_losses = []
        message_cluster_accuracies = []
        production_output_losses = []
        production_output_accuracies = []

        for p in range(self.object_size):
            (
                test_loss,
                cluster_accuracy,
                prediction_loss,
                prediction_accuracy,
            ) = self._run_analogy_compositionality_network(taken_out_param=p)
            message_losses.append(test_loss)
            message_cluster_accuracies.append(cluster_accuracy)
            production_output_losses.append(prediction_loss)
            production_output_accuracies.append(prediction_accuracy)

        mean_message_loss = np.mean(message_losses)
        mean_message_acc = np.mean(message_cluster_accuracies)
        mean_prediction_loss = np.mean(production_output_losses)
        mean_prediction_acc = np.mean(production_output_accuracies)

        logging.info(f"Mean analogy network message loss: {mean_message_loss}")
        logging.info(f"Mean analogy network message accuracy: {mean_message_acc}")
        logging.info(f"Mean analogy network prediction loss: {mean_prediction_loss}")
        logging.info(f"Mean analogy network prediction accuracy: {mean_prediction_acc}")

        return {
            f"analogy_compositionality_net_message_mean_loss": mean_message_loss,
            f"analogy_compositionality_net_message_cluster_mean_accuracy": mean_message_acc,
            f"analogy_compositionality_net_prediction_mean_loss": mean_prediction_loss,
            f"analogy_compositionality_net_prediction_mean_accuracy": mean_prediction_acc,
        }

    def _run_analogy_compositionality_network(
        self, taken_out_param: int, visualize: bool = False
    ):
        train_input_messages = []
        train_target_messages = []

        test_input_messages = []
        test_target_messages = []
        test_function_idxs = []
        test_encoder_contexts = []
        test_decoder_contexts = []

        for d1, d2 in itertools.permutations(range(self.object_size), 2):
            (
                function_selectors,
                encoder_contexts,
                decoder_contexts,
                messages,
            ) = self._generate_funcs_contexts_messages(self.num_exemplars)

            function_idxs = function_selectors.argmax(dim=1)
            argmin_mask = function_idxs % 2 == 0
            argmax_mask = function_idxs % 2 == 1
            d1_mask = function_idxs // 2 == d1
            d2_mask = function_idxs // 2 == d2

            d1_argmin_messages = messages[d1_mask * argmin_mask]
            d1_argmax_messages = messages[d1_mask * argmax_mask]
            d2_argmin_messages = messages[d2_mask * argmin_mask]

            target_messages_mask = d2_mask * argmax_mask
            d2_argmax_messages = messages[target_messages_mask]

            # Train to predict [argmax_d2] from [d1_argmin_messages, d1_argmax_messages, d2_argmin_messages].

            if taken_out_param == d2:
                inputs = test_input_messages
                targets = test_target_messages
                test_function_idxs.append(function_idxs[target_messages_mask])
                test_encoder_contexts.append(encoder_contexts[target_messages_mask])
                test_decoder_contexts.append(decoder_contexts[target_messages_mask])
            else:
                inputs = train_input_messages
                targets = train_target_messages

            inputs.append(
                torch.cat(
                    [d1_argmin_messages, d1_argmax_messages, d2_argmin_messages], dim=1
                )
            )
            targets.append(d2_argmax_messages)

        train_input_messages = torch.cat(train_input_messages)
        train_target_messages = torch.cat(train_target_messages)
        test_input_messages = torch.cat(test_input_messages)
        test_target_messages = torch.cat(test_target_messages)
        test_encoder_contexts = torch.cat(test_encoder_contexts)
        test_decoder_contexts = torch.cat(test_decoder_contexts)
        test_function_idxs = torch.cat(test_function_idxs)

        hidden_size = 64
        num_epochs = 1000
        mini_batch_size = 64

        layers = [
            torch.nn.Linear(self.message_size * 3, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.message_size),
        ]

        model = torch.nn.Sequential(*layers)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for inputs_batch, targets_batch in zip(
                train_input_messages.split(mini_batch_size),
                train_target_messages.split(mini_batch_size),
            ):
                pred = model(inputs_batch)
                optimizer.zero_grad()
                loss = loss_func(pred, targets_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}:\t{loss.item():.2e}")

        with torch.no_grad():
            inferred_messages = model(test_input_messages)

        # Visualize network predictions vs targets

        if visualize:
            num_test_messages = inferred_messages.shape[0]
            mask1 = np.array([True] * num_test_messages + [False] * num_test_messages)
            mask2 = mask1 ^ True
            utils.plot_raw(
                data=torch.cat([test_target_messages, inferred_messages], dim=0),
                masks=[mask1, mask2],
                labels=["Target messages", "Inferred messages"],
                title="Analogy network predictions vs. targets",
            )

        (
            messages_loss,
            message_cluster_accuracy,
            prediction_loss,
            prediction_accuracy,
        ) = self._evaluate_inferred_messages(
            test_target_messages,
            inferred_messages,
            test_encoder_contexts,
            test_decoder_contexts,
            test_function_idxs,
        )
        logging.info(
            f"Analogy compositionality messages loss for taken-out param {taken_out_param}: {messages_loss}"
        )
        logging.info(
            f"Analogy compositionality network accuracy for taken-out param {taken_out_param}: {message_cluster_accuracy}"
        )
        logging.info(
            f"Analogy compositionality output loss for taken-out {taken_out_param}: {prediction_loss}"
        )
        logging.info(
            f"Analogy compositionality output accuracy for taken-out {taken_out_param}: {prediction_accuracy}"
        )
        return (
            messages_loss,
            message_cluster_accuracy,
            prediction_loss,
            prediction_accuracy,
        )

    def _evaluate_inferred_messages(
        self,
        target_messages: torch.Tensor,
        inferred_messages: torch.Tensor,
        encoder_contexts: torch.Tensor,
        decoder_contexts: torch.Tensor,
        target_function_idxs: torch.Tensor,
    ):
        # Evaluate production

        loss_func = torch.nn.MSELoss()
        messages_loss = loss_func(inferred_messages, target_messages).item()

        predicted_clusters = self.clustering_model.predict(inferred_messages)
        predicted_function_idxs_by_clusters = np.array(
            [self.cluster_label_to_func_idx[c] for c in predicted_clusters]
        )
        message_cluster_accuracy = (
            predicted_function_idxs_by_clusters == target_function_idxs.numpy()
        ).mean()

        # Evaluate perception

        predicted_output_by_inferred_messages = self._predict_by_message(
            inferred_messages, decoder_contexts
        )
        target_output = self._target(
            decoder_contexts,
            torch.nn.functional.one_hot(
                target_function_idxs, num_classes=self.num_functions
            ).float(),
        )

        prediction_loss = loss_func(
            predicted_output_by_inferred_messages, target_output
        ).item()
        prediction_accuracy = self._evaluate_object_prediction_accuracy(
            decoder_contexts, predicted_output_by_inferred_messages, target_output,
        )

        return (
            messages_loss,
            message_cluster_accuracy,
            prediction_loss,
            prediction_accuracy,
        )

    def _evaluate_compositionality_network(self):
        message_losses = []
        message_cluster_accuracies = []
        production_output_losses = []
        production_output_accuracies = []

        for p in range(self.object_size):
            (
                test_loss,
                cluster_accuracy,
                prediction_loss,
                prediction_accuracy,
            ) = self._run_compositionality_network(taken_out_param=p)
            message_losses.append(test_loss)
            message_cluster_accuracies.append(cluster_accuracy)
            production_output_losses.append(prediction_loss)
            production_output_accuracies.append(prediction_accuracy)

        mean_message_loss = np.mean(message_losses)
        mean_message_acc = np.mean(message_cluster_accuracies)
        mean_prediction_loss = np.mean(production_output_losses)
        mean_prediction_acc = np.mean(production_output_accuracies)

        logging.info(f"Mean compositionality network message loss: {mean_message_loss}")
        logging.info(
            f"Mean compositionality network message accuracy: {mean_message_acc}"
        )
        logging.info(
            f"Mean compositionality network prediction loss: {mean_prediction_loss}"
        )
        logging.info(
            f"Mean compositionality network prediction accuracy: {mean_prediction_acc}"
        )

        return {
            f"compositionality_net_message_mean_loss": mean_message_loss,
            f"compositionality_net_message_cluster_mean_accuracy": mean_message_acc,
            f"compositionality_net_prediction_mean_loss": mean_prediction_loss,
            f"compositionality_net_prediction_mean_accuracy": mean_prediction_acc,
        }

    def _run_compositionality_network(self, taken_out_param: int):
        """Train to predict argmin_d from argmax_d using all d's except one. """
        (
            function_selectors,
            encoder_contexts,
            decoder_contexts,
            messages,
        ) = self._generate_funcs_contexts_messages(self.num_exemplars)

        function_idxs = function_selectors.argmax(dim=1)
        argmin_mask = function_idxs % 2 == 0
        argmax_mask = function_idxs % 2 == 1

        taken_out_mask = function_idxs // 2 == taken_out_param
        training_mask = taken_out_mask ^ True
        training_input_mask = training_mask * argmax_mask
        training_target_mask = training_mask * argmin_mask
        test_target_mask = taken_out_mask * argmin_mask

        training_input_messages = messages[training_input_mask]
        training_target_messages = messages[training_target_mask]

        test_input_messages = messages[taken_out_mask * argmax_mask]
        test_target_messages = messages[taken_out_mask * argmin_mask]

        test_encoder_contexts = encoder_contexts[test_target_mask]
        test_decoder_contexts = decoder_contexts[test_target_mask]
        test_function_idxs = function_idxs[test_target_mask]

        hidden_size = 64
        num_epochs = 1000
        mini_batch_size = 64

        layers = [
            torch.nn.Linear(self.message_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.message_size),
        ]

        model = torch.nn.Sequential(*layers)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for inputs_batch, targets_batch in zip(
                training_input_messages.split(mini_batch_size),
                training_target_messages.split(mini_batch_size),
            ):
                pred = model(inputs_batch)
                optimizer.zero_grad()
                loss = loss_func(pred, targets_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}:\t{loss.item():.2e}")

        with torch.no_grad():
            inferred_messages = model(test_input_messages)

        (
            messages_loss,
            message_cluster_accuracy,
            prediction_loss,
            prediction_accuracy,
        ) = self._evaluate_inferred_messages(
            test_target_messages,
            inferred_messages,
            test_encoder_contexts,
            test_decoder_contexts,
            test_function_idxs,
        )
        logging.info(
            f"Compositionality network messages loss for taken-out param {taken_out_param}: {messages_loss}"
        )
        logging.info(
            f"Compositionality network network accuracy for taken-out param {taken_out_param}: {message_cluster_accuracy}"
        )
        logging.info(
            f"Compositionality network output loss for taken-out {taken_out_param}: {prediction_loss}"
        )
        logging.info(
            f"Compositionality network output accuracy for taken-out {taken_out_param}: {prediction_accuracy}"
        )
        return (
            messages_loss,
            message_cluster_accuracy,
            prediction_loss,
            prediction_accuracy,
        )

    def _run_unsupervised_clustering(self, visualize: bool = False):
        num_clusters = self._detect_num_clusters()

        (_, _, _, training_messages,) = self._generate_funcs_contexts_messages(1000)

        k_means = cluster.KMeans(n_clusters=num_clusters)
        training_labels = k_means.fit_predict(training_messages)

        if visualize:
            utils.plot_clusters(training_messages, training_labels, "Messages clusters")

        # Align cluster ids with with function/message ids:
        # Generate messages for each function,
        # pair a cluster id with the function most common in it.
        (
            alignment_func_selectors,
            _,
            _,
            alignment_messages,
        ) = self._generate_funcs_contexts_messages(1000)
        alignment_func_idxs = alignment_func_selectors.argmax(dim=1)
        alignment_labels = k_means.predict(alignment_messages)

        func_counts_per_cluster = collections.defaultdict(collections.Counter)
        for i, cluster_label in enumerate(alignment_labels):
            function_idx = alignment_func_idxs[i]
            func_counts_per_cluster[cluster_label][function_idx] += 1

        cluster_label_to_func_idx = {
            cluster_label: func_counts.most_common(1)[0][0]
            for cluster_label, func_counts in func_counts_per_cluster.items()
        }

        assert len(cluster_label_to_func_idx) == num_clusters

        self.clustering_model = k_means
        self.cluster_label_to_func_idx = cluster_label_to_func_idx

    def _evaluate_average_cluster_message_perception(
        self, num_messages_to_average: int = 10,
    ):
        """Sample unseen message from each cluster (the average of N cluster messages),
        feed to decoder, check if prediction is close to prediction of encoder's message for same context."""
        (_, _, _, messages,) = self._generate_funcs_contexts_messages(
            num_messages_to_average
        )
        cluster_label_per_message = self.clustering_model.predict(messages)

        average_messages = []
        function_selectors = []
        for cluster_idx in self.cluster_label_to_func_idx.keys():
            mask = cluster_label_per_message == cluster_idx
            cluster_messages = messages[mask]
            cluster_average_message = cluster_messages.mean(dim=0).unsqueeze(dim=0)
            average_messages.append(
                torch.cat([cluster_average_message] * self.num_exemplars, dim=0)
            )
            cluster_function_selectors = torch.nn.functional.one_hot(
                torch.tensor(
                    [self.cluster_label_to_func_idx[cluster_idx]] * self.num_exemplars
                ),
                num_classes=self.num_functions,
            ).float()
            function_selectors.append(cluster_function_selectors)

        function_selectors = torch.cat(function_selectors, dim=0)
        average_messages = torch.cat(average_messages, dim=0)

        batch_size = average_messages.shape[0]
        encoder_context = self._generate_contexts(batch_size)
        decoder_context = self._get_decoder_context(batch_size, encoder_context)

        target_objects = self._target(decoder_context, function_selectors)
        predictions_by_average_msg = self._predict_by_message(
            average_messages, decoder_context
        )
        predictions_by_average_msg_accuracy = self._evaluate_object_prediction_accuracy(
            decoder_context, predictions_by_average_msg, target_objects
        )

        logging.info(
            f"Perception evaluation: prediction by average message accuracy {predictions_by_average_msg_accuracy}"
        )

        target_predictions = self._predict(
            encoder_context, function_selectors, decoder_context
        )

        predictions_by_average_msg_loss = torch.nn.MSELoss()(
            predictions_by_average_msg, target_predictions
        ).item()
        logging.info(
            f"Perception evaluation: prediction by average message loss: {predictions_by_average_msg_loss}"
        )
        return {
            "predictions_by_average_msg_loss": predictions_by_average_msg_loss,
            "predictions_by_average_msg_accuracy": predictions_by_average_msg_accuracy,
        }

    def _evaluate_clusterization_f_score(self) -> float:
        """Sample unseen messages, clusterize them, return F-score of
        inferred F from Cluster(M) vs. actual F that generated M."""
        (
            func_selectors,
            encoder_contexts,
            decoder_contexts,
            messages,
        ) = self._generate_funcs_contexts_messages(self.num_exemplars)
        cluster_label_per_message = self.clustering_model.predict(messages)

        predicted_func_idxs = np.array(
            [
                self.cluster_label_to_func_idx[cluster_label]
                for cluster_label in cluster_label_per_message
            ]
        )
        function_prediction_f_score = metrics.f1_score(
            func_selectors.argmax(dim=1).numpy(), predicted_func_idxs, average="micro",
        )
        logging.info(f"Unsupervised clustering F score: {function_prediction_f_score}")
        return function_prediction_f_score

    def _evaluate_object_prediction_by_cluster(self) -> float:
        """Generate messages M, get two predictions:
        1. Decoder output for M.
        2. Encoder-Decoder output based on functions F inferred from clusters of M.
        -> Return loss of 1 vs 2.
        """
        (
            func_selectors,
            encoder_contexts,
            decoder_contexts,
            messages,
        ) = self._generate_funcs_contexts_messages(self.num_exemplars)

        cluster_label_per_message = self.clustering_model.predict(messages)

        inferred_func_idxs = np.array(
            [
                self.cluster_label_to_func_idx[cluster_label]
                for cluster_label in cluster_label_per_message
            ]
        )
        inferred_func_selectors = torch.nn.functional.one_hot(
            torch.tensor(inferred_func_idxs), num_classes=self.num_functions
        ).float()

        predictions_by_inferred_func = self._predict(
            encoder_contexts, inferred_func_selectors, decoder_contexts
        )
        predictions_by_func_selectors = self._predict(
            encoder_contexts, func_selectors, decoder_contexts
        )

        object_prediction_loss = torch.nn.MSELoss()(
            predictions_by_func_selectors, predictions_by_inferred_func
        ).item()
        logging.info(f"Loss for unseen message/information: {object_prediction_loss}")

        return object_prediction_loss
