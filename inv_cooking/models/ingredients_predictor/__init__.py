# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn

from inv_cooking.config import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)
from inv_cooking.models.modules.ff_decoder import FFDecoder
from inv_cooking.models.modules.rnn_decoder import DecoderRNN
from inv_cooking.models.modules.transformer_decoder import DecoderTransformer
from inv_cooking.models.modules.utils import freeze_fn
from inv_cooking.utils.metrics import SoftIoULoss, TargetDistributionLoss


def label2_k_hots(labels, pad_value, remove_eos=False):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    k_hots = torch.zeros(labels.size(0), labels.size(1), pad_value + 1).type_as(inp_)
    k_hots.scatter_(2, inp_, 1)
    k_hots, _ = k_hots.max(dim=1)

    # remove pad position
    k_hots = k_hots[:, :-1]

    # handle eos
    if remove_eos:
        # this is used by tfset/lstmset when computing losses and
        # by all auto-regressive models when computing f1 metrics
        k_hots = k_hots[:, 1:]
    return k_hots


def mask_from_eos(prediction, eos_value, mult_before=True):
    mask = torch.ones(prediction.size()).type_as(prediction).byte()
    mask_aux = torch.ones(prediction.size(0)).type_as(prediction).byte()

    # find eos in label prediction
    for idx in range(prediction.size(1)):
        # force mask to have 1s in the first position to avoid division
        # by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (prediction[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (prediction[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask


def predictions_to_idxs(
    label_logits,
    maxnumlabels,  ## TODO check how this is used, and whether the +1 in the set predictor construction is necessary
    pad_value,
    th:float=1,
    cardinality_prediction=None,
    which_loss="bce",
    use_empty_set=False,
):

    assert th > 0 and th <= 1

    card_offset = 0 if use_empty_set else 1

    # select topk elements
    probs, idxs = torch.topk(
        label_logits, k=maxnumlabels, dim=1, largest=True, sorted=True
    )
    idxs_clone = idxs.clone()

    # mask to identify elements within the top-maxnumlabel ones which satisfy the threshold th
    if which_loss == "td":
        # cumulative threshold
        mask = torch.ones(probs.size()).type_as(probs).byte()
        for idx in range(probs.size(1)):
            mask_step = torch.sum(probs[:, 0:idx], dim=-1) < th
            mask[:, idx] = mask[:, idx] * mask_step
    else:
        # probility threshold
        mask = (probs > th).byte()

    # if the model has cardinality prediction
    if cardinality_prediction is not None:

        # get the argmax for each element in the batch to get the cardinality
        # (note that the output is N - 1, e.g. argmax = 0 means that there's 1 element)
        # unless we are in the empty set case, e.g. argmax = 0 means there there are 0 elements

        # select cardinality
        _, card_idx = torch.max(cardinality_prediction, dim=-1)

        mask = torch.ones(probs.size()).type_as(probs).byte()
        aux_mask = torch.ones(mask.size(0)).type_as(probs).byte()

        for i in range(mask.size(-1)):
            # If the cardinality prediction is higher than i, it means that from this point
            # on the mask must be 0. Predicting 0 cardinality means 0 objects when
            # use_empty_set=True and 1 object when use_empty_set=False
            # real cardinality value is
            above_cardinality = i < card_idx + card_offset
            # multiply the auxiliar mask with this condition
            # (once you multiply by 0, the following entries will also be 0)
            aux_mask = aux_mask * above_cardinality
            mask[:, i] = aux_mask
    else:
        if not use_empty_set:
            mask[:, 0] = 1

    idxs_clone[mask == 0] = pad_value

    return idxs_clone


def get_ingr_predictor(
    config: IngredientPredictorConfig,
    vocab_size: int,
    maxnumlabels: int,
    eos_value: int,
):
    """
    Create the ingredient predictor based on the configuration
    """
    if "ff" in config.model:
        config = cast(IngredientPredictorFFConfig, config)
        return FeedForwardIngredientsPredictor.from_config(config, maxnumlabels, vocab_size)
    elif "lstm" in config.model:
        config = cast(IngredientPredictorLSTMConfig, config)
        return AutoRegressiveIngredientsPredictor.create_lstm_from_config(config, maxnumlabels, vocab_size, eos_value)
    elif "tf" in config.model:
        config = cast(IngredientPredictorTransformerConfig, config)
        return AutoRegressiveIngredientsPredictor.create_tf_from_config(config, maxnumlabels, vocab_size, eos_value)


class IngredientsPredictor(nn.Module, abc.ABC):
    """
    Interface of any ingredient predictor implementation
    """

    def __init__(self, remove_eos: bool):
        super().__init__()
        self.remove_eos = remove_eos

    def forward(
            self,
            img_features: torch.Tensor,
            label_target: Optional[torch.Tensor] = None,
            compute_losses: bool = False,
            compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict the ingredients of the image features extracted by an image encoder
        :param img_features: image features - shape (N, embedding_size, sequence_length)
        :param label_target: ground truth, the ingredients to find - shape (N, max_num_labels)
        :param compute_losses: whether or not to compute loss between output and target
        :param compute_predictions: whether or not to output the predicted ingredients
        """
        assert (label_target is not None and compute_losses) or (label_target is None and not compute_losses)

        losses: Dict[str, torch.Tensor] = {}
        predictions: Optional[torch.Tensor] = None
        if not compute_losses and not compute_predictions:
            return losses, predictions
        else:
            return self._forward_impl(img_features, label_target, compute_losses, compute_predictions)

    @abc.abstractmethod
    def _forward_impl(
            self,
            img_features: torch.Tensor,
            label_target: Optional[torch.Tensor] = None,
            compute_losses: bool = False,
            compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ...


class AutoRegressiveIngredientsPredictor(IngredientsPredictor):
    """
    Implementation of an ingredient predictor based on an auto-regressive architecture
    """

    @staticmethod
    def create_tf_from_config(
            config: IngredientPredictorTransformerConfig,
            maxnumlabels: int,
            vocab_size: int,
            eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":

        maxnumlabels += 1  # required for EOS token
        print(
            "Building Transformer decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {} / "
            "Num. Attention Heads {} / Num. Layers {}.".format(
                config.model,
                config.embed_size,
                config.dropout,
                maxnumlabels,
                config.n_att,
                config.layers,
            ),
            flush=True,
        )
        decoder = DecoderTransformer(
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=maxnumlabels,
            attention_nheads=config.n_att,
            pos_embeddings=False,
            num_layers=config.layers,
            learned=False,
            normalize_before=True,
        )

        pad_value = vocab_size - 1
        if config.with_set_prediction:
            label_loss = nn.BCELoss(reduction="mean")
            eos_loss = nn.BCELoss(reduction="none")
        else:
            label_loss = nn.CrossEntropyLoss(ignore_index=pad_value, reduction="mean")
            eos_loss = None

        model = AutoRegressiveIngredientsPredictor(
            decoder,
            maxnumlabels,
            vocab_size,
            crit=label_loss,
            crit_eos=eos_loss,
            pad_value=pad_value,
            eos_value=eos_value,
            perminv=config.with_set_prediction,
        )

        if config.freeze:
            freeze_fn(model)
        return model

    @staticmethod
    def create_lstm_from_config(
            config: IngredientPredictorLSTMConfig,
            maxnumlabels: int,
            vocab_size: int,
            eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":

        maxnumlabels += 1  # required for EOS token
        print(
            "Building LSTM decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {}. ".format(
                config.model, config.embed_size, config.dropout, maxnumlabels
            ),
            flush=True,
        )
        decoder = DecoderRNN(
            config.embed_size,
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=maxnumlabels,
        )

        pad_value = vocab_size - 1
        if config.with_set_prediction:
            label_loss = nn.BCELoss(reduction="mean")
            eos_loss = nn.BCELoss(reduction="none")
        else:
            label_loss = nn.CrossEntropyLoss(ignore_index=pad_value, reduction="mean")
            eos_loss = None

        model = AutoRegressiveIngredientsPredictor(
            decoder,
            maxnumlabels,
            vocab_size,
            crit=label_loss,
            crit_eos=eos_loss,
            pad_value=pad_value,
            eos_value=eos_value,
            perminv=config.with_set_prediction,
        )

        if config.freeze:
            freeze_fn(model)
        return model

    def __init__(
            self,
            decoder: nn.Module,
            max_num_labels: int,
            vocab_size: int,
            crit=None,
            crit_eos=None,
            pad_value: int = 0,
            eos_value: int = 0,
            perminv: bool = True,
            eps: float = 1e-8,
    ):
        super().__init__(remove_eos=False)
        self.decoder = decoder
        self.maxnumlabels = max_num_labels
        self.crit = crit
        self.perminv = perminv
        self.pad_value = pad_value
        self.eos_value = eos_value
        self.crit_eos = crit_eos
        self.eps = eps
        self.vocab_size = vocab_size

    def _forward_impl(
            self,
            img_features: torch.Tensor,
            label_target: Optional[torch.Tensor] = None,
            compute_losses: bool = False,
            compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        losses: Dict[str, torch.Tensor] = {}
        predictions: Optional[torch.Tensor] = None

        # use auto-regressive decoder to predict labels (sample function)
        # output label_logits is only used to compute losses in case of self.perminv (no teacher forcing)
        # predictions output is used for all auto-regressive models
        predictions, label_logits = self.decoder.sample(
            img_features, None, first_token_value=0, replacement=False,
        )

        if compute_predictions:
            # mask labels after finding eos (cardinality)
            sample_mask = mask_from_eos(
                predictions, eos_value=self.eos_value, mult_before=False
            )
            predictions[sample_mask == 0] = self.pad_value

        if compute_losses:
            # add dummy first word to sequence and remove last
            first_word = torch.zeros(len(label_target)).type_as(label_target)
            shift_target = torch.cat([first_word.unsqueeze(-1), label_target], -1)[
                           :, :-1
                           ]
            if self.perminv:
                # autoregressive mode for decoder when training with permutation invariant objective
                # e.g. lstmset and tfset

                # apply softmax nonlinearity before pooling across timesteps
                label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

                # find idxs for eos label
                # eos probability is the one assigned to the position self.eos_value of the softmax
                # this is used with bce loss only
                eos = label_probs[:, :, self.eos_value]
                eos_pos = (
                        label_target == self.eos_value
                )  # all zeros except position where eos is in the gt
                eos_head = (label_target != self.pad_value) & (
                        label_target != self.eos_value
                )  # 1s for gt label positions, 0s starting from eos position in the gt
                eos_target = (
                    ~eos_head
                )  # 0s for gt label positions, 1s starting from eos position in the gt

                # select transformer steps to pool (steps corresponding to set elements, i.e. labels)
                label_probs = label_probs * eos_head.float().unsqueeze(-1)

                # pool
                label_probs, _ = torch.max(label_probs, dim=1)

                # compute label loss
                target_k_hot = label2_k_hots(
                    label_target, self.pad_value, remove_eos=True
                )
                loss = self.crit(label_probs[:, 1:], target_k_hot.float())
                losses["label_loss"] = loss

                # compute eos loss
                eos_loss = self.crit_eos(eos, eos_target.float())
                # eos loss is computed for all timesteps <= eos in gt and
                # equally penalizes the head (all 0s) and the true eos position (1)
                losses["eos_loss"] = (
                        0.5
                        * (eos_loss * eos_pos.float()).sum(1)
                        / (eos_pos.float().sum(1) + self.eps)
                        + 0.5
                        * (eos_loss * eos_head.float()).sum(1)
                        / (eos_head.float().sum(1) + self.eps)
                ).mean()

            else:
                # other autoregressive models
                # we need to recompute logits using teacher forcing (forward pass)
                label_logits, _ = self.decoder(img_features, None, shift_target)
                label_logits_v = label_logits.view(
                    label_logits.size(0) * label_logits.size(1), -1
                )

                # compute label loss
                label_target_v = label_target.view(-1)
                loss = self.crit(label_logits_v, label_target_v)
                losses["label_loss"] = loss

        return losses, predictions


class FeedForwardIngredientsPredictor(IngredientsPredictor):
    """
    Implementation of an ingredient predictor based on a feed-forward architecture
    """

    @staticmethod
    def from_config(config: IngredientPredictorFFConfig, maxnumlabels: int, vocab_size: int) -> "FeedForwardIngredientsPredictor":
        cardinality_pred = config.cardinality_pred
        print(
            "Building feed-forward decoder {}. Embed size {} / Dropout {} / "
            " Max. Num. Labels {} / Num. Layers {}".format(
                config.model,
                config.embed_size,
                config.dropout,
                maxnumlabels,
                config.layers,
            ),
            flush=True,
        )
        decoder = FFDecoder(
            config.embed_size,
            vocab_size,
            config.embed_size,
            dropout=config.dropout,
            pred_cardinality=cardinality_pred,
            nobjects=maxnumlabels,
            n_layers=config.layers,
        )

        # cardinality loss
        if cardinality_pred == "cat":
            print("Using categorical cardinality loss.", flush=True)
            cardinality_loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            print("Using no cardinality loss.", flush=True)
            cardinality_loss = None

        # label and eos loss
        label_losses = {
            "bce": nn.BCEWithLogitsLoss(reduction="mean")
            if "ff" in config.model
            else nn.BCELoss(reduction="mean"),
            "iou": SoftIoULoss(reduction="mean"),
            "td": TargetDistributionLoss(reduction="mean"),
        }
        pad_value = vocab_size - 1
        loss_key = {k for k in label_losses.keys() if k in config.model}.pop()
        label_loss = label_losses[loss_key]

        model = FeedForwardIngredientsPredictor(
            decoder,
            maxnumlabels,
            vocab_size,
            crit=label_loss,
            crit_cardinality=cardinality_loss,
            pad_value=pad_value,
            loss_label=loss_key,
            card_type=cardinality_pred,
        )

        if config.freeze:
            freeze_fn(model)
        return model

    def __init__(
            self,
            decoder: nn.Module,
            max_num_labels: int,
            vocab_size: int,
            crit=None,
            crit_cardinality=None,
            pad_value: int = 0,
            threshold: float = 0.5,
            loss_label: str = "bce",
            card_type: str = "none",
            use_empty_set: bool = False,
            eps: float = 1e-8,
    ):
        super().__init__(remove_eos=True)
        self.decoder = decoder
        self.maxnumlabels = max_num_labels
        self.crit = crit
        self.threshold = threshold
        self.pad_value = pad_value
        self.crit_cardinality = crit_cardinality
        self.loss_label = loss_label
        self.card_type = card_type
        self.eps = eps
        self.use_empty_set = use_empty_set
        self.vocab_size = vocab_size

    def _forward_impl(
            self,
            img_features: torch.Tensor,
            label_target: Optional[torch.Tensor] = None,
            compute_losses: bool = False,
            compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        losses: Dict[str, torch.Tensor] = {}
        predictions: Optional[torch.Tensor] = None
        label_logits, cardinality_logits = self.decoder(img_features)

        if compute_losses:
            target_k_hot = label2_k_hots(label_target, self.pad_value)
            target_k_hot = target_k_hot.type_as(label_logits)
            cardinality_target = target_k_hot.sum(dim=-1).unsqueeze(1)

            # compute labels loss
            losses["label_loss"] = self.crit(label_logits, target_k_hot)

            # compute cardinality loss if needed
            if self.crit_cardinality is not None:
                # subtract 1 from num_target to match class idxs (1st label corresponds to class 0) only
                # 1st label corresponds to 0 only if use_empty_set is false
                # otherwise, 1st label corresponds to 1
                offset = 0 if self.use_empty_set else 1
                losses["cardinality_loss"] = self.crit_cardinality(
                    cardinality_logits,
                    (cardinality_target.squeeze() - offset).long(),
                )

        if compute_predictions:
            # consider cardinality
            if self.card_type == "cat":
                cardinality = torch.nn.functional.log_softmax(
                    cardinality_logits + self.eps, dim=-1
                )
            else:
                cardinality = None

            # apply nonlinearity to label logits
            if self.loss_label == "td":
                label_probs = nn.functional.softmax(label_logits, dim=-1)
            else:
                label_probs = torch.sigmoid(label_logits)

            # get label ids
            predictions = predictions_to_idxs(
                label_probs,
                self.maxnumlabels,
                self.pad_value,
                th=self.threshold,
                cardinality_prediction=cardinality,
                which_loss=self.loss_label,
                use_empty_set=self.use_empty_set,
            )

        return losses, predictions
