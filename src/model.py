# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import math
from modules.encoder import EncoderCNN
from modules.ff_decoder import FFDecoder
from modules.transformer_decoder import DecoderTransformer
from modules.rnn_decoder import DecoderRNN
from utils.metrics import softIoU, softIoULoss, DCLoss, DC, targetDistLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2_k_hots(labels, pad_value, remove_eos=False):
    # labels is a list of (possibly variable length) lists.
    # labels are numpy array
    if type(labels) == list:
        tmp = np.array([i + [pad_value]*(len(max(labels, key=len))-len(i)) for i in labels])
        labels = torch.from_numpy(tmp).to(device)

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    k_hots = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
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
    mask = torch.ones(prediction.size()).to(device).byte()
    mask_aux = torch.ones(prediction.size(0)).to(device).byte()

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


def predictions_to_idxs(label_logits,
                    maxnumlabels,
                    pad_value,
                    th=1,
                    cardinality_prediction=None,
                    which_loss='bce',
                    accumulate_probs=False,
                    use_empty_set=False):

    assert th > 0 and th <= 1


    card_offset = 0 if use_empty_set else 1

    # select topk elements
    probs, idxs = torch.topk(label_logits, k=maxnumlabels, dim=1, largest=True, sorted=True)
    idxs_clone = idxs.clone()

    # mask to identify elements within the top-maxnumlabel ones which satisfy the threshold th
    if which_loss == 'td':
        # cumulative threshold
        mask = torch.ones(probs.size()).to(device).byte()
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

        if accumulate_probs:
            for c in range(cardinality_prediction.size(-1)):
                value = torch.sum(torch.log(probs[:, 0:c + 1]), dim=-1)
                cardinality_prediction[:, c] += value

        # select cardinality
        _, card_idx = torch.max(cardinality_prediction, dim=-1)

        mask = torch.ones(probs.size()).to(device).byte()
        aux_mask = torch.ones(mask.size(0)).to(device).byte()

        for i in range(mask.size(-1)):
            # If the cardinality prediction is higher than i, it means that from this point
            # on the mask must be 0. Predicting 0 cardinality means 0 objects when
            # use_empty_set=True and 1 object when use_empty_set=False
            # real cardinality value is
            above_cardinality = (i < card_idx + card_offset)
            # multiply the auxiliar mask with this condition
            # (once you multiply by 0, the following entries will also be 0)
            aux_mask = aux_mask * above_cardinality
            mask[:, i] = aux_mask
    else:
        if not use_empty_set:
            mask[:, 0] = 1

    idxs_clone[mask == 0] = pad_value

    return idxs_clone


def get_model(args, vocab_size):

    # build image encoder
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder,
                               args.image_model)

    use_empty_set = (True if args.dataset in ['coco', 'nuswide'] else False)

    # build set predictor
    if args.decoder == 'ff':
        print(
            'Building feed-forward decoder. Embed size {} / Dropout {} / '
            'Cardinality Prediction {} / Max. Num. Labels {} / Num. Layers {}'.format(
                args.embed_size, args.dropout_decoder, args.pred_cardinality, args.maxnumlabels,
                args.ff_layers),
            flush=True)

        decoder = FFDecoder(
            args.embed_size,
            vocab_size,
            args.embed_size,
            dropout=args.dropout_decoder,
            pred_cardinality=args.pred_cardinality,
            nobjects=args.maxnumlabels,
            n_layers=args.ff_layers,
            use_empty_set=use_empty_set)

    elif args.decoder == 'lstm':
        print(
            'Building LSTM decoder. Embed size {} / Dropout {} / Max. Num. Labels {}. '.format(
                args.embed_size, args.dropout_decoder, args.maxnumlabels),
            flush=True)

        decoder = DecoderRNN(
            args.embed_size,
            args.embed_size,
            vocab_size,
            dropout=args.dropout_decoder,
            seq_length=args.maxnumlabels,
            num_instrs=1)

    elif args.decoder == 'tf':
        print(
            'Building Transformer decoder. Embed size {} / Dropout {} / Max. Num. Labels {} / '
            'Num. Attention Heads {} / Num. Layers {}.'.format(
                args.embed_size, args.dropout_decoder, args.maxnumlabels, args.n_att,
                args.tf_layers),
            flush=True)

        decoder = DecoderTransformer(
            args.embed_size,
            vocab_size,
            dropout=args.dropout_decoder,
            seq_length=args.maxnumlabels,
            num_instrs=1,
            attention_nheads=args.n_att,
            pos_embeddings=False,
            num_layers=args.tf_layers,
            learned=False,
            normalize_before=True)

    # label and eos loss
    label_losses = {
        'bce': nn.BCEWithLogitsLoss(reduction='mean') if args.decoder == 'ff' else nn.BCELoss(reduction='mean'), 
        'iou': softIoULoss(reduction='mean'),
        'td': targetDistLoss(reduction='mean'), 
    }
    pad_value = vocab_size - 1
    print('Using {} loss.'.format(args.label_loss), flush=True)
    if args.decoder == 'ff':
        label_loss = label_losses[args.label_loss]
        eos_loss = None
    elif args.decoder in ['tf', 'lstm'] and args.perminv:
        label_loss = label_losses[args.label_loss]
        eos_loss = nn.BCELoss(reduction='mean')
    else:
        label_loss = nn.CrossEntropyLoss(ignore_index=pad_value, reduction='mean')
        eos_loss = None

    # cardinality loss
    if args.pred_cardinality == 'dc':
        print('Using Dirichlet-Categorical cardinality loss.', flush=True)
        cardinality_loss = DCLoss(U=args.U, dataset=args.dataset, reduction='mean')
    elif args.pred_cardinality == 'cat':
        print('Using categorical cardinality loss.', flush=True)
        cardinality_loss = nn.CrossEntropyLoss(reduction='mean')
    else:
        print('Using no cardinality loss.', flush=True)
        cardinality_loss = None

    model = SetPred(
        decoder,
        encoder_image,
        args.maxnumlabels,
        crit=label_loss,
        crit_eos=eos_loss,
        crit_cardinality=cardinality_loss,
        pad_value=pad_value,
        perminv=args.perminv,
        decoder_ff=True if args.decoder == 'ff' else False,
        th=args.th,
        loss_label=args.label_loss,
        replacement=args.replacement,
        card_type=args.pred_cardinality,
        dataset=args.dataset,
        U=args.U,
        use_empty_set=use_empty_set)

    return model


class SetPred(nn.Module):

    def __init__(self,
                 decoder,
                 image_encoder,
                 maxnumlabels,
                 crit=None,
                 crit_eos=None,
                 crit_cardinality=None,
                 pad_value=0,
                 perminv=True,
                 decoder_ff=False,
                 th=0.5,
                 loss_label='bce',
                 replacement=False,
                 card_type='none',
                 dataset='voc',
                 U=2.36,
                 use_empty_set=False,
                 eps=1e-8):

        super(SetPred, self).__init__()
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.decoder_ff = decoder_ff
        self.maxnumlabels = maxnumlabels
        self.crit = crit
        self.th = th
        self.perminv = perminv
        self.pad_value = pad_value
        self.crit_eos = crit_eos
        self.crit_cardinality = crit_cardinality
        self.loss_label = loss_label
        self.replacement = replacement
        self.card_type = card_type
        self.dataset = dataset
        self.u_term = math.log(U)
        self.eps = eps
        self.use_empty_set = use_empty_set

    def forward(self, img_inputs, label_target=None, maxnumlabels=0, keep_cnn_gradients=False, compute_losses=False, compute_predictions=False):

        losses = {}
        predictions = None

        assert (label_target is not None and compute_losses) or (label_target is None and not compute_losses)

        if not compute_losses and not compute_predictions:
            return losses, predictions

        # encode image
        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        if self.decoder_ff:
            # use ff decoder to predict set of labels and cardinality
            label_logits, cardinality_logits = self.decoder(img_features)

            if compute_losses:
                # label target to k_hot
                target_k_hot = label2_k_hots(label_target, self.pad_value)
                # cardinality target
                cardinality_target = target_k_hot.sum(dim=-1).unsqueeze(1)

                # compute labels loss
                losses['label_loss'] = self.crit(label_logits, target_k_hot)

                # compute cardinality loss if needed
                if self.crit_cardinality is not None:
                    # subtract 1 from num_target to match class idxs (1st label corresponds to class 0) only
                    # 1st label corresponds to 0 only if use_empty_set is false
                    # otherwise, 1st label corresponds to 1
                    offset = 0 if self.use_empty_set else 1
                    losses['cardinality_loss'] = self.crit_cardinality(
                        cardinality_logits, (cardinality_target.squeeze() - offset).long())

            if compute_predictions:
                # consider cardinality
                if self.card_type == 'dc' and self.loss_label == 'bce':
                    offset = 0 if self.use_empty_set else 1
                    cardinality = torch.log(DC(cardinality_logits, dataset=self.dataset))
                    u_term = np.array(list(range(cardinality.size(-1)))) + offset
                    u_term = u_term * self.u_term
                    u_term = torch.from_numpy(u_term).to(device).unsqueeze(0).float()
                    cardinality = cardinality + u_term
                elif self.card_type == 'cat':
                    cardinality = torch.nn.functional.log_softmax(cardinality_logits + self.eps, dim=-1)
                else:
                    cardinality = None

                # apply nonlinearity to label logits
                if self.loss_label == 'td':
                    label_probs = nn.functional.softmax(label_logits, dim=-1)
                else:
                    label_probs = torch.sigmoid(label_logits)

                # get label ids
                predictions = predictions_to_idxs(
                    label_probs,
                    maxnumlabels,
                    self.pad_value,
                    th=self.th,
                    cardinality_prediction=cardinality,
                    which_loss=self.loss_label,
                    accumulate_probs=self.card_type == 'dc' and self.loss_label == 'bce',
                    use_empty_set=self.use_empty_set)

        else:  # auto-regressive models

            # use auto-regressive decoder to predict labels (sample function)
            # output label_logits is only used to compute losses in case of self.perminv (no teacher forcing)
            # predictions output is used for all auto-regressive models
            predictions, label_logits = self.decoder.sample(
                img_features,
                None,
                first_token_value=0,
                replacement=self.replacement)

            if compute_predictions:
                # mask labels after finding eos (cardinality)
                sample_mask = mask_from_eos(predictions, eos_value=0, mult_before=False)
                predictions[sample_mask == 0] = self.pad_value
            else:
                predictions = None

            if compute_losses:
                # add dummy first word to sequence and remove last
                first_word = torch.zeros(label_target.size(0))
                shift_target = torch.cat([first_word.unsqueeze(-1).to(device).long(), label_target],
                                         -1)[:, :-1]
                if self.perminv:
                    # autoregressive mode for decoder when training with permutation invariant objective
                    # e.g. lstmset and tfset

                    # apply softmax nonlinearity before pooling across timesteps
                    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

                    # find idxs for eos label
                    # eos probability is the one assigned to the first position of the softmax
                    # this is used with bce loss only
                    eos = label_probs[:, :, 0]
                    eos_pos = (label_target == 0)  # all zeros except position where eos is in the gt
                    eos_head = ((label_target != self.pad_value) & (label_target != 0))  # 1s for gt label positions, 0s starting from eos position in the gt
                    eos_target = ~eos_head  # 0s for gt label positions, 1s starting from eos position in the gt

                    # select transformer steps to pool (steps corresponding to set elements, i.e. labels)
                    label_probs = label_probs * eos_head.float().unsqueeze(-1)

                    # pool
                    label_probs, _ = torch.max(label_probs, dim=1)

                    # compute label loss
                    target_k_hot = label2_k_hots(label_target, self.pad_value, remove_eos=True)
                    loss = self.crit(label_probs[:, 1:], target_k_hot)
                    losses['label_loss'] = loss

                    # compute eos loss
                    eos_loss = self.crit_eos(eos, eos_target.float())
                    # eos loss is computed for all timesteps <= eos in gt and
                    # equally penalizes the head (all 0s) and the true eos position (1)
                    losses['eos_loss'] = 0.5 * (eos_loss * eos_pos.float()).sum(1) + \
                                    0.5 * (eos_loss * eos_head.float()).sum(1) / (
                                            eos_head.float().sum(1) + self.eps)

                else:
                    # other autoregressive models
                    # we need to recompute logits using teacher forcing (forward pass)
                    label_logits, _ = self.decoder(img_features, None, shift_target)
                    label_logits_v = label_logits.view(label_logits.size(0) * label_logits.size(1), -1)

                    # compute label loss
                    label_target_v = label_target.view(-1)
                    loss = self.crit(label_logits_v, label_target_v)
                    losses['label_loss'] = loss

        return losses, predictions
