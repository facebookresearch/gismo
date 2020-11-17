import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl

from models.im2ingr import Im2Ingr
from models.ingredients_predictor import label2_k_hots
from utils.metrics import update_error_counts, compute_metrics

class LitInverseCooking(pl.LightningModule):

  def __init__(self,
               im_args,
               ingrpred_args,
               optim_args,
               dataset_name,  ## TODO: check if needed at all
               maxnumlabels,
               ingr_vocab_size):
    super().__init__()

    self.model = Im2Ingr(im_args, ingrpred_args, ingr_vocab_size, dataset_name, maxnumlabels)

    self.freeze_image_encoder = im_args.freeze_pretrained
    self.lr = optim_args.lr
    self.scale_lr_pretrained = optim_args.scale_lr_pretrained
    self.lr_decay_rate = optim_args.lr_decay_rate
    self.lr_decay_every = optim_args.lr_decay_every
    self.weight_decay = optim_args.weight_decay
    self.ingr_prediction_loss_weights = optim_args.ingr_prediction_loss_weights

    self._reset_error_counts(overall=True)

  def forward(self, img, label_target=None, compute_losses=False, compute_predictions=False):
    losses, predictions = self.model(img, label_target, compute_losses=compute_losses, compute_predictions=compute_predictions)
    return losses, predictions

  def training_step(self, batch, batch_idx):
    x, y = batch
    losses, _ = self(x, y, compute_losses=True)

    return losses

  def validation_step(self, batch, batch_idx):
    metrics = self._shared_eval(batch, batch_idx, 'val')
    return metrics

  def test_step(self, batch, batch_idx):
    metrics = self._shared_eval(batch, batch_idx, 'test')
    return metrics

  def _shared_eval(self, batch, batch_idx, prefix):
    x, y = batch

    # get model predictions
    # predictions format can either be a matrix of size batch_size x maxnumlabels, where
    # each row contains the integer labels of an image, followed by pad_value
    # or a list of sublists, where each sublist contains the integer labels of an image
    # and len(list) = batch_size and len(sublist) is variable
    _, predictions = self(x, compute_predictions=True)

    # convert model predictions and targets to k-hots
    pred_k_hots = label2_k_hots(
        predictions, self.model.ingr_vocab_size - 1, 
        remove_eos=not self.model.ingr_predictor.is_decoder_ff)
    target_k_hots = label2_k_hots(
        y, self.model.ingr_vocab_size - 1, remove_eos=not self.model.ingr_predictor.is_decoder_ff)

    # update overall and per class error counts
    update_error_counts(self.overall_error_counts, pred_k_hots, target_k_hots, which_metrics=['o_f1', 'c_f1', 'i_f1'])

    # compute i_f1 metric and save n_samples
    metrics = compute_metrics(self.overall_error_counts, which_metrics=['i_f1'])
    metrics['n_samples'] = pred_k_hots.shape[0]    

    return metrics
    
  def validation_epoch_end(self, valid_step_outputs):
    self.eval_epoch_end(valid_step_outputs, 'val')

  def test_epoch_end(self, test_step_outputs):
    self.eval_epoch_end(test_step_outputs, 'test')

  def eval_epoch_end(self, eval_step_outputs, split):
    # compute validation set metrics
    overall_metrics = compute_metrics(self.overall_error_counts, ['o_f1', 'c_f1'])

    # init avg metrics to 0
    avg_metrics = dict(zip(eval_step_outputs[0].keys(), [0]*len(eval_step_outputs[0].keys())))

    # update avg metrics
    for out in eval_step_outputs:
      for k in out.keys():
        avg_metrics[k] += out[k]

    # log all validation metrics
    for k in avg_metrics.keys():
      if k != 'n_samples':
        self.log(f'{split}_{k}', avg_metrics[k]/avg_metrics['n_samples'])

    self.log(f'{split}_o_f1', overall_metrics['o_f1'])
    self.log(f'{split}_c_f1', overall_metrics['c_f1'])

    self._reset_error_counts(overall=True)

  def training_step_end(self, losses):

    total_loss = 0

    # avg losses across gpus
    for k in losses.keys():
      losses[k] = losses[k].mean()

    if 'label_loss' in losses.keys():
      self.log('label_loss', losses['label_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      total_loss += (losses['label_loss'] * self.ingr_prediction_loss_weights['label_loss'])
    if 'cardinality_loss' in losses.keys():
      self.log('cardinality_loss', losses['cardinality_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      total_loss += (losses['cardinality_loss'] * self.ingr_prediction_loss_weights['cardinality_loss'])
    if 'eos_loss' in losses.keys():
      self.log('eos_loss', losses['eos_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      total_loss += (losses['eos_loss'] * self.ingr_prediction_loss_weights['eos_loss'])

    self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return total_loss
    
    
  def validation_step_end(self, metrics):
    valid_step_outputs = self._shared_eval_step_end(metrics)

    return valid_step_outputs

  def test_step_end(self, metrics):
    test_step_outputs = self._shared_eval_step_end(metrics)
    
    return test_step_outputs

  def _shared_eval_step_end(self, metrics):
    eval_step_outputs = {}
    
    # sum metric within mini-batches
    for k in metrics.keys():
      eval_step_outputs[k] = sum(metrics[k])

    # reset per sample error counts
    self._reset_error_counts(perimage=True)
      
    return eval_step_outputs

  def _reset_error_counts(self, overall=False, perimage=False):
    # reset all error counts (done at the end of each epoch)
    if overall:
      self.overall_error_counts = {
          'c_tp': 0,
          'c_fp': 0,
          'c_fn': 0,
          'c_tn': 0,
          'o_tp': 0,
          'o_fp': 0,
          'o_fn': 0,
          'i_tp': 0,
          'i_fp': 0,
          'i_fn': 0
      }
    # reset per sample error counts (done at the end of each iteration)
    if perimage:
      self.overall_error_counts['i_tp'] = 0
      self.overall_error_counts['i_fp'] = 0
      self.overall_error_counts['i_fn'] = 0

  def configure_optimizers(self):  ## TODO: adapt to all scenarios -- e.g. pretrained parts: lr*scale_lr_pretrained

    optimizer = torch.optim.Adam(
      [{
          'params': self.model.ingr_predictor.parameters()
      }, {
          'params': self.model.image_encoder.parameters(),
          'lr': self.lr*self.scale_lr_pretrained
      }],
      lr=self.lr,
      weight_decay=self.weight_decay)

   
    scheduler = {'scheduler': ExponentialLR(optimizer, self.lr_decay_rate),
                 'interval': 'epoch',
                 'frequency': self.lr_decay_every}

    return [optimizer], [scheduler]


