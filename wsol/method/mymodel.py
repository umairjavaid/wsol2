import torch
import torch.nn as nn

__all__ = ['mymodel']

def change_dir(grad):
  return -1 * grad

def custom(grad, model_output,labels_):
  i = 0
  for col in labels_:
    label = col.data 
    max_ouput = torch.max(model_output[i,label,:,:])
    norm_output = model_output[i,label,:,:]/max_ouput
    mask = norm_output > 0.8
    chd = change_dir(grad[i,label,0,0])
    grad.select(0, i).select(0,label).copy_(grad[i,label,:,:].masked_fill_(mask,chd))
    i += 1 
  return grad

def get_loss(output_dict, gt_labels, **kwargs):
  return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
          nn.CrossEntropyLoss()(output_dict['x'], gt_labels.long())
