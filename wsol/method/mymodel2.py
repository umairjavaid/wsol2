import torch
import torch.nn as nn

__all__ = ['mymodel2']


def change_dir(grad):
  return -1 * grad

def custom(grad, model_output,labels_):
  i = 0
  for col in labels_:
    label = col.data 
    max_ouput = torch.max(model_output[i,label,:,:])
    norm_output = model_output[i,label,:,:]/max_ouput
    mask = norm_output > 0.85
    chd = change_dir(grad[i,label,0,0])
    grad.select(0, i).select(0,label).copy_(grad[i,label,:,:].masked_fill_(mask,chd))
    i += 1 
  return grad


class MyModel2(nn.Module):
  def __init__(self):
      super(MyModel2, self).__init__()
      #self.attention = None

  def forward(self, input_):
      if not self.training:
          return input_
      else:
          attention = torch.mean(input_, dim=1, keepdim=True)
          importance_map = torch.sigmoid(attention)
          return input_.mul(importance_map)


