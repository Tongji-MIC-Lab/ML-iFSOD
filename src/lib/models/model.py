from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import glob

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net


_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']   
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]

  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


def load_feature_extractor(model, model_path):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']   
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    elif k.startswith('hm') or k.startswith('wh') or k.startswith('reg'): 
      continue
    else:
      state_dict[k] = state_dict_[k]

  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict: 
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  
  return model

def load_fs_stat(model,meta_stat_dict):
  start_epoch = 0
  state_dict = {}
  
  # convert data_parallal to model
  for k in meta_stat_dict:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = meta_stat_dict[k]
    elif k.startswith('learner'):
      state_dict[k[8:]] = meta_stat_dict[k]
    else:
      state_dict[k] = meta_stat_dict[k]

  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]   
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict: 
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]

  
  return state_dict


def load_ft_locator(fs_model,ft_model_state):
  base_inds = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
  def backhook(module,grad_input,grad_output):
    grad_input = list(grad_input)
    mask_w = torch.ones_like(grad_input[1])
    mask_b = torch.ones_like(grad_input[2])
    mask_w[base_inds] = 0
    mask_b[base_inds] = 0 
    grad_input[1] *= mask_w.float()
    grad_input[2] *= mask_b.float()
    return tuple(grad_input)
  
  fs_state = fs_model.state_dict()

  start_epoch = 0
  state_dict = {}
  
  # convert data_parallal to model
  for k in fs_state:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = fs_state[k]
    else:
      state_dict[k] = fs_state[k]
      if k.startswith('hm.2'):
        state_dict[k][base_inds,...] = ft_model_state[k][base_inds,...] if isinstance(state_dict[k],torch.FloatTensor) else ft_model_state[k][base_inds,...].cuda()
        #state_dict[k] = ft_model_state[k] if isinstance(state_dict[k],torch.FloatTensor) else ft_model_state[k].cuda()


  fs_model.load_state_dict(state_dict)
  fs_model.hm[2].register_backward_hook(backhook)

  
  return fs_model


def save_hm_head(path,model):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()

  state_dict = {key:state_dict[key] for key in state_dict if key.startswith('hm.2')}
  data = {'state_dict': state_dict}
  torch.save(data, path)

def save_entire_model(fs_model,hm_dirpath,save_path,epoch,num_classes=80):
  entire_stats = fs_model.state_dict()
  heads_path = glob.glob(os.path.join(hm_dirpath,'*'))
  heads_path = sorted(heads_path,key = lambda x:os.path.basename(x).split('.')[0].split('_')[-1])
  heads_stats = [torch.load(path)['state_dict'] for path in heads_path]

  hm_weight = torch.cat([stat['hm.2.weight'] for stat in heads_stats],dim = 0)
  hm_bias = torch.cat([stat['hm.2.bias'] for stat in heads_stats],dim = 0)

  assert hm_weight.shape[0] == num_classes and hm_bias.shape[0] == num_classes

  entire_stats['hm.2.weight'] = hm_weight
  entire_stats['hm.2.bias'] = hm_bias

  data = {'state_dict':entire_stats,'epoch':epoch}
  torch.save(data,save_path)

  return entire_stats
  
