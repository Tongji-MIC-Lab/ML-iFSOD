from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import numpy as np

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model,load_feature_extractor,load_fs_stat,save_hm_head,save_entire_model,load_ft_locator
from logger import Logger
from datasets.dataset_factory import get_dataset,dataset_factory
from models.networks.resnet_dcn import get_meta_net,get_fs_net
from models.networks.pose_dla_dcn import get_dla_meta_net,get_dla_fs_net
from trains.meta_train import MetaTrainer
from detectors.detector_factory import detector_factory
from progress.bar import Bar
from tqdm import tqdm


from models.losses import *
from models.utils import _sigmoid
from utils.oracle_utils import gen_oracle_map
from test import PrefetchDataset

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
trains_path = os.path.join(os.path.dirname(__file__),'lib','trains')
add_path(trains_path)

from utils.utils import AverageMeter


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  if opt.fs_train:
    if not opt.fs_voc:
      fewShotDataset = get_dataset('coco_fewshot','few_shot')
    else:
      fewShotDataset = get_dataset('pascal_fewshot','few_shot')
    fs_loader = torch.utils.data.DataLoader(
      fewShotDataset(opt,'train'),
      batch_size=opt.fs_batch_size,
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=False
      )
    
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  print('Creating model...')
  if 'res' in opt.arch:
    model = get_meta_net(int(opt.arch.split('_')[-1]),opt)
    model = load_feature_extractor(model,opt.fte_path)
    fs_model = get_fs_net(int(opt.arch.split('_')[-1]),opt.heads,opt.head_conv)
  elif 'dla' in opt.arch:
    model = get_dla_meta_net(34,opt.heads,opt)
    model = load_feature_extractor(model,opt.fte_path)
    fs_model = get_dla_fs_net(34,opt.heads,opt.head_conv)

  base_model = create_model(opt.arch, opt.heads, opt.head_conv)
  base_model= load_model(base_model,opt.fte_path)

  loss_stats,loss_ = _get_losses(opt)

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
  start_epoch = 0



  if opt.load_model:
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
  print('start epoch:{}'.format(start_epoch))


  trainer = MetaTrainer(opt,model,optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'),
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return
  if not opt.fs_train:

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

  print('Starting training...')
  losses = []
  if not opt.fs_train:
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
      if epoch > 1:
        opt.data_seed = epoch - 1
        train_loader.dataset.__init__(opt,'train')
      dec = (epoch - 1) // opt.lr_interval
      for param_group in optimizer.param_groups:
          param_group['lr'] *= (0.5**dec)


      mark = epoch if opt.save_all else 'last'

      log_dict_train, _ = trainer.train(epoch, train_loader)
      avg_loss = log_dict_train['loss']
      losses.append(avg_loss)
      logger.write('epoch: {} |'.format(epoch))
      for k, v in log_dict_train.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                  epoch, model, optimizer)

      else:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                  epoch, model, optimizer)
      logger.write('\n')
  else:
    log_dict_fs = fewshot_train(fs_model,model,base_model,loss_,fs_loader,opt,'all')
  logger.close()


def fewshot_train(fs_model,learner,base_model,loss_,data_loader,opt,fs_train_type):
  def set_device(model,fs_optimizer,gpus, device):
    if len(gpus) > 1:
        model = torch.nn.DataParallel(
            model,device_ids = gpus,
            ).to(device)
    else:
        model = model.to(device)
    for state in fs_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device, non_blocking=True)
    return model,fs_optimizer


  save_entire_dir = os.path.join(opt.save_dir,'entire_model')
  log = open(os.path.join(opt.save_dir,'fewshot_train_{}_log.txt'.format(fs_train_type)),'w')
  opt = opt
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters


  if not opt.ablate:
    fs_stat = load_fs_stat(fs_model,learner.state_dict()) 
    fs_model.load_state_dict(fs_stat,strict=False)
    fs_model = load_ft_locator(fs_model,base_model.state_dict())
  else:
    fs_model = load_feature_extractor(fs_model,opt.fte_path)
    
  num_epoch = opt.fs_epoch
  fs_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fs_model.parameters()),opt.fs_lr)
  fs_model.train()
  fs_model,fs_optimizer = set_device(fs_model,fs_optimizer,opt.gpus,opt.device)
  model_with_loss = ModelWithLoss(fs_model,loss_)
  model_with_loss.train()

  losses = []
  if opt.Kshot == 10:
    warm_up_epochs = 15
  elif opt.Kshot == 5:
    warm_up_epochs = 30
  elif opt.Kshot == 1:
    warm_up_epochs = 100
  
  warm_up = True
  for ep in range(1,num_epoch + 1):
    if warm_up:
      if opt.Kshot == 10:
        if (ep - 1) < warm_up_epochs:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] = opt.fs_lr * (ep / warm_up_epochs)
        if ep -1 == 40:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] *= 0.2
        if ep -1 == 80:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] *= 0.2
      elif opt.Kshot ==5:
        if (ep - 1) < warm_up_epochs:
          for param_group in fs_optimizer.param_groups:
              param_group['lr'] = opt.fs_lr * (ep / warm_up_epochs)
        if ep - 1 == 40:
          for param_group in fs_optimizer.param_groups:
              param_group['lr'] *= 0.2
        if ep -1 == 80:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] *= 0.2
      elif opt.Kshot == 1:
        if (ep - 1) in range(0,warm_up_epochs,5):
          for param_group in fs_optimizer.param_groups:
              param_group['lr'] = opt.fs_lr * ((ep - 1 + 5) / warm_up_epochs)
        if ep -1 == 100:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] *= 0.2
        if ep -1 == 400:
          for param_group in fs_optimizer.param_groups:
            param_group['lr'] *= 0.2

    losses_e = []   
    pbar = tqdm(total=len(data_loader))
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
          break

      for k in batch:
          if k != 'meta':
              batch[k] = batch[k].to(device=opt.device, non_blocking=True)     
      inputs = batch

      _,loss,_ =  model_with_loss(inputs)
      loss = loss.mean()
      losses_e.append(loss.item())

      fs_optimizer.zero_grad()
      loss.backward()
      fs_optimizer.step()

      fs_lr = fs_optimizer.param_groups[0]['lr']

      avg_loss = sum(losses_e)/len(losses_e)
      pbar.set_description('Fs_train_type:{} |Epoch:{} |Iteration:{} |fs_lr:{:.3e} |Loss:{:2f} |Avg_loss:{:.3f}'.format(fs_train_type.capitalize(),ep,iter_id,fs_lr,float(loss.item()),avg_loss))
      log.write('Fs_train_type:{} |Epoch:{} |Iteration:{} |fs_lr:{:.3e} |Loss:{:2f} |Avg_loss:{:.3f} \n'.format(fs_train_type.capitalize(),ep,iter_id,fs_lr,float(loss.item()),avg_loss))
      pbar.update(1)

      del loss
    pbar.close()
    losses.append(avg_loss)
    if float(avg_loss) < 0:
     break
  log.close()


  if not os.path.exists(save_entire_dir):
    os.mkdir(save_entire_dir)

  save_entire_path = os.path.join(save_entire_dir,'entire_model_{}_last_test.pth'.format(str(opt.Kshot)))
  save_model(save_entire_path,num_epoch,fs_model)

  opt.entire_model = save_entire_path


  ret = {'loss':avg_loss}
  return ret



class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch):
    outputs = self.model(batch['input']) 
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    for s in range(opt.num_stacks): 
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) /
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks

      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

def _get_losses(opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss



if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
