from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter



class MetaTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.model = model

    def set_device(self, gpus, chunk_sizes, device):     
        if len(gpus) > 1:
            self.model = DataParallel(
                self.model,device_ids = gpus,
                chunk_sizes = chunk_sizes).to(device)
        else:
            self.model = self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        if phase == 'train':
            self.model.train()
        else:
            if len(self.opt.gpus) > 1:
                self.model = self.model.module
            self.model.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        #avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        avg_loss = AverageMeter()
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        lr = self.optimizer.param_groups[0]['lr']
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)     #batch中各个head对应的tensor都要to device
            loss = self.model(batch).mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            avg_loss.update(
                    loss.item(), batch['input'].size(0))
            Bar.suffix = Bar.suffix + '| Learner_Loss: {:.4f} | Avg_Learner_Loss: {:.4f} |lr:{:.4e} |update_lr:{:.2e}'.format(avg_loss.val,avg_loss.avg,lr,opt.update_lr)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
        
            #if opt.debug > 0:
                #self.debug(batch, output, iter_id)
        
            #if opt.test:
                #self.save_result(output, batch, results)
            del loss
        
        bar.finish()
        ret = {'loss':avg_loss.avg}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results
  
    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        """reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]"""
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError
  
    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)


    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)