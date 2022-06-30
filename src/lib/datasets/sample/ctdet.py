from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
lib_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
add_path(lib_path)


from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math



class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])  #获得一张图片对应的标注的id，注意一张图片通常有多个标注；
    anns = [ann for ann in self.coco.loadAnns(ids=ann_ids) if ann['category_id'] in self.base_ids]  #ann_ids是多个标注形成的列表，anns以列表形式存储多个标注的信息，每个标注用字典形式表示;
    num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
    if self.opt.keep_res:    #whether keep the original resolution during validation;这个参数值在test有用;
      input_h = (height | self.opt.pad) + 1   #opt.pad设置为127，即2^7-1,在二进制为7个1; ‘binary1|binary2’为按位或运算，即两个字符中有一个为1，则为1;可以保证结果始终是128的倍数，比如当height不足127时，结果为128；
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0   #s取原图长宽中的较大者;
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) 
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])    #input_w,input_h是在参数中指定的大小，而在仿射变换中，是输出空间图的参考.用于寻找输出空间中的三个点;
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)   #根据仿射变换矩阵trans_input，将输入img进行仿射变换
    inp = (inp.astype(np.float32) / 255.) 
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)   #cv2读取的img通道在第2维，即(h,w,c)，因此需变换到(c,h,w)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   #根据特征图的大小，给出一个仿射变换矩阵;

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)  #为图片中的每一个obj都设置一个class specific的大小标签;
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)  #cat_spec_wh的mask，cat_spec_wh非零的位置标记为1
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian  
    #centerNet中mse_loss设置为false,因此关注 draw_umich_gaussian函数

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)   #根据特征图的仿射变换矩阵，将对应的框的坐标调整到合适位置
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)  #hm:heatmap;ct_int:center;cls_id:第k个obj的类别id;也就是说一个gt_obj只会对heatmap中的某一层进行赋值;
        wh[k] = 1. * w, 1. * h   #wh是[num_objs,2]的tensor，存储了一张图片中每一个objects的宽和高
        ind[k] = ct_int[1] * output_w + ct_int[0]    #ind表示第k个目标的中心点在特征图上的索引
        reg[k] = ct - ct_int   #reg表示offset，即将float转换为int所产生的误差
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    #print((hm==1).astype(float).sum(),num_objs)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id} #c是特征图的中心（图片中心，不是框中心）; s取特征图长宽的较大者;gt_det存储了一张图片中所有目标的bbox四个坐标和类别信息;
      ret['meta'] = meta
    return ret  






class metaDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  def concatRet(self,rets):   #rets中的数据都是Numpy形式，这里没有考虑meta属性,返回的是torch.tensor形式
    train_ret = {}
    for head in rets[0].keys():
      train_ret[head] = []  
    for ret in rets:
      for head,ts in ret.items():
        if head != 'meta':  #meta以字典形式存储一张图片的gt，不参与拼接;
          ret[head] = np.expand_dims(ts,0)
        train_ret[head].append(ret[head])
    for head,values in train_ret.items():
      if head != 'meta':  #最终一个task中，'meta'对应一个列表，列表按顺序储存一个task的gt，每一个gt都是一个字典;
        train_ret[head] = torch.from_numpy(np.concatenate(values,0))
    return train_ret

  def __getitem__(self, index):
    img_ids = self.tasks[index]['samples']
    img_cls = self.tasks[index]['cls_id']
    file_names = [info['file_name'] for info in self.coco.loadImgs(ids=img_ids)]   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
    ann_ids = [self.coco.getAnnIds(imgIds=img_id) for img_id in img_ids]  #一张图片的所有标注的id以列表形式作为一个元素；
    rets = []
    for i in range(len(file_names)):
      img_id = img_ids[i]
      img_path = os.path.join(self.img_dir, file_names[i])
      anns = [ann for ann in self.coco.loadAnns(ids=ann_ids[i]) if ann['category_id'] == img_cls]   #筛选出标注对应图片类别的标注，其他忽略;
      num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量

      img = cv2.imread(img_path)

      height, width = img.shape[0], img.shape[1]
      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
      if self.opt.keep_res:    #whether keep the original resolution during validation;这个参数值在test有用;
        input_h = (height | self.opt.pad) + 1   #opt.pad设置为127，即2^7-1,在二进制为7个1; ‘binary1|binary2’为按位或运算，即两个字符中有一个为1，则为1;可以保证结果始终是128的倍数，比如当height不足127时，结果为128；
        input_w = (width | self.opt.pad) + 1
        s = np.array([input_w, input_h], dtype=np.float32)
      else:
        s = max(img.shape[0], img.shape[1]) * 1.0   #s取原图长宽中的较大者;
        input_h, input_w = self.opt.input_h, self.opt.input_w
      
      flipped = False
      if self.split == 'train':
        if not self.opt.not_rand_crop:
          s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
          w_border = self._get_border(128, img.shape[1])
          h_border = self._get_border(128, img.shape[0])
          c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
          c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) 
        else:
          sf = self.opt.scale
          cf = self.opt.shift
          c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
          c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
          s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        
        if np.random.random() < self.opt.flip:
          flipped = True
          img = img[:, ::-1, :]
          c[0] =  width - c[0] - 1
          

      trans_input = get_affine_transform(
        c, s, 0, [input_w, input_h])    #input_w,input_h是在参数中指定的大小，而在仿射变换中，是输出空间图的参考.用于寻找输出空间中的三个点;
      inp = cv2.warpAffine(img, trans_input, 
                          (input_w, input_h),
                          flags=cv2.INTER_LINEAR)   #根据仿射变换矩阵trans_input，将输入img进行仿射变换
      inp = (inp.astype(np.float32) / 255.) 
      if self.split == 'train' and not self.opt.no_color_aug:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      inp = (inp - self.mean) / self.std
      inp = inp.transpose(2, 0, 1)   #cv2读取的img通道在第2维，即(h,w,c)，因此需变换到(c,h,w)

      output_h = input_h // self.opt.down_ratio
      output_w = input_w // self.opt.down_ratio
      num_classes = self.num_classes
      trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   #根据特征图的大小，给出一个仿射变换矩阵;

      hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)  #热图由num_class维修改到1维
      wh = np.zeros((self.max_objs, 2), dtype=np.float32)
      dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
      reg = np.zeros((self.max_objs, 2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
      #cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)  #为图片中的每一个obj都设置一个class specific的大小标签;
      #cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)  #cat_spec_wh的mask，cat_spec_wh非零的位置标记为1
      
      draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian  
      #centerNet中mse_loss设置为false,因此关注 draw_umich_gaussian函数

      gt_det = []
      for k in range(num_objs):
        ann = anns[k]
        bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
        cls_id = int(self.cat_ids[ann['category_id']])
        if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[:2] = affine_transform(bbox[:2], trans_output)   #根据特征图的仿射变换矩阵，将对应的框的坐标调整到合适位置
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
        if h > 0 and w > 0:
          radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius = max(0, int(radius))
          radius = self.opt.hm_gauss if self.opt.mse_loss else radius
          ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
          ct_int = ct.astype(np.int32)
          draw_gaussian(hm[cls_id], ct_int, radius)  #这里不需要考虑类别了，全部修改为hm[0]
          #print(hm)
          wh[k] = 1. * w, 1. * h   #wh是[num_objs,2]的tensor，存储了一张图片中每一个objects的宽和高
          ind[k] = ct_int[1] * output_w + ct_int[0]    #ind表示第k个目标的中心点在特征图上的索引
          reg[k] = ct - ct_int   #reg表示offset，即将float转换为int所产生的误差
          reg_mask[k] = 1
          #cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
          #cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
          gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                        ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
      
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
      if self.opt.dense_wh:
        hm_a = hm.max(axis=0, keepdims=True)
        dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        del ret['wh']
      #elif self.opt.cat_spec_wh:
        #ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
        #del ret['wh']
      if self.opt.reg_offset:
        ret.update({'reg': reg})
      if self.opt.debug > 0 or not self.split == 'train':
        gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
        meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id} #c是特征图的中心（图片中心，不是框中心）; s取特征图长宽的较大者;gt_det存储了一张图片中所有目标的bbox四个坐标和类别信息;
        ret['meta'] = meta
      rets.append(ret)


    rets = self.concatRet(rets)  

    #print(rets['meta'])
      
    return rets

class fewShotDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  def concatRet(self,rets):   #rets中的数据都是Numpy形式，这里没有考虑meta属性,返回的是torch.tensor形式
    train_ret = {}
    for head in rets[0].keys():
      train_ret[head] = []  
    for ret in rets:
      for head,ts in ret.items():
        if head != 'meta':  #meta以字典形式存储一张图片的gt，不参与拼接;
          ret[head] = np.expand_dims(ts,0)
        train_ret[head].append(ret[head])
    for head,values in train_ret.items():
      if head != 'meta':  #最终一个task中，'meta'对应一个列表，列表按顺序储存一个task的gt，每一个gt都是一个字典;
        train_ret[head] = torch.from_numpy(np.concatenate(values,0))
    return train_ret

  def __getitem__(self, index):
    obj_cls,img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])  #获得一张图片对应的标注的id，注意一张图片通常有多个标注；
    anns = self.coco.loadAnns(ids=ann_ids)
    #print([ann['id'] for ann in anns])
    anns = [ann for ann in self.coco.loadAnns(ids=ann_ids) if ann['category_id'] == obj_cls]  
    #print([ann['id'] for ann in anns])
    num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量
    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
    if self.opt.keep_res:    #whether keep the original resolution during validation;这个参数值在test有用;
      input_h = (height | self.opt.pad) + 1   #opt.pad设置为127，即2^7-1,在二进制为7个1; ‘binary1|binary2’为按位或运算，即两个字符中有一个为1，则为1;可以保证结果始终是128的倍数，比如当height不足127时，结果为128；
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0   #s取原图长宽中的较大者;
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) 
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])    #input_w,input_h是在参数中指定的大小，而在仿射变换中，是输出空间图的参考.用于寻找输出空间中的三个点;
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)   #根据仿射变换矩阵trans_input，将输入img进行仿射变换
    inp = (inp.astype(np.float32) / 255.) 
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)   #cv2读取的img通道在第2维，即(h,w,c)，因此需变换到(c,h,w)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   #根据特征图的大小，给出一个仿射变换矩阵;

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)  #为图片中的每一个obj都设置一个class specific的大小标签;
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)  #cat_spec_wh的mask，cat_spec_wh非零的位置标记为1
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian  
    #centerNet中mse_loss设置为false,因此关注 draw_umich_gaussian函数

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)   #根据特征图的仿射变换矩阵，将对应的框的坐标调整到合适位置
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
      #print('h:{},w:{}'.format(h,w))
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)      #hm:heatmap;ct_int:center;cls_id:第k个obj的类别id;也就是说一个gt_obj只会对heatmap中的某一层进行赋值;
        #print(np.sum(hm[cls_id]))
        wh[k] = 1. * w, 1. * h   #wh是[num_objs,2]的tensor，存储了一张图片中每一个objects的宽和高
        ind[k] = ct_int[1] * output_w + ct_int[0]    #ind表示第k个目标的中心点在特征图上的索引
        reg[k] = ct - ct_int   #reg表示offset，即将float转换为int所产生的误差
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    #print(np.sum(hm))
    #print((hm==1).astype(float).sum())
    #sys.exit(0)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    #np.save(os.path.join(vis_dir,file_name.split('.')[0] + '.npy'),hm)
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id} #c是特征图的中心（图片中心，不是框中心）; s取特征图长宽的较大者;gt_det存储了一张图片中所有目标的bbox四个坐标和类别信息;
      ret['meta'] = meta
    return ret  

# class fewShotDataset(data.Dataset):
#   def _coco_box_to_bbox(self, box):
#     bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
#                     dtype=np.float32)
#     return bbox

#   def _get_border(self, border, size):
#     i = 1
#     while size - border // i <= border // i:
#         i *= 2
#     return border // i
#   def concatRet(self,rets):   #rets中的数据都是Numpy形式，这里没有考虑meta属性,返回的是torch.tensor形式
#     train_ret = {}
#     for head in rets[0].keys():
#       train_ret[head] = []  
#     for ret in rets:
#       for head,ts in ret.items():
#         if head != 'meta':  #meta以字典形式存储一张图片的gt，不参与拼接;
#           ret[head] = np.expand_dims(ts,0)
#         train_ret[head].append(ret[head])
#     for head,values in train_ret.items():
#       if head != 'meta':  #最终一个task中，'meta'对应一个列表，列表按顺序储存一个task的gt，每一个gt都是一个字典;
#         train_ret[head] = torch.from_numpy(np.concatenate(values,0))
#     return train_ret

#   def __getitem__(self, index):
#     obj_cls,img_id = self.images[index]
#     file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
#     img_path = os.path.join(self.img_dir, file_name)
#     ann_ids = self.coco.getAnnIds(imgIds=[img_id])  #获得一张图片对应的标注的id，注意一张图片通常有多个标注；
#     #anns_ = self.coco.loadAnns(ids=ann_ids)
#     anns = [ann for ann in self.coco.loadAnns(ids=ann_ids) if ann['id'] in self.ann_ids]  #ann_ids是多个标注形成的列表，anns以列表形式存储多个标注的信息，每个标注用字典形式表示;
#     print([ann['id'] for ann in anns])
#     #raise SystemExit
#     num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量
#     img = cv2.imread(img_path)

#     height, width = img.shape[0], img.shape[1]
#     c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
#     if self.opt.keep_res:    #whether keep the original resolution during validation;这个参数值在test有用;
#       input_h = (height | self.opt.pad) + 1   #opt.pad设置为127，即2^7-1,在二进制为7个1; ‘binary1|binary2’为按位或运算，即两个字符中有一个为1，则为1;可以保证结果始终是128的倍数，比如当height不足127时，结果为128；
#       input_w = (width | self.opt.pad) + 1
#       s = np.array([input_w, input_h], dtype=np.float32)
#     else:
#       s = max(img.shape[0], img.shape[1]) * 1.0   #s取原图长宽中的较大者;
#       input_h, input_w = self.opt.input_h, self.opt.input_w
    
#     flipped = False
#     if self.split == 'train':
#       if not self.opt.not_rand_crop:
#         s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
#         w_border = self._get_border(128, img.shape[1])
#         h_border = self._get_border(128, img.shape[0])
#         c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
#         c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) 
#       else:
#         sf = self.opt.scale
#         cf = self.opt.shift
#         c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
#         c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
#         s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
#       if np.random.random() < self.opt.flip:
#         flipped = True
#         img = img[:, ::-1, :]
#         c[0] =  width - c[0] - 1
        

#     trans_input = get_affine_transform(
#       c, s, 0, [input_w, input_h])    #input_w,input_h是在参数中指定的大小，而在仿射变换中，是输出空间图的参考.用于寻找输出空间中的三个点;
#     inp = cv2.warpAffine(img, trans_input, 
#                          (input_w, input_h),
#                          flags=cv2.INTER_LINEAR)   #根据仿射变换矩阵trans_input，将输入img进行仿射变换
#     inp = (inp.astype(np.float32) / 255.) 
#     if self.split == 'train' and not self.opt.no_color_aug:
#       color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
#     inp = (inp - self.mean) / self.std
#     inp = inp.transpose(2, 0, 1)   #cv2读取的img通道在第2维，即(h,w,c)，因此需变换到(c,h,w)

#     output_h = input_h // self.opt.down_ratio
#     output_w = input_w // self.opt.down_ratio
#     num_classes = self.num_classes
#     trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   #根据特征图的大小，给出一个仿射变换矩阵;

#     hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
#     wh = np.zeros((self.max_objs, 2), dtype=np.float32)
#     dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
#     reg = np.zeros((self.max_objs, 2), dtype=np.float32)
#     ind = np.zeros((self.max_objs), dtype=np.int64)
#     reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
#     cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)  #为图片中的每一个obj都设置一个class specific的大小标签;
#     cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)  #cat_spec_wh的mask，cat_spec_wh非零的位置标记为1
    
#     draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian  
#     #centerNet中mse_loss设置为false,因此关注 draw_umich_gaussian函数

#     gt_det = []
#     for k in range(num_objs):
#       ann = anns[k]
#       bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
#       cls_id = int(self.cat_ids[ann['category_id']])
#       if flipped:
#         bbox[[0, 2]] = width - bbox[[2, 0]] - 1
#       bbox[:2] = affine_transform(bbox[:2], trans_output)   #根据特征图的仿射变换矩阵，将对应的框的坐标调整到合适位置
#       bbox[2:] = affine_transform(bbox[2:], trans_output)
#       bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
#       bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
#       h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
#       #print('h:{},w:{}'.format(h,w))
#       if h > 0 and w > 0:
#         radius = gaussian_radius((math.ceil(h), math.ceil(w)))
#         radius = max(0, int(radius))
#         radius = self.opt.hm_gauss if self.opt.mse_loss else radius
#         ct = np.array(
#           [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
#         ct_int = ct.astype(np.int32)
#         draw_gaussian(hm[cls_id], ct_int, radius)      #hm:heatmap;ct_int:center;cls_id:第k个obj的类别id;也就是说一个gt_obj只会对heatmap中的某一层进行赋值;
#         #print(np.sum(hm[cls_id]))
#         wh[k] = 1. * w, 1. * h   #wh是[num_objs,2]的tensor，存储了一张图片中每一个objects的宽和高
#         ind[k] = ct_int[1] * output_w + ct_int[0]    #ind表示第k个目标的中心点在特征图上的索引
#         reg[k] = ct - ct_int   #reg表示offset，即将float转换为int所产生的误差
#         reg_mask[k] = 1
#         cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
#         cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
#         if self.opt.dense_wh:
#           draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
#         gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
#                        ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
#     #print(np.sum(hm))
#     #print((hm==1).astype(float).sum())
#     #sys.exit(0)
#     ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
#     #np.save(os.path.join(vis_dir,file_name.split('.')[0] + '.npy'),hm)
#     if self.opt.dense_wh:
#       hm_a = hm.max(axis=0, keepdims=True)
#       dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
#       ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
#       del ret['wh']
#     elif self.opt.cat_spec_wh:
#       ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
#       del ret['wh']
#     if self.opt.reg_offset:
#       ret.update({'reg': reg})
#     if self.opt.debug > 0 or not self.split == 'train':
#       gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
#                np.zeros((1, 6), dtype=np.float32)
#       meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id} #c是特征图的中心（图片中心，不是框中心）; s取特征图长宽的较大者;gt_det存储了一张图片中所有目标的bbox四个坐标和类别信息;
#       ret['meta'] = meta
#     return ret  




  
    
class VisualDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])  #获得一张图片对应的标注的id，注意一张图片通常有多个标注；
    #anns = [ann for ann in self.coco.loadAnns(ids=ann_ids) if ann['category_id'] in self.base_ids]  #ann_ids是多个标注形成的列表，anns以列表形式存储多个标注的信息，每个标注用字典形式表示;
    anns = [ann for ann in self.coco.loadAnns(ids=ann_ids)]
    num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量

    vis_dir = os.path.join(self.data_dir,'hmvis')
    os.system('cp {} {}'.format(img_path,vis_dir))

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
    num_classes = 80
    hm = np.zeros((num_classes, height, width), dtype=np.float32)   
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian  
    #centerNet中mse_loss设置为false,因此关注 draw_umich_gaussian函数

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
      cls_id = int(self.cat_ids[ann['category_id']])
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)  #hm:heatmap;ct_int:center;cls_id:第k个obj的类别id;也就是说一个gt_obj只会对heatmap中的某一层进行赋值;
    
    np.save(os.path.join(vis_dir,file_name.split('.')[0] + '.npy'),hm)

    return hm



class TwoStageDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   #coco.loadimgs返回一个列表，其中一个元素就是一个图片的信息，以字典形式存储;'file_name'对应的值是图片的名称
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])  #获得一张图片对应的标注的id，注意一张图片通常有多个标注；
    if self.opt.train_origin:
      anns = self.coco.loadAnns(ids=ann_ids)
    else:
      anns = [ann for ann in self.coco.loadAnns(ids=ann_ids) if ann['category_id'] in self.base_ids]  #ann_ids是多个标注形成的列表，anns以列表形式存储多个标注的信息，每个标注用字典形式表示;
    num_objs = min(len(anns), self.max_objs)  #一张图片可能会对应多个标注（多个物体），通过max_objs限制最大标注数量

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   #center of the image
    if self.opt.keep_res:    #whether keep the original resolution during validation;这个参数值在test有用;
      input_h = (height | self.opt.pad) + 1   #opt.pad设置为127，即2^7-1,在二进制为7个1; ‘binary1|binary2’为按位或运算，即两个字符中有一个为1，则为1;可以保证结果始终是128的倍数，比如当height不足127时，结果为128；
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0   #s取原图长宽中的较大者;
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) 
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])    #input_w,input_h是在参数中指定的大小，而在仿射变换中，是输出空间图的参考.用于寻找输出空间中的三个点;
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)   #根据仿射变换矩阵trans_input，将输入img进行仿射变换
    inp = (inp.astype(np.float32) / 255.) 
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)   #cv2读取的img通道在第2维，即(h,w,c)，因此需变换到(c,h,w)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   #根据特征图的大小，给出一个仿射变换矩阵;

    hm = np.zeros((output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat = np.zeros((self.max_objs),dtype=np.int64)
    
    draw_gaussian = draw_umich_gaussian  

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])   #[x1,y1,x2,y2]
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)   #根据特征图的仿射变换矩阵，将对应的框的坐标调整到合适位置
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]   # h,w是特征图上的框的大小
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)   #特征图上的框的中心
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm, ct_int, radius)  #hm:heatmap;ct_int:center;cls_id:第k个obj的类别id;也就是说一个gt_obj只会对heatmap中的某一层进行赋值;
        wh[k] = 1. * w, 1. * h   #wh是[num_objs,2]的tensor，存储了一张图片中每一个objects的宽和高
        ind[k] = ct_int[1] * output_w + ct_int[0]    #ind表示第k个目标的中心点在特征图上的索引
        reg[k] = ct - ct_int   #reg表示offset，即将float转换为int所产生的误差
        reg_mask[k] = 1
        cat[k] = cls_id
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    #print((hm==1).astype(float).sum(),num_objs)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'reg': reg,'cat':cat}
    # if self.opt.debug > 0 or not self.split == 'train':
    #   gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
    #            np.zeros((1, 6), dtype=np.float32)
    #   meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id} #c是特征图的中心（图片中心，不是框中心）; s取特征图长宽的较大者;gt_det存储了一张图片中所有目标的bbox四个坐标和类别信息;
    #   ret['meta'] = meta
    return ret  
