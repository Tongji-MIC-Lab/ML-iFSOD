from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval as COCOevalOrigin,Params
import numpy as np
import json
import os
from collections import defaultdict
import torch.utils.data as data
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
import time
import cv2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
data_path = os.path.dirname(os.path.dirname(__file__))
lib_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
add_path(data_path)
add_path(lib_path)
from sample.ctdet import *
from opts import opts


class COCOeval(COCOevalOrigin):
  def __init__(self,log_dir,img_ids,cat_ids, cocoGt=None, cocoDt=None,iouType='segm',dataset='coco'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(img_ids)
            self.params.catIds = sorted(cat_ids)

        self.log_path = os.path.join(log_dir,'cocoEval.txt')
        self.dataset = dataset
        
class COCO(data.Dataset):
  num_classes = 80
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(COCO, self).__init__()
    self.opt = opt
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    elif split == 'visual':
      self.img_dir = '/data/cm/clone/CenterNet/CenterNet_ROOT/data/coco/visimage'
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_train2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_{}2017.json').format(split)
    self.max_objs = 128
    self.class_name = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 
    self._valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]

    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    self.novel_class = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
      'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 
      'sheep', 'couch', 'train', 'tv']

    self.base_class = [c for c in self.class_name if not c in self.novel_class]
    self.novel_class_ind = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
    self.base_class_ind = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
                          33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                          53, 54, 55, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 
                          77, 78, 79]
    self.catname2ind =     {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 
                  'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
                  'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 
                  'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 
                  'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 
                  'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 
                  'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 
                  'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 
                  'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 
                  'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 
                  'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 
                  'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}

    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.novel_ids = [self.coco.getCatIds([cls_name])[0] for cls_name in self.novel_class]
    self.base_ids = [self.coco.getCatIds([cls_name])[0] for cls_name in self.base_class if cls_name != '__background__']
    self.imgs_base = [self.coco.getImgIds(catIds=[cat_id]) for cat_id in self.base_ids] 
    self.imgs_novel = [self.coco.getImgIds(catIds=[cat_id]) for cat_id in self.novel_ids]
    self.images = []
    self.test_type = opt.test_type
    if split == 'train':
      if opt.train_origin:
        self.images = self.coco.getImgIds()
      else:
        for i in range(len(self.imgs_base)): 
            self.images = list(set(self.images)|set(self.imgs_base[i]))
    else:
      if opt.test_type == 'all':
        self.catIds = self.coco.getCatIds()
      elif opt.test_type == 'novel':
        self.catIds = self.novel_ids 
      elif opt.test_type == 'base':
        self.catIds = self.base_ids
      elif opt.test_type == 'incremental':
        self.catIds = self.coco.getCatIds()
      else:
        self.catIds_novel = self.novel_ids
        self.catIds_base = self.base_ids
        self.catIds_all = self.coco.getCatIds()
        
      self.images = self.coco.getImgIds()
    if self.split == 'visual':
      self.images = self.coco.getImgIds(imgIds=481028)
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:  
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]  
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir,directly=False):
    if not directly:
      self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))

    if self.test_type in ['base','novel','all']:
      coco_eval = COCOeval(save_dir,self.images,self.catIds,self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      print('Test {} class.'.format(self.test_type))
      coco_eval.summarize()
    elif self.test_type == 'incremental':
      coco_eval = COCOeval(save_dir,self.images,self.catIds,self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      precision = coco_eval.eval['precision']
      recall = coco_eval.eval['recall']
      cat_name = self.opt.cat_name
      cat_ind = self.catname2ind[cat_name]
      novel_class_inds = self.novel_class_ind[:self.novel_class_ind.index(cat_ind)+1]
      cal_inds = self.base_class_ind + novel_class_inds
      print(cat_ind)
      print(cal_inds)
      log = open(os.path.join(self.opt.save_dir,'fewshot_train_{}.txt'.format(self.opt.cat_name)),'a')
      for i,name in enumerate(self.class_name[1:]):
        p_c = precision[:,:,i,0,-1]
        r_c = recall[:,i,0,-1]
        mean_p_c = np.mean(p_c[p_c>-1])
        mean_r_c = np.mean(r_c[r_c>-1])
        print('class_name:{} | precision: {:.3f} | recall: {:.3f} \n'.format(name,mean_p_c,mean_r_c))
        log.write('class_name:{} | precision: {:.3f} | recall: {:.3f} \n'.format(name,mean_p_c,mean_r_c))
      pbn = precision[:,:,cal_inds,0,-1]
      rbn = recall[:,cal_inds,0,-1]
      pb = precision[:,:,self.base_class_ind,0,-1]
      rb = recall[:,self.base_class_ind,0,-1]
      p = precision[:,:,cat_ind,0,-1]
      r = recall[:,cat_ind,0,-1]
      mean_pbn = np.mean(pbn[pbn>-1])
      mean_rbn = np.mean(rbn[rbn>-1])
      mean_pb = np.mean(pb[pb>-1])
      mean_rb = np.mean(rb[rb>-1])
      mean_p = np.mean(p[p>-1])
      mean_r = np.mean(r[r>-1])
      print('Test scales: {} \n'.format(self.opt.test_scales))
      print('fs_seed: {}'.format(self.opt.fs_seed))
      print('Calculate results of incremental classes with base classes: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |'.format(cat_name,mean_pbn,mean_rbn))
      print('Calculate results of base classes: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |'.format(cat_name,mean_pb,mean_rb))
      print('Calculate results of incremental class: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |'.format(cat_name,mean_p,mean_r))
      
      log.write('fs_seed: {}'.format(self.opt.fs_seed))
      log.write('Test scales: {} \n'.format(self.opt.test_scales))
      log.write('Calculate results of incremental classes with base classes: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |\n'.format(cat_name,mean_pbn,mean_rbn))
      log.write('Calculate results of base classes: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |\n'.format(cat_name,mean_pb,mean_rb))
      log.write('Calculate results of incremental class: \n Incremental class: {} | Precision: {:0.3f} | Recall: {:0.3f} |\n'.format(cat_name,mean_p,mean_r))
      log.close()

    else:
      coco_eval = COCOeval(save_dir,self.images,self.catIds_novel,self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      print('Test novel class.')
      coco_eval.summarize()
      print('\n')

      coco_eval = COCOeval(save_dir,self.images,self.catIds_base,self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      print('Test base class.')
      coco_eval.summarize()
      print('\n')

      coco_eval = COCOeval(save_dir,self.images,self.catIds_all,self.coco, coco_dets, "bbox")
      coco_eval.evaluate()
      coco_eval.accumulate()
      print('Test all class.')
      coco_eval.summarize()




if __name__ == '__main__':

  class Dataset(COCO,TwoStageDataset):
    pass
  opt = opts().parse()
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  dataset = Dataset(opt,'train')
  train_loader = data.DataLoader(
      dataset, 
      batch_size=10, 
      shuffle=True,
      num_workers=1,
      pin_memory=True,
      drop_last=True
  )

  for i,batch in enumerate(train_loader): 
    if i == 1:
      break
    img = Image.fromarray((batch['hm'][0]*255).numpy())
    img.show()