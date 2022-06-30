from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import json
import os

import torch.utils.data as data

class fewShotPascalVOC(data.Dataset):
  num_classes = 80
  default_resolution = [384, 384]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(fewShotPascalVOC, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'voc')
    self.img_dir = os.path.join(self.data_dir, 'images')
    _ann_name = {'train': 'trainval0712', 'val': 'test2007'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      'pascal_{}.json').format(_ann_name[split])
    self.K = opt.Kshot
    if self.K == 10 :
      self.fs_sample = os.path.join(self.data_dir,'few_shot_samples_pascal.json')
    elif self.K == 5:
      self.fs_sample = os.path.join(self.data_dir,'five_shot_samples_pascal.json')
    elif self.K == 1:
      self.fs_sample = os.path.join(self.data_dir,'one_shot_samples_pascal.json')
    else:
      raise Exception('Incorrect Args of K')

    self.max_objs = 50
    self.class_name = ["aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
     "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
     "train", "tvmonitor"]
    self._valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]
    self._valid_ids_voc = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]   #novel_ids in coco
    self.cocoCat_to_vocCat = {v:i+1 for i,v in enumerate(self._valid_ids_voc)}
    #coco中按0-79编号时，novel_class按顺序在在其中的索引；
    self.voc_ind = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]  #各个类在hm中的通道位置;
    #self.novel_ind = [ind + 1 for ind in self._valid_ids_voc]
    #1-80
    self._valid_ids_ = np.arange(1, 21, dtype=np.int32)
    #self.catIds = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    self.cat_ids = {v: self.voc_ind[i] for i, v in enumerate(self._valid_ids_)}  #根据voc设置的类别编号得到在COCO所对应的80个类别下的索引;
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

    

    print('==> initializing pascal {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)

    with open(self.fs_sample,'r') as f:
      fs_samples = json.load(f)

    self.images = []
    
    for c,imgs in fs_samples.items():
        for img in imgs:
            self.images.append((int(c),img))

    self.num_samples = len(self.images)

    self.catIds = self.coco.getCatIds()

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format_(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      #for j in range(1, self.num_classes + 1):
      for j,ind in enumerate(self.novel_ind):
        if isinstance(all_bboxes[img_id][ind], np.ndarray):
          detections[j+1][i] = all_bboxes[img_id][ind].tolist()
        else:
          detections[j+1][i] = all_bboxes[img_id][ind]
    return detections

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:  #all_boxes是字典，key是图片id，value是results
      for cls_ind in all_bboxes[image_id]:   #cls_ind : 1~80
        category_id = self.cocoCat_to_vocCat.get(self._valid_ids[cls_ind - 1],0)  # 如cls_ind - 1对应的coco类id不在voc id中，则返回0，记为背景；
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

  def run_eval_(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    os.system('python tools/reval.py ' + \
              '{}/results.json'.format(save_dir))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(save_dir,self.images,self.catIds,self.coco, coco_dets, "bbox",'pascal')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
  with open('/home/cm/path/to/clone/CenterNet_ROOT/exp/ctdet/coco_dla_meta_9/results.json','r') as f:
    results = json.load(f)
  print(results)