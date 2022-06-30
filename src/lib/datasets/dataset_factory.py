from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset,metaDataset,fewShotDataset,TwoStageDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.coco_meta import MetaCOCO,fewShotCOCO,valCOCO,classSpecFewShotCOCO
from .dataset.pascal import PascalVOC
from .dataset.pascal_meta import fewShotPascalVOC
from .dataset.kitti import KITTI



dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_meta':MetaCOCO,
  'coco_fewshot':fewShotCOCO,
  'coco_fewshot_increm':classSpecFewShotCOCO,
  'pascal_fewshot':fewShotPascalVOC
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'ctdet_meta':metaDataset,
  'few_shot':fewShotDataset,
  'ctdet_twostage':TwoStageDataset
}


def get_dataset(dataset, task):    #dataset,task会分别从dataset_factory和 _sample_factory中分别取出一个类，然后Dataset类会继承这两个类;而由于Datase中没有进行任何定义，因此其实就整合这两个类;
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
