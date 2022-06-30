import torch
import torch.nn as nn
import os
import sys
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
model_path =os.path.join(os.path.dirname(__file__),'src','lib','models')
add_path(model_path)
from networks.resnet_dcn import get_meta_net,get_fs_net
from networks.pose_dla_dcn import Learner

learner_model = '/home/cm/path/to/clone/CenterNet_ROOT/exp/ctdet_meta/coco_resnet_meta_4/model_last.pth'
ft_model = '/home/cm/path/to/clone/CenterNet_ROOT/exp/ctdet/coco_resdcn101_base_256_1/model_best.pth'
hm_model = '/home/cm/path/to/clone/CenterNet_ROOT/exp/ctdet_meta/coco_dla_meta/hm_head/hm_head_0.pth'
entire_model = '/home/cm/path/to/clone/CenterNet_ROOT/exp/ctdet_meta/coco_resnet_meta_7/entire_model/entire_model_last.pth'

annot_path = '/home/cm/path/to/clone/CenterNet_ROOT/data/voc/annotations/pascal_test2007.json'

base_inds = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
#m1 = torch.load(ft_model)['state_dict']
#print(m1['hm.2.weight'].shape)
#for key in m1['state_dict']:
    #print(key)

#m2 = torch.load(learner_model)
#print(m2['state_dict']['learner.hm.0.bias'][:10])
#print(m2['optimizer']['param_groups'][0]['lr'])
def backforward_hook(module,grad_input,grad_output):
    #print('module:{}'.format(module))
    print('grad_input:{}'.format(grad_input[1].shape))
    grad_input = list(grad_input)
    mask = torch.zeros_like(grad_input[1])
    mask[l] = 1
    grad_input[1] *= mask.float()
    #print('grad_input:{}'.format(grad_input))
    #print('grad_output:{}'.format(grad_output))
    return tuple(grad_input)

heads = {'hm': 80,'wh': 2,'reg': 2}
fs_model = get_fs_net(101,heads,256)
fs_model.hm[2].register_backward_hook(backforward_hook)
m4 = torch.load(entire_model)['state_dict']
"""for k in m4:
    if k.startswith('hm.2'):
        print(m4[k].shape)
        print(m1[k].shape)
        m4[k][base_inds,...] = m1[k][base_inds,...]
        print(m4[k][7] == m1[k][7])"""

l = [0,2]
model = nn.Conv2d(3,5,1)
model.register_backward_hook(backforward_hook)

#print(dir(model))

for p in model.parameters():
    #p[0].detach()
    print(p.shape)

x = torch.rand(1,3,5,5)
target = torch.rand(1,5,5,5)

flag = 1
if flag == 1:

    optimizer = torch.optim.SGD(model.parameters(),1)
    y = model(x)
    MSE = nn.MSELoss()
    loss = MSE(y,target)
    #print(loss)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
"""

    #for p in model.parameters():
        #print(p)
        

#print(m4['hm.0.bias'][:10])
#print(m1['state_dict']['hm.2.bias'])
#print(m2['state_dict']['learner.hm.2.bias'].shape)
#print(m4['hm.2.bias'].shape)




heads = {'hm':80,'wh':2,'reg':2}
head_conv = 256
final_kernel = 1
channels = [16, 32, 64, 128, 256, 512]
first_level = int(np.log2(4))
m = Learner(heads,head_conv,final_kernel,first_level,channels)


optimizer = torch.optim.Adam(m.parameters(),0.5)

optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2

print(optimizer.param_groups[0]['lr'])"""

#coco = coco.COCO(annot_path)
#
#print(coco.getCatIds())