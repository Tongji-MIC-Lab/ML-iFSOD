import os
import torch
import sys



def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    


from opts import opts

from models.model import *
from models.networks.pose_dla_dcn import *
from datasets.dataset_factory import get_dataset



opt = opts().parse()

Dataset = get_dataset(opt.dataset,opt.task)
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
fs_model = get_fs_net(34,opt.heads,opt.head_conv)
model = get_meta_net(34,opt.heads,opt,opt.head_conv)
model.state_dict()['learner.reg.2.weight']
fs = load_fs_net(fs_model, model.state_dict())

#print(model.state_dict()['learner.reg.2.weight'] == fs.state_dict()['reg.2.weight'])
fs_stat = fs.state_dict()
stat = {key:fs_stat[key] for key in fs_stat if key.startswith('reg') or key.startswith('wh')}
for w in stat:
    print(w)
#print(fs.state_dict().keys())