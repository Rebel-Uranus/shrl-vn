import os
import json
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from datasets.constants import AI2THOR_TARGET_CLASSES

class PreVisTranfsDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.data_dir = args.data_dir
        self.detection_alg = args.detection_alg

        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[60]) if item in AI2THOR_TARGET_CLASSES[22]]

        self.annotation_file = os.path.join(self.data_dir, 'annotation_{}.json'.format(data_type))
        with open(self.annotation_file, 'r') as rf:
            self.annotations = json.load(rf)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        location = annotation['location']
        target = annotation['target']
        optimal_action = annotation['optimal_action']

        annotation_path = os.path.join(self.data_dir, 'seg', '{}.npz'.format(location))

        data = np.load(annotation_path)

        seg_feature = data['seg_feature']
        global_feature = data['resnet18_feature']

        if self.detection_alg == 'detr':
            features = data['detr_feature'][:, :256]
            scores = data['detr_feature'][:, 256]
            labels = data['detr_feature'][:, 257]
            bboxes = data['detr_feature'][:, 260:]

            # generate target indicator array based on detection results labels
            target_embedding_array = np.zeros((data['detr_feature'].shape[0], 1))
            target_embedding_array[labels[:] == (AI2THOR_TARGET_CLASSES[22].index(target) + 1)] = 1

        elif self.detection_alg == 'fasterrcnn':
            features = data['fasterrcnn_feature'][:, :512]
            bboxes = data['fasterrcnn_feature'][:, 512:516]
            scores = data['fasterrcnn_feature'][:, 516]
            labels = np.arange(data['fasterrcnn_feature'].shape[0])

            target_embedding_array = np.zeros((data['fasterrcnn_feature'].shape[0], 1))
            target_embedding_array[AI2THOR_TARGET_CLASSES[22].index(target)] = 1

        elif self.detection_alg == 'fasterrcnn_bottom':
            load_feature = data['fasterrcnn_feature'][self.targets_index, :]
            features = load_feature[:, :256]
            bboxes = load_feature[:, 256:260]
            scores = load_feature[:, 260]
            labels = np.arange(load_feature.shape[0])

            target_embedding_array = np.zeros((load_feature.shape[0], 1))
            target_embedding_array[AI2THOR_TARGET_CLASSES[22].index(target)] = 1

        
        #pred.hdf5 index 是从0开始的
        for i in range(100):
            if target_embedding_array[i] == 1:
                bound = bboxes[i]
                seg_feature[self.clamp_number(int(bound[0]),0,300):self.clamp_number(int(bound[2]),0,300), self.clamp_number(int(bound[1]),0,300):self.clamp_number(int(bound[3]),0,300)]=150
                
                        
        local_feature = {
            'features': features,
            'seg_features': seg_feature,
            'scores': scores,
            'labels': labels,
            'bboxes': bboxes,
            'indicator': target_embedding_array,
            'locations': location,
            'targets': target,
            'idx': idx,
        }    

        return global_feature, local_feature, optimal_action

   
    def clamp_number(self,num,a,b):
        return max(min(num, max(a, b)), min(a, b))

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = False
    return



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
