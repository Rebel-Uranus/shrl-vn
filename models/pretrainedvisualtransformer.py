from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder
from .vtnetmodel import VisualTransformer, get_gloabal_pos_embedding


class PreTrainedVisualTransformer(nn.Module):
    def __init__(self, args):
        super(PreTrainedVisualTransformer, self).__init__()
        self.image_size = 300
        self.detection_alg = args.detection_alg
        self.wo_location_enhancement = args.wo_location_enhancement

        # same layers as VisualTransformer visual representation learning part
        self.global_conv = nn.Conv2d(512, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)

        if self.detection_alg == 'detr' and not self.wo_location_enhancement:
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 249),
                nn.ReLU(),
            )
        elif self.detection_alg == 'detr' and self.wo_location_enhancement:
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 255),
                nn.ReLU(),
            )
        elif self.detection_alg == 'fasterrcnn':
            self.local_embedding = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 249),
                nn.ReLU(),
            )
        elif self.detection_alg == 'fasterrcnn_bottom':
            self.local_embedding = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 249),
                nn.ReLU(),
            )

        self.visual_transformer = VisualTransformer(
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout_rate,
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_rate)
        )

        # pretraining network action predictor, should be used in Visual Transformer model
        self.pretrain_fc = nn.Linear(3136, 6)
        #avoid feature pretrian
        self.avoid_query_conv = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(1,1,kernel_size=4,stride=2),
            nn.ReLU()
        )
        self.avoid_key_conv = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(1,1,kernel_size=4,stride=2),
            nn.ReLU()
        )
        self.avoid_value_conv = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=3,stride=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(1,1,kernel_size=4,stride=2),
            nn.ReLU()
        )
        # self.avoid_p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        # self.avoid_c2 = nn.Conv2d(1,1,kernel_size=4,stride=2)
        self.avoid_mlp = nn.Linear(576, 6)
        self.avoid_pos_embedding = get_seg_pos_embedding(24,24)
        self.avoid_softmax = nn.Softmax(dim=2)
        self.avoid_pretain = True

    def forward(self, global_feature: torch.Tensor, local_feature: dict):
        batch_size = global_feature.shape[0]
        
        if not self.avoid_pretain:

            global_feature = global_feature.squeeze(dim=1)
            image_embedding = F.relu(self.global_conv(global_feature))
            image_embedding = image_embedding + self.global_pos_embedding.repeat([batch_size, 1, 1, 1]).cuda()
            image_embedding = image_embedding.reshape(batch_size, -1, 49)

            if self.wo_location_enhancement:
                detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
                local_input = torch.cat((detection_input_features, local_feature['indicator']), dim=2)
            else:
                detection_input_features = self.local_embedding(local_feature['features'].unsqueeze(dim=0)).squeeze(dim=0)
                local_input = torch.cat((
                    detection_input_features,
                    local_feature['labels'].unsqueeze(dim=2),
                    local_feature['bboxes'] / self.image_size,
                    local_feature['scores'].unsqueeze(dim=2),
                    local_feature['indicator']
                ), dim=2)

            visual_representation, _ = self.visual_transformer(src=local_input, query_embed=image_embedding)

            visual_rep = self.visual_rep_embedding(visual_representation)
            visual_rep = visual_rep.reshape(batch_size, -1)

            action = self.pretrain_fc(visual_rep)
        else:
            seg_feature = local_feature['seg_features']
            x = seg_feature.unsqueeze(dim=1)
            gpu_id = x.get_device()
            pos_embed = self.avoid_pos_embedding.repeat([batch_size, 1, 1]).cuda(gpu_id)
            proj_query  = self.avoid_query_conv(x).view(batch_size,-1,24*24)
            proj_query = (proj_query + pos_embed).permute(0,2,1) # B X CX(N)
            proj_key =  self.avoid_key_conv(x).view(batch_size,-1,24*24) # B X C x (*W*H)
            proj_key = proj_key + pos_embed
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.avoid_softmax(energy) # BX (N) X (N) 
            proj_value = self.avoid_value_conv(x).view(batch_size,-1,24*24) # B X C X N

            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            # print(out.shape)
            out = out.view(batch_size,-1)
            action = self.avoid_mlp(out)
        return {
            'action': action,
            'fc_weights': self.pretrain_fc.weight,
            # 'visual_reps': visual_rep.reshape(batch_size, 64, 49)
        }

def get_seg_pos_embedding(size_feature_map, c_pos_embedding):
    # size_feature_map = 24, c_pos_embedding = 24
    d = size_feature_map*size_feature_map
    mask = torch.ones(1, )

    # y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(0, dtype=torch.float32)

    dim_t = torch.arange(d, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / d)

    pos_x = x_embed[:, None] / dim_t
    # pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    # pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos_x