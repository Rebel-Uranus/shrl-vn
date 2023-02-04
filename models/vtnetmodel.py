from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder

from torch.distributions import Categorical, Bernoulli


class VisualTransformer(Transformer):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(VisualTransformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed):
        bs, n, c = src.shape
        src = src.permute(1, 0, 2)
        query_embed = query_embed.permute(2, 0, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src)
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        return hs.transpose(0, 1), memory.permute(1, 2, 0).view(bs, c, n)


class HighPolicy(nn.Module):

    def __init__(self, args):
        super(HighPolicy, self).__init__()

        resnet_embedding_sz = 512
        self.lstm_input_sz = 640
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        self.hidden_state_sz = hidden_state_sz
        self.hidden_state_sz_high_tune = 64
        action_space = 6
        high_action_space = 4

        self.high_conv = nn.Conv2d(resnet_embedding_sz, 256, 1)
        self.high_visual_rep_embedding = nn.Sequential(
            nn.Linear(49, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.target_feature_embedding = nn.Sequential(
                nn.Linear(256, 248),
                nn.ReLU(),
            )
        
        self.high_embed_action = nn.Linear(high_action_space, 64)
        self.last_embed_action = nn.Linear(action_space, 64)
                
        self.high_target_embedding = nn.Linear(22,128)
    

        # high level controllor, policy_flag = 0
        self.high_lstm = nn.LSTM(self.lstm_input_sz, self.hidden_state_sz_high_tune, 2)
        self.high_critic_linear_1 = nn.Linear(self.hidden_state_sz_high_tune, 32)
        self.high_critic_linear_2 = nn.Linear(32, 1) # only this consider as Q func
        self.high_actor_linear = nn.Linear(self.hidden_state_sz_high_tune, high_action_space)

        self.terminations = nn.Sequential(
            nn.Linear(self.hidden_state_sz_high_tune, 32),
            nn.ReLU(),
            nn.Linear(32, high_action_space)
        )
        # network inital
        self.apply(weights_init)
        self.high_actor_linear.weight.data = norm_col_init(self.high_actor_linear.weight.data, 0.01)
        self.high_actor_linear.bias.data.fill_(0)
        self.high_critic_linear_1.weight.data = norm_col_init(self.high_critic_linear_1.weight.data, 1.0)
        self.high_critic_linear_1.bias.data.fill_(0)
        self.high_critic_linear_2.weight.data = norm_col_init(self.high_critic_linear_2.weight.data, 1.0)
        self.high_critic_linear_2.bias.data.fill_(0)
        self.high_lstm.bias_ih_l0.data.fill_(0)
        self.high_lstm.bias_ih_l1.data.fill_(0)
        self.high_lstm.bias_hh_l0.data.fill_(0)
        self.high_lstm.bias_hh_l1.data.fill_(0)
    
    def get_state(self, res_embedding, current_action, target, last_action = 0):
        img_feature = F.relu(self.high_conv(res_embedding))
        img_feature = img_feature.reshape(1,-1,49)
        target_feature = F.relu(self.high_target_embedding(target)).unsqueeze(dim=2)
        action_feature = F.relu(self.high_embed_action(current_action)).unsqueeze(dim=2)
        last_action_feature = F.relu(self.last_embed_action(last_action)).unsqueeze(dim=2)
        actions_feature = torch.cat((action_feature,last_action_feature),dim = -2)

        target_action = torch.cat((target_feature,actions_feature),dim = -2)
        feature = torch.cat((img_feature,target_action),dim = -1)
        feature = feature.permute(2, 0, 1)
        out = self.high_visual_rep_embedding(feature)
        out = out.reshape(1, -1)
        return out


    def predict_skill_termination(self, feature, current_skill, hx, cx):
        feature = feature.reshape([1, 1, -1])
        output, (hx, cx) = self.high_lstm(feature, (hx, cx))
        x = output.reshape([1, self.hidden_state_sz_high_tune])
        termination = self.terminations(x)[:, current_skill].sigmoid()
        Q= self.get_Q(x)
        next_skill = F.softmax(Q, dim=-1)
        return termination, next_skill, (hx, cx)
    
    def get_Q(self, x):
        critic_out = self.high_critic_linear_1(x)
        critic_out = self.high_critic_linear_2(critic_out)
        return critic_out
    
    def get_actor_critic(self, feature, current_skill, hx, cx):
        output, (hx, cx) = self.high_lstm(feature, (hx, cx))
        x = output.reshape([1, self.hidden_state_sz_high_tune])
        actor_out = self.high_actor_linear(x)
        critic_out = self.high_critic_linear_1(x)
        critic_out = self.high_critic_linear_2(critic_out)
        return actor_out, critic_out, (hx, cx)
    

class VTNetModel(nn.Module):

    def __init__(self, args):
        super(VTNetModel, self).__init__()
        # visual representation part
        # the networks used to process local visual representation should be replaced with linear transformer
        self.num_cate = args.num_category
        self.image_size = 300
        self.action_embedding_before = args.action_embedding_before
        self.detection_alg = args.detection_alg
        self.wo_location_enhancement = args.wo_location_enhancement

        # global visual representation learning networks
        resnet_embedding_sz = 512
        
        self.global_conv = nn.Conv2d(resnet_embedding_sz, 256, 1)
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)

        # previous action embedding networks
        action_space = args.action_space
        if not self.action_embedding_before:
            self.embed_action = nn.Linear(action_space, 64)
        else:
            self.embed_action = nn.Linear(action_space, 256)

        # local visual representation learning networks
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
        )

        self.visual_rep_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_rate)
        )

        # ==================================================
        # navigation policy part
        # this part should be fixed in this model
        self.lstm_input_sz = 3200
        hidden_state_sz = args.hidden_state_sz
        self.hidden_state_sz = hidden_state_sz
        self.hidden_state_sz_high_tune = 64
        self.lstm_input_sz_high_tune = 128
        low_action_space = 6
        # previous action embedding networks
        action_space = args.action_space
        if not self.action_embedding_before:

            self.seek_embed_action = nn.Linear(low_action_space, 64)
            self.tune_embed_action = nn.Linear(low_action_space, 64)
            self.avoid_embed_action = nn.Linear(low_action_space, 64)
        else:
            self.seek_embed_action = nn.Linear(low_action_space, 256)
            self.tune_embed_action = nn.Linear(low_action_space, 256)
            self.avoid_embed_action = nn.Linear(low_action_space, 256)

        self.detect_embedding = nn.Linear(6,64)
        self.tune_detec_embedding = nn.Sequential(
                nn.Linear(256, 250),
                nn.ReLU(),
            )
        self.tune_re_embed = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
            )
        self.tune_embedding = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        
        
        #low level seek, policy_flag = 1
        self.seek_lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        self.seek_critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.seek_critic_linear_2 = nn.Linear(64, 1)
        self.seek_actor_linear = nn.Linear(hidden_state_sz, low_action_space)
        #low level tune, policy_flag = 2
        self.tune_lstm = nn.LSTM(self.lstm_input_sz, self.hidden_state_sz, 2)
        self.tune_critic_linear_1 = nn.Linear(self.hidden_state_sz, 32)
        self.tune_critic_linear_2 = nn.Linear(32, 1)
        self.tune_actor_linear = nn.Linear(self.hidden_state_sz, low_action_space)


        #low level avoid, policy_flag = 3
        self.avoid_lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        self.avoid_critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.avoid_critic_linear_2 = nn.Linear(64, 1)
        self.avoid_actor_linear = nn.Linear(hidden_state_sz, low_action_space)

        self.high_model = HighPolicy(args)
        self.high_model_prime = deepcopy(self.high_model)

        # ==================================================
        # weights initialization
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.global_conv.weight.data.mul_(relu_gain)
        self.conv1.weight.data.mul_(relu_gain)

        

        self.seek_actor_linear.weight.data = norm_col_init(self.seek_actor_linear.weight.data, 0.01)
        self.seek_actor_linear.bias.data.fill_(0)
        self.seek_critic_linear_1.weight.data = norm_col_init(self.seek_critic_linear_1.weight.data, 1.0)
        self.seek_critic_linear_1.bias.data.fill_(0)
        self.seek_critic_linear_2.weight.data = norm_col_init(self.seek_critic_linear_2.weight.data, 1.0)
        self.seek_critic_linear_2.bias.data.fill_(0)
        self.seek_lstm.bias_ih_l0.data.fill_(0)
        self.seek_lstm.bias_ih_l1.data.fill_(0)
        self.seek_lstm.bias_hh_l0.data.fill_(0)
        self.seek_lstm.bias_hh_l1.data.fill_(0)

        self.tune_actor_linear.weight.data = norm_col_init(self.tune_actor_linear.weight.data, 0.01)
        self.tune_actor_linear.bias.data.fill_(0)
        self.tune_critic_linear_1.weight.data = norm_col_init(self.tune_critic_linear_1.weight.data, 1.0)
        self.tune_critic_linear_1.bias.data.fill_(0)
        self.tune_critic_linear_2.weight.data = norm_col_init(self.tune_critic_linear_2.weight.data, 1.0)
        self.tune_critic_linear_2.bias.data.fill_(0)
        self.tune_lstm.bias_ih_l0.data.fill_(0)
        self.tune_lstm.bias_ih_l1.data.fill_(0)
        self.tune_lstm.bias_hh_l0.data.fill_(0)
        self.tune_lstm.bias_hh_l1.data.fill_(0)

        self.avoid_actor_linear.weight.data = norm_col_init(self.avoid_actor_linear.weight.data, 0.01)
        self.avoid_actor_linear.bias.data.fill_(0)
        self.avoid_critic_linear_1.weight.data = norm_col_init(self.avoid_critic_linear_1.weight.data, 1.0)
        self.avoid_critic_linear_1.bias.data.fill_(0)
        self.avoid_critic_linear_2.weight.data = norm_col_init(self.avoid_critic_linear_2.weight.data, 1.0)
        self.avoid_critic_linear_2.bias.data.fill_(0)
        self.avoid_lstm.bias_ih_l0.data.fill_(0)
        self.avoid_lstm.bias_ih_l1.data.fill_(0)
        self.avoid_lstm.bias_hh_l0.data.fill_(0)
        self.avoid_lstm.bias_hh_l1.data.fill_(0)


        

    def embedding(self, state, detection_inputs, action_embedding_input, policy_flag):
        if self.wo_location_enhancement:
            detection_input_features = self.local_embedding(detection_inputs['features'].unsqueeze(dim=0)).squeeze(dim=0)
            detection_input = torch.cat((detection_input_features, detection_inputs['indicator']), dim=1).unsqueeze(dim=0)
        else:
            detection_input_features = self.local_embedding(detection_inputs['features'].unsqueeze(dim=0)).squeeze(dim=0)
            detection_input = torch.cat((
                detection_input_features,
                detection_inputs['labels'].unsqueeze(dim=1),
                detection_inputs['bboxes'] / self.image_size,
                detection_inputs['scores'].unsqueeze(dim=1),
                detection_inputs['indicator']
            ), dim=1).unsqueeze(dim=0)

        image_embedding = F.relu(self.global_conv(state))
        gpu_id = image_embedding.get_device()
        image_embedding = image_embedding + self.global_pos_embedding.cuda(gpu_id)
        image_embedding = image_embedding.reshape(1, -1, 49)

        if not self.action_embedding_before:
            visual_queries = image_embedding
            visual_representation, encoded_rep = self.visual_transformer(src=detection_input,
                                                                         query_embed=visual_queries)
            out = self.visual_rep_embedding(visual_representation)
            if policy_flag == 1:
                action_embedding = F.relu(self.seek_embed_action(action_embedding_input)).unsqueeze(dim=1)
            else:
                action_embedding = F.relu(self.tune_embed_action(action_embedding_input)).unsqueeze(dim=1)
            out = torch.cat((out, action_embedding), dim=1)
        else:
            action_embedding = F.relu(self.seek_embed_action(action_embedding_input)).unsqueeze(dim=2)
            visual_queries = torch.cat((image_embedding, action_embedding), dim=-1)
            visual_representation, encoded_rep = self.visual_transformer(src=detection_input,
                                                                         query_embed=visual_queries)
            out = self.visual_rep_embedding(visual_representation)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c, policy_flag=0):
        embedding = embedding.reshape([1, 1, -1])
        if policy_flag == 0:#high,target
            output, (hx, cx) = self.high_lstm(embedding, (prev_hidden_h, prev_hidden_c))
            x = output.reshape([1, self.hidden_state_sz_high_tune])
            actor_out = self.high_actor_linear(x)
            critic_out = self.high_critic_linear_1(x)
            critic_out = self.high_critic_linear_2(critic_out)
        elif policy_flag == 1:#seek,vtnet
            output, (hx, cx) = self.seek_lstm(embedding, (prev_hidden_h, prev_hidden_c))
            x = output.reshape([1, self.hidden_state_sz])
            actor_out = self.seek_actor_linear(x)
            critic_out = self.seek_critic_linear_1(x)
            critic_out = self.seek_critic_linear_2(critic_out)
        elif policy_flag == 2:#tune,target
            output, (hx, cx) = self.tune_lstm(embedding, (prev_hidden_h, prev_hidden_c))
            x = output.reshape([1, self.hidden_state_sz])
            actor_out = self.tune_actor_linear(x)
            critic_out = self.tune_critic_linear_1(x)
            critic_out = self.tune_critic_linear_2(critic_out)
        else:#avoid,resnet18
            output, (hx, cx) = self.avoid_lstm(embedding, (prev_hidden_h, prev_hidden_c))
            x = output.reshape([1, self.hidden_state_sz])
            actor_out = self.avoid_actor_linear(x)
            critic_out = self.avoid_critic_linear_1(x)
            critic_out = self.avoid_critic_linear_2(critic_out)
        

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options, policy_flag=0, gpu_id=-1):
        state = model_input.state
        (hx, cx) = model_input.hidden

        detection_inputs = model_input.detection_inputs
        action_probs = model_input.action_probs
        if policy_flag == 1:#seek,vtnet
            with torch.no_grad():
                x, image_embedding = self.embedding(state, detection_inputs, action_probs, policy_flag)
                x = x.reshape(1, -1)
            actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx, policy_flag)
        elif policy_flag == 2:#tune,target
            with torch.no_grad():
                x, image_embedding = self.embedding(state, detection_inputs, action_probs, policy_flag)
                x = x.reshape(1, -1)
            actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx, policy_flag)
        else:#avoid,resnet18
            image_embedding = F.relu(self.conv1(state))
            action_embedding = F.relu(self.avoid_embed_action(action_probs))
            x = torch.cat((image_embedding.reshape(1,-1), action_embedding), dim=-1)
            x = x.reshape(1, -1)
            actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx, policy_flag)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
            x=x
        )

    def high_forward(self, model_input, model_mode, model_options, policy_flag=0, gpu_id=-1):
        if model_mode == 'high_model':
            model = self.high_model
        else:
            model = self.high_model_prime
        
        #增添模型使不使用梯度的部分
        
        state = model_input.state
        (hhx, hcx) = model_input.high_hidden
        target = model_input.target
        action_probs = model_input.action_probs
        last_action_probs = model_input.last_action

        
        # 使用vtnet.detach() 作为主要表示
        detection_inputs = model_input.detection_inputs
        action_probs = model_input.action_probs
        index = 0
        max_scores = 0
        duplicate = torch.zeros(1).cuda(gpu_id)
        if model_input.target>0:
            duplicate[0] = 1
        for i in range(len(detection_inputs['scores'])):
            if detection_inputs['indicator'][i] == 1:
                if max_scores<detection_inputs['scores'][i]:
                    index = i
                    max_scores = detection_inputs['scores'][i]
        if index>=0:
            target_feature = model.target_feature_embedding(detection_inputs['features'][index].unsqueeze(dim=0)).squeeze(dim=0)
            detection_input = torch.cat((
                target_feature,
                detection_inputs['labels'].unsqueeze(dim=1)[index],
                detection_inputs['bboxes'][index] / self.image_size,
                detection_inputs['scores'].unsqueeze(dim=1)[index],
                detection_inputs['indicator'][index],
                duplicate
            ), dim=0).unsqueeze(dim=0)
        else:
            detection_input = torch.zeros((1,256)).cuda(gpu_id)
            if model_input.target>0:
                detection_input[0,-1] = 1
        
        img_embedding = F.relu(model.high_conv(state))
        high_action_embedding = F.relu(model.high_embed_action(action_probs))
        last_action_embedding = F.relu(model.last_embed_action(last_action_probs))
        img_embedding = img_embedding.reshape(1,256,49)
        img_embedding = model.high_visual_rep_embedding(img_embedding)
        img_embedding = img_embedding.reshape(1,-1)
        x = torch.cat((detection_input, img_embedding, last_action_embedding, high_action_embedding),dim=1)

        x = x.reshape([1, 1, -1])
        current_skill = (policy_flag-1)%4

        actor_out, critic_out, (hhx, hcx) = model.get_actor_critic(x, current_skill, hhx, hcx)

        return ModelOutput(
            high_action=actor_out,
            value=critic_out,
            high_hidden = (hhx, hcx)
        )
        

def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)
    # dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / c_pos_embedding)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    #there may be a problem in the following lines
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos