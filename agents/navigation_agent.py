import torch
import numpy as np
import h5py
import os

from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput

from .agent import ThorAgent


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, scenes, targets, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, scenes, targets, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.keep_ori_obs = args.keep_ori_obs

        self.glove = {}
        if 'SP' in self.model_name:
            with h5py.File(os.path.expanduser('~/wangshuo/Code/vn/glove_map300d.hdf5'), 'r') as rf:
                for i in rf:
                    self.glove[i] = rf[i][:]

        self.detection_alg = args.detection_alg
        if self.detection_alg == 'fasterrcnn':
            self.detection_processer = self.process_faster_input
        elif self.detection_alg == 'fasterrcnn_bottom':
            self.detection_processer = self.process_faster_bottom_input
        elif self.detection_alg == 'detr':
            self.detection_processer = self.process_detr_input
        self.detr_padding = args.detr_padding

    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame
        if self.policy_flag == 0:
            model_input.hidden = self.seek_hidden
            model_input.action_probs = self.seek_last_action_probs
        elif self.policy_flag == 1:
            model_input.hidden = self.seek_hidden
            model_input.action_probs = self.seek_last_action_probs
        elif self.policy_flag == 2:
            model_input.hidden = self.tune_hidden
            model_input.action_probs = self.tune_last_action_probs
        else:
            model_input.hidden = self.avoid_hidden
            model_input.action_probs = self.avoid_last_action_probs

        model_input = self.detection_processer(model_input)

        

        if 'SP' in self.model_name:
            model_input.glove = self.glove[self.episode.target_object]

        if 'Memory' in self.model_name:
            if self.model_name.startswith('VR'):
                state_length = 64 * 7 * 7
            else:
                state_length = self.hidden_state_sz

            if len(self.episode.state_reps) == 0:
                model_input.states_rep = torch.zeros(1, state_length)
            else:
                model_input.states_rep = torch.stack(self.episode.state_reps)

            if self.keep_ori_obs:
                if len(self.episode.obs_reps) == 0:
                    self.episode.obs_reps.append(torch.zeros(3136))
                model_input.obs_reps = torch.stack(self.episode.obs_reps)
            else:
                if 'State' in self.model_name:
                    dim_obs = 3136
                else:
                    dim_obs = 512
                if len(self.episode.obs_reps) == 0:
                    model_input.obs_reps = torch.zeros(1, dim_obs)
                else:
                    model_input.obs_reps = torch.stack(self.episode.obs_reps)

            if len(self.episode.state_memory) == 0:
                model_input.states_memory = torch.zeros(1, state_length)
            else:
                model_input.states_memory = torch.stack(self.episode.state_memory)

            if len(self.episode.action_memory) == 0:
                model_input.action_memory = torch.zeros(1, 6)
            else:
                model_input.action_memory = torch.stack(self.episode.action_memory)

            model_input.states_rep = toFloatTensor(model_input.states_rep, self.gpu_id)
            model_input.states_memory = toFloatTensor(model_input.states_memory, self.gpu_id)
            model_input.action_memory = toFloatTensor(model_input.action_memory, self.gpu_id)
            model_input.obs_reps = toFloatTensor(model_input.obs_reps, self.gpu_id)

        return model_input, self.model.forward(model_input, model_options, self.policy_flag, self.gpu_id)
    
    def avoid_eval_at_state(self, model_options):
        model_input = ModelInput()
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame
        model_input.hidden = self.avoid_hidden
        model_input.action_probs = self.avoid_last_action_probs
        model_input = self.detection_processer(model_input)
        # model inputs

        return model_input, self.model.forward(model_input, model_options, 3, self.gpu_id)
    
    def high_eval_at_state(self, model_options, model_mode='high_model'):
        model_input = ModelInput()
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        if model_mode == 'high_model':
            model_input.high_hidden = self.high_hidden
        else:
            model_input.high_hidden = self.high_hidden_prime

        model_input = self.detection_processer(model_input)
        model_input.action_probs = self.high_last_action_probs
        # model inputs
        with torch.cuda.device(self.gpu_id):
            index = self.targets.index(self.episode.target_object)
            target = torch.zeros((1,22)).cuda()
            target[0,index] = 1
        model_input.target = self.duplicate_flag[-1]
        model_input.last_action = self.low_last_action_probs.detach()
        return model_input, self.model.high_forward(model_input, model_mode, model_options, 0, self.gpu_id)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        if 'SingleLayerLSTM' not in self.model_name:
            with torch.cuda.device(self.gpu_id):
                self.seek_hidden = (
                    torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                    torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                )
                self.high_hidden = (
                    torch.zeros(2, 1, 64).cuda(),
                    torch.zeros(2, 1, 64).cuda(),
                )
                self.high_hidden_prime = (
                    torch.zeros(2, 1, 64).cuda(),
                    torch.zeros(2, 1, 64).cuda(),
                )
                self.tune_hidden = (
                    torch.zeros(2, 1,  self.hidden_state_sz).cuda(),
                    torch.zeros(2, 1,  self.hidden_state_sz).cuda(),
                )
                self.avoid_hidden = (
                    torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                    torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                )
        else:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (
                    torch.zeros(1, 1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, 1, self.hidden_state_sz).cuda(),
                )
                
        self.seek_last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )
        self.tune_last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )
        self.avoid_last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )
        self.high_last_action_probs = gpuify(
            torch.zeros((1, 4)), self.gpu_id
        )
        self.low_last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

    def repackage_hidden(self):
        self.seek_hidden = (self.seek_hidden[0].detach(), self.seek_hidden[1].detach())
        self.tune_hidden = (self.tune_hidden[0].detach(), self.tune_hidden[1].detach())
        self.high_hidden = (self.high_hidden[0].detach(), self.high_hidden[1].detach())
        self.high_hidden_prime = (self.high_hidden_prime[0].detach(), self.high_hidden_prime[1].detach())
        self.avoid_hidden = (self.avoid_hidden[0].detach(), self.avoid_hidden[1].detach())
        self.seek_last_action_probs = self.seek_last_action_probs.detach()
        self.tune_last_action_probs = self.tune_last_action_probs.detach()
        self.avoid_last_action_probs = self.avoid_last_action_probs.detach()
        self.high_last_action_probs = self.high_last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def process_detr_input(self, model_input):
        # process detection features from DETR detector
        current_detection_feature = self.episode.current_detection_feature()

        zero_detect_feats = np.zeros_like(current_detection_feature)
        ind = 0
        for cate_id in range(len(self.targets) + 2):
            cate_index = current_detection_feature[:, 257] == cate_id
            if cate_index.sum() > 0:
                index = current_detection_feature[cate_index, 256].argmax(0)
                zero_detect_feats[ind, :] = current_detection_feature[cate_index, :][index]
                ind += 1
        current_detection_feature = zero_detect_feats

        if self.detr_padding:
            current_detection_feature[current_detection_feature[:, 257] == (len(self.targets) + 1)] = 0

        detection_inputs = {
            'features': current_detection_feature[:, :256],
            'scores': current_detection_feature[:, 256],
            'labels': current_detection_feature[:, 257],
            'bboxes': current_detection_feature[:, 260:],
            'target': self.targets.index(self.episode.target_object),
        }

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((detection_inputs['features'].shape[0], 1))
        target_embedding_array[
            detection_inputs['labels'][:] == (self.targets.index(self.episode.target_object) + 1)] = 1
        detection_inputs['indicator'] = target_embedding_array

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.detection_inputs = detection_inputs

        return model_input

    def process_faster_input(self, model_input):
        # process detection features from faster-RCNN detector
        current_detection_feature = self.episode.current_detection_feature()[:]

        self.episode.detection_results.append(
            list(current_detection_feature[self.targets.index(self.episode.target_object), 512:]))

        detection_inputs = {
            'features': current_detection_feature[:, :512],
            'scores': current_detection_feature[:, 516],
            'labels': np.arange(current_detection_feature.shape[0]),
            'bboxes': current_detection_feature[:, 512:516],
            'target': self.targets.index(self.episode.target_object),
        }

        target_embedding_array = np.zeros((len(self.targets), 1))
        target_embedding_array[self.targets.index(self.episode.target_object)] = 1
        detection_inputs['indicator'] = target_embedding_array

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.detection_inputs = detection_inputs

        return model_input

    def process_faster_bottom_input(self, model_input):
        # process detection features from faster-RCNN detector
        current_detection_feature = self.episode.current_detection_feature()[:]
        current_detection_feature = current_detection_feature[self.targets_index, :]

        self.episode.detection_results.append(
            list(current_detection_feature[self.targets.index(self.episode.target_object), 256:]))

        detection_inputs = {
            'features': current_detection_feature[:, :256],
            'scores': current_detection_feature[:, 260],
            'labels': np.arange(current_detection_feature.shape[0]),
            'bboxes': current_detection_feature[:, 256:260],
            'target': self.targets.index(self.episode.target_object),
        }

        target_embedding_array = np.zeros((len(self.targets), 1))
        target_embedding_array[self.targets.index(self.episode.target_object)] = 1
        detection_inputs['indicator'] = target_embedding_array

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.detection_inputs = detection_inputs

        return model_input

    def dict_toFloatTensor(self, dict_input):
        '''Convert all values in dict_input to float tensor

        '''
        for key in dict_input:
            dict_input[key] = toFloatTensor(dict_input[key], self.gpu_id)

        return dict_input