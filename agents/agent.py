""" Base class for all Agents. """
from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import random
import copy

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datasets.constants import DONE_ACTION_INT, AI2THOR_TARGET_CLASSES
from models.model_io import ModelInput, ModelOutput
from utils.model_util import gpuify, toFloatTensor
from torch.distributions import Categorical, Bernoulli


class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, rank, scenes, targets, episode=None, max_episode_length=1e3, gpu_id=-1
    ):
        self.scenes = scenes
        self.targets = targets
        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[22]) if item in self.targets]

        self.gpu_id = gpu_id

        self._model = None
        self.model = model
        self._episode = episode
        self.eps_len = 0
        self.true_eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.max_length = False
        self.hidden = None
        self.actions = []
        self.probs = []
        self.last_action_probs = None
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.max_episode_length = max_episode_length
        self.success = False
        self.backprop_t = 0
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

        self.verbose = args.verbose
        # self.learned_loss = args.learned_loss
        self.learned_input = None
        self.learned_t = 0
        self.num_steps = args.num_steps
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space

        self.targets_types = None
        self.model_name = args.model

        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []

        self.meta_duplicate_action = False
        self.meta_failed_action = False

        self.memory_duplicate_learning = False

        self.duplicate_states_actions = {}

        self.imitation_learning = False
        self.il_duplicate_action = False
        self.il_failed_action = False
        self.il_update_actions = {}

        self.record_attention = args.record_attention
        self.last_state_rep = ModelOutput()
        # self.contrastive_learning = args.contrastive_learning
        #HRL parameters
        self.policy_flag = 0
        self.high_reward = 0

        self.high_last_index = -1
        self.high_hidden = None
        self.high_hidden_prime = None
        self.high_last_action_probs = None
        self.low_last_action_probs = None
        self.high_entropies = []
        self.high_values = []
        self.high_log_probs = []
        self.high_rewards = []

        self.next_skill_term_probs = []
        self.next_Q_primes = []
        self.skills = []
        self.skill_term_probs = []
        self.Q = []

        self.pretrain_Q = []
        self.pretrain_term = []

        self.seek_last_action_probs = None
        self.seek_hidden = None
        self.seek_entropies = []
        self.seek_values = []
        self.seek_log_probs = []
        self.seek_rewards = []

        self.tune_last_action_probs = None
        self.tune_hidden = None
        self.tune_entropies = []
        self.tune_values = []
        self.tune_log_probs = []
        self.tune_rewards = []

        self.avoid_start_flag = 0
        self.avoid_hidden = None
        self.avoid_last_action_probs = None
        self.avoid_entropies = []
        self.avoid_values = []
        self.avoid_log_probs = []
        self.avoid_rewards = []

        self.high_records = [[],[],[],[]]
        self.low_done_count = 0
        self.low_skills = copy.deepcopy(args.low_skills) 
        self.low_skills_bak = copy.deepcopy(args.low_skills) 

        self.random_avoid = [[1,0,5],[2,0,5]]
        
        self.avoid_policy = self.random_avoid[np.random.randint(2)].copy()
        self.state_buffer = []
        self.window = 8
        self.avoid_count = 0
        self.lost_window = 4
        self.lost_target = [0] *self.lost_window
        self.lost_index = 0
        self.lost_flag = False
        self.teminate_actomatical = False
        self.avoidance = False
        self.last_policy_flag = -1
        self.last_state = ''
        self.high_num = 0
        self.avoid_num = 0
        self.low_pretrain = args.low_pretrain
        self.high_pretrain = args.high_pretrain

        self.termination = True
        self.current_skill = 0
        self.next_skill = 0
        self.episode_start_flag = True
        self.change_skill = True

        self.term_change_reg = 0.01
        self.term_skill_inconsistent_reg = 0.01
        self.duplicate_states_num = 0
        self.delay = 0.01 / 100000
        self.duplicate_flag = []
        

        with torch.cuda.device(self.gpu_id):
            self.seek_adj_hidden = (
                                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                            )
        self.seek_adj_probs = gpuify(
                        torch.zeros((1, self.action_space)), self.gpu_id
                    )

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, model_options):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self.gpu_id >= 0 and self._model is not None:
            with torch.cuda.device(self.gpu_id):
                self._model = self.model.cuda()

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def action(self, model_options, training, test_update):
        """ Train the agent. """
        low_pretrain = self.low_pretrain
        high_pretrain = self.high_pretrain
        
        pretrain_termination = False
        if self.episode_start_flag:
            self.state_buffer.append(str(self.episode.environment.controller.state))
            self.duplicate_flag.append(0)

        if training or test_update:
            self.model.train()
        else:
            self.model.eval()
        
        self.episode.states.append(str(self.episode.environment.controller.state))
       
        # clc Q & term for state to prepare for loss update
        if training or test_update:
            high_model_input, high_out = self.high_eval_at_state(model_options, 'high_model')
        else:
            with torch.no_grad():
                high_model_input, high_out = self.high_eval_at_state(model_options, 'high_model')
  
        self.high_hidden = high_out.high_hidden
        high_prob = F.softmax(high_out.high_action, dim=1)
        if training:
            next_skill = high_prob.multinomial(1).data
        else:
            next_skill = high_prob.argmax(dim=1, keepdim=True)
        high_log_prob = F.log_softmax(high_out.high_action, dim=1)
        high_entropy = -(high_log_prob * high_prob).sum(1)
        high_log_prob = high_log_prob.gather(1, Variable(next_skill))

        if not high_pretrain and not low_pretrain:
            self.next_skill = next_skill
        if (high_pretrain or low_pretrain) and ((not self.avoidance and self.current_skill!=2) or self.episode_start_flag):
            if self.termination:
                self.next_skill = self.low_skills.pop(0)
                self.termination = False
                
        if self.current_skill != self.next_skill or self.episode_start_flag:
            self.termination = True
        else:
            self.termination = False
        
        if self.termination:
            
            if not self.episode_start_flag:
                self.change_skill = not self.change_skill
            self.episode_start_flag = False
            self.current_skill = self.next_skill
            self.policy_flag = (self.current_skill + 1) % 4
            self.termination = False
            if self.change_skill:
                self.change_skill = not self.change_skill
                if self.policy_flag == 1:
                    self.seek_hidden = (self.seek_adj_hidden[0].detach(),self.seek_adj_hidden[1].detach())
                    self.seek_last_action_probs = self.seek_adj_probs.detach()
                    self.seek_entropies.append([])
                    self.seek_values.append([])
                    self.seek_log_probs.append([])
                    self.seek_rewards.append([])
                    self.avoidance = False
                    self.avoid_count = 0
                elif self.policy_flag == 2:
                    self.episode.last_target = None
                    self.episode.tune_step = 0
                    self.episode.tune_areas = []
                    self.episode.tune_states = set()

                    self.tune_hidden = (self.seek_adj_hidden[0].detach(),self.seek_adj_hidden[1].detach())
                    self.tune_last_action_probs = self.seek_adj_probs.detach()
                    self.tune_entropies.append([])
                    self.tune_values.append([])
                    self.tune_log_probs.append([])
                    self.tune_rewards.append([])
                    self.avoidance = False
                    self.avoid_count = 0
                elif self.policy_flag == 3:
                    self.avoid_entropies.append([])
                    self.avoid_values.append([])
                    self.avoid_log_probs.append([])
                    self.avoid_rewards.append([])
                    self.avoid_start_flag = True 
                    self.avoid_policy = self.random_avoid[np.random.randint(2)].copy()
                    self.avoidance = True
                    self.avoid_count = 0
            if self.policy_flag == 0:
                action = torch.tensor([[3]])
                prob = gpuify(torch.Tensor([[0,0,0,1,0,0]]), self.gpu_id)
                
        if self.policy_flag!=0:
            if low_pretrain and (training or test_update):
                model_input, out = self.eval_at_state(model_options)
            else:
                with torch.no_grad():
                    model_input, out = self.eval_at_state(model_options)

            # agent operates the asked action
            prob = F.softmax(out.logit, dim=1)
            self.episode.action_outputs.append(prob.tolist())
        
            if training:
                action = prob.multinomial(1).data
            else:
                action = prob.argmax(dim=1, keepdim=True)
            
            log_prob = F.log_softmax(out.logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            log_prob_prime = log_prob
            log_prob = log_prob.gather(1, Variable(action))
            if (high_pretrain or low_pretrain) and action[0,0] == 5:
                pretrain_termination = True
        else:
            model_input = ModelInput()
        
       
        if self.policy_flag == 3:
            
            if self.avoidance:
                if len(self.state_buffer)>1 and str(self.episode.environment.controller.state) == self.state_buffer[-2]:
                    ret = random.uniform(0.0, 1.0)
                    if ret <=0.5:
                        action[0,0] = 1
                        self.avoid_count += 1
                        prob = gpuify(torch.Tensor([[0,0.5,0.5,0,0,0]]), self.gpu_id)
                        
                    elif ret>0.5:
                        action[0,0] = 2
                        self.avoid_count += 1
                        prob = gpuify(torch.Tensor([[0,0.5,0.5,0,0,0]]), self.gpu_id)
                else:
                    action[0,0] = 0
                    self.avoid_count += 1
                    prob = gpuify(torch.Tensor([[1,0,0,0,0,0]]), self.gpu_id)

            if self.avoid_count>3:
                action[0,0] = 5
                prob = gpuify(torch.Tensor([[0,0,0,0,0,1]]), self.gpu_id)
                pretrain_termination = True
                if high_pretrain:
                    self.next_skill = 1
            
                    
        self.reward, self.done, self.info, self.high_reward = self.episode.step(action[0, 0], self.policy_flag, self.high_last_index, model_input, self.avoid_start_flag)
        
        if int(action[0,0]) !=5 and not self.avoidance and self.avoid_count == 0 and (high_pretrain and self.policy_flag==2): 
            if self.window >=  len(self.state_buffer):
                window = 0
            else:
                window = -1 * self.window
            if len(self.state_buffer)>0 and str(self.episode.environment.controller.state) == self.state_buffer[-1]:
                self.avoidance = True
                self.avoid_count = 0
                if high_pretrain:
                    if self.current_skill != 2:
                        pretrain_termination = True  
                    self.next_skill = 2
        if str(self.episode.environment.controller.state) == self.state_buffer[-1]:
            self.duplicate_flag.append(1)
        else:
            self.duplicate_flag.append(0)
        if action[0,0] != 5 and self.policy_flag!=0:
            if str(self.episode.environment.controller.state) in self.state_buffer and self.policy_flag != 3:
                self.duplicate_states_num += 1
            
            self.state_buffer.append(str(self.episode.environment.controller.state))
        

        self.high_last_action_probs = high_prob
        self.low_last_action_probs = prob
        
        self.episode.prev_frame = model_input.state
        self.episode.current_frame = self.state()
        
        self.high_records[0].append(self.termination)
        if high_pretrain or low_pretrain:
            self.high_records[1].append(self.policy_flag)
            self.high_records[2].append(action.cpu().detach().numpy().tolist()[0][0])
        else:
            self.high_records[1].append(self.policy_flag.cpu().detach().numpy().tolist())
            self.high_records[2].append(action.cpu().detach().numpy().tolist())

        
        if high_pretrain or low_pretrain:
            if action[0,0] == 5 or pretrain_termination:
                self.termination = True
            else:
                self.termination = False
        
        if self.policy_flag == 2:
            self.tune_hidden = out.hidden
            self.seek_adj_hidden = out.hidden
        elif self.policy_flag == 1:
            self.seek_hidden = out.hidden
            self.seek_adj_hidden = out.hidden

        # record the actions those should be used to compute the loss
        self.skills.append(self.current_skill)
        self.Q.append(high_out.high_action)

        self.high_entropies.append(high_entropy)
        self.high_values.append(high_out.value)
        self.high_log_probs.append(high_log_prob)
        self.high_rewards.append(self.high_reward)

                
        if self.policy_flag == 1:
            self.seek_last_action_probs = prob
            self.seek_adj_probs = prob
            self.seek_entropies[-1].append(entropy)
            self.seek_values[-1].append(out.value)
            self.seek_log_probs[-1].append(log_prob)
            self.seek_rewards[-1].append(self.reward)
        elif self.policy_flag == 2:
            self.tune_last_action_probs = prob
            self.seek_adj_probs = prob
            self.tune_entropies[-1].append(entropy)
            self.tune_values[-1].append(out.value)
            self.tune_log_probs[-1].append(log_prob)
            self.tune_rewards[-1].append(self.reward)
        elif self.policy_flag == 3:
            self.avoid_start_flag = False
        if self.policy_flag!=0:
            self.probs.append(prob)
            
            
            self.episode.action_probs.append(prob)
            self.episode.actions_record.append(action)
        self.actions.append(action)

        
        if (self.policy_flag != 0 and action[0,0] != 5) or (self.policy_flag == 0 and action[0,0] == 3):
            self.true_eps_len += 1
        self._increment_episode_length()

        if self.episode.strict_done and self.policy_flag == 0 and int(action[0,0])==3:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length
        
        
        if int(action[0,0]) == 5 and self.policy_flag!=0:
            self.low_done_count += 1
        
        

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.probs = []
        self.reward = 0
        self.backprop_t = 0
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.learned_input = None
        self.learned_t = 0
        self.il_update_actions = {}
        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []
        self.duplicate_states_actions = {}

        # HRL parameters
        self.policy_flag = 0
        self.high_reward = 0

        self.high_last_index = -1
        self.high_entropies = []
        self.high_values = []
        self.high_log_probs = []
        self.high_rewards = []

        self.next_skill_term_probs = []
        self.next_Q_primes = []
        self.skills = []
        self.skill_term_probs = []
        self.Q = []

        self.pretrain_Q = []
        self.pretrain_term = []
        self.seek_entropies = []
        self.seek_values = []
        self.seek_log_probs = []
        self.seek_rewards = []

        self.tune_entropies = []
        self.tune_values = []
        self.tune_log_probs = []
        self.tune_rewards = []

        self.avoid_start_flag = 0
        self.avoid_entropies = []
        self.avoid_values = []
        self.avoid_log_probs = []
        self.avoid_rewards = []

        self.high_records = [[],[],[],[]]
        self.low_done_count = 0
        self.episode.last_target = None
        self.episode.avoid_last_index = -2
        self.episode.avoid_last_length = 0
        self.low_skills = copy.deepcopy(self.low_skills_bak)
        self.random_avoid = [[1,0,5],[2,0,5]]

        self.avoid_policy = self.random_avoid[np.random.randint(2)].copy()
        self.state_buffer = []
        # modify
        self.window = 8
        self.avoid_count = 0
        self.lost_window = 4
        self.lost_target = [0] *self.lost_window
        self.lost_index = 0
        self.lost_flag = False
        self.teminate_actomatical = False
        self.avoidance = False
        self.last_policy_flag = -1
        self.last_state = ''
        self.high_num = 0
        self.avoid_num = 0
        self.termination = True
        self.current_skill = 0
        self.next_skill = 0
        self.episode_start_flag = True
        self.change_skill = True

        self.term_change_reg = 0.01
        self.term_skill_inconsistent_reg = 0.01
        self.duplicate_states_num = 0
        self.delay = 0.01 / 100000
        self.duplicate_flag = []

        with torch.cuda.device(self.gpu_id):
            self.seek_adj_hidden = (
                                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                            )
        self.seek_adj_probs = gpuify(
                        torch.zeros((1, self.action_space)), self.gpu_id
                    )

        return self

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self._episode.reset()
