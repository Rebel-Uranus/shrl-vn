""" Contains the Episodes for Navigation. """
import random
import sys
import torch
import numpy as np
import math

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, DUPLICATE_STATE, UNSEEN_STATE
from datasets.constants import DONE
from datasets.environment import Environment

from utils.model_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.model_util import gpuify
from .episode import Episode



class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.scene = None

        self.scene_states = []
        if args.eval:
            random.seed(args.seed)

        self._episode_times = 0
        self.seen_percentage = 0

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []
        self.obs_reps = []

        self.episode_length = 0
        self.target_object_detected = False

        # tools
        self.states = []
        self.frames = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []


        self.action_failed_il = False

        self.action_probs = []

        self.meta_predictions = []

        self.warm_up = args.warm_up
        self.num_workers = args.num_workers
        self.episode_num = 0

        self.last_target = None
        self.avoid_last_index = -2
        self.avoid_last_length = 0
        self.tune_step = 0
        self.tune_areas = []
        self.tune_states = set()
        

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    @property
    def episode_times(self):
        return self._episode_times

    @episode_times.setter
    def episode_times(self, times):
        self._episode_times = times

    def cosine_similarity(vector1, vector2):
        dot_product= 0.0
        normA= 0.0
        normB= 0.0
        for a, b in zip(vector1, vector2):
            dot_product+= a* b
            normA+= a** 2
            normB+= b** 2
        if normA== 0.0 or normB== 0.0:
            return 0
        else:
            return 5 - round(dot_product/ ((normA**0.5)*(normB**0.5))* 5,2)

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_detection_feature(self):
        return self.environment.current_detection_feature

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int, policy_flag, high_index, model_input, avoid_start_flag):

        
        if policy_flag != 0:
            action = self.actions_list[action_as_int]
            self.episode_length += 1
            if action["action"] != DONE:
                
                self.environment.step(action)
        else:
            action = {'action':'end'}
            self.done_count += 1

        reward, terminal, action_was_successful, high_reward = self.judge(action, policy_flag, high_index, model_input, avoid_start_flag)
        return reward, terminal, action_was_successful, high_reward

    def judge(self, action, policy_flag, high_index, model_input, avoid_start_flag):
        """ Judge the last event. """
        reward = STEP_PENALTY
        high_reward = STEP_PENALTY

        if policy_flag!=0:
            self.scene_states.append(self.environment.controller.state)

        done = False
        action_was_successful = self.environment.last_action_success
        if policy_flag == 0: 
            if action["action"] == 'end':
                action_was_successful = False
                for id_ in self.task_data:
                    if self.environment.object_is_visible(id_):
                        high_reward = GOAL_SUCCESS_REWARD
                        done = True
                        action_was_successful = True
                        break
                if not done:
                    high_reward = STEP_PENALTY
                
        elif policy_flag == 1:#seek
            if action["action"] == DONE:
                for i in model_input.detection_inputs['indicator']:
                    if i == 1:
                        reward = GOAL_SUCCESS_REWARD
                        break
        elif policy_flag == 2:#tune
            
            if action["action"] == DONE:
                reward = STEP_PENALTY
                for id_ in self.task_data:
                    if self.environment.object_is_visible(id_):
                        reward = GOAL_SUCCESS_REWARD
                        break

        return reward, done, action_was_successful, high_reward

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def _new_episode(self, args, scenes, targets):
        """ New navigation episode. """
        scene = random.choice(scenes)
        # sence = 'FloorPlan1'
        self.scene = scene

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                detection_feature_file_name=args.detection_feature_file_name,
                images_file_name=args.images_file_name,
                visible_object_map_file_name=args.visible_map_file_name,
                local_executable_path=args.local_executable_path,
                optimal_action_file_name=args.optimal_action_file_name,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        self.task_data = []

        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        # Randomize the start location.
        # self._env.randomize_agent_location()

        warm_up_path_len = 200
        if (self.episode_num * self.num_workers) < 500000:
            warm_up_path_len = 5 * (int((self.episode_num * self.num_workers) / 50000) + 1)
        else:
            self.warm_up = False

        if self.warm_up:
            for _ in range(10):
                self._env.randomize_agent_location()
                shortest_path_len = 1000
                for _id in self.task_data:
                    path_len = self._env.controller.shortest_path_to_target(self._env.start_state, _id)[1]
                    if path_len < shortest_path_len:
                        shortest_path_len = path_len
                if shortest_path_len <= warm_up_path_len:
                    break
        else:
            self._env.randomize_agent_location()

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

    def new_episode(self, args, scenes, targets):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.episode_length = 0
        self.prev_frame = None
        self.current_frame = None
        self.scene_states = []

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []

        self.target_object_detected = False

        self.episode_times += 1
        self.episode_num += 1

        self.states = []
        self.frames = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []
        self.obs_reps = []

        self.action_failed_il = False

        self.action_probs = []
        self.meta_predictions = []

        self.last_target = None
        self.avoid_last_index = -2
        self.avoid_last_length = 0
        self.tune_step = 0
        self.tune_areas = []
        self.tune_states = set()


        self._new_episode(args, scenes, targets)
