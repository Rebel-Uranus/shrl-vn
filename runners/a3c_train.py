from __future__ import division

import time
import torch
import numpy as np

from datasets.constants import AI2THOR_TARGET_CLASSES

import setproctitle

from datasets.data import num_to_name
from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent
# from utils.model_util import ReplayBuffer
# from torchviz import make_dot
import time

import random

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def a3c_train(
        rank,
        args,
        create_shared_model,
        shared_model,
        initialize_agent,
        optimizer,
        res_queue,
        end_flag,
        scenes,
        replay_buffer=None,
):

    setproctitle.setproctitle('Training Agent: {}'.format(rank))

    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    
    random.seed(args.seed + rank)
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    player = initialize_agent(create_shared_model, args, rank, scenes, targets, gpu_id=gpu_id)
    compute_grad = not isinstance(player, RandomNavigationAgent)

    model_options = ModelOptions()

    episode_num = 0
    count = 0
    success_count = 0
    reward_count = 0
    while not end_flag.value:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        player.true_eps_len = 0
        player.episode.episode_times = episode_num
        new_episode(args, player)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, True)

            
            loss = compute_loss(args, player, gpu_id, model_options)
            

            if compute_grad and loss['total_loss'] != 0:
                # Compute gradient.
                player.model.zero_grad()
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()

            # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        count += 1


        end_episode(
            player,
            res_queue,
            title=num_to_name(int(player.episode.scene[9:])),
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
        )
        reset_player(player)

        episode_num = (episode_num + 1) % len(args.scene_types)

    player.exit()
