from __future__ import division

import time
import torch
import setproctitle
import copy
import numpy as np
import random

from datasets.constants import AI2THOR_TARGET_CLASSES
from datasets.data import name_to_num

from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)


def a3c_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
    scenes,
):

    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    if scene_type == "living_room":
        args.max_episode_length = 200 # 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Agent: {}".format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        saved_state = torch.load(
            model_to_open, map_location=lambda storage, loc: storage
        )
        shared_model.load_state_dict(saved_state['model'])

    player = initialize_agent(model_create_fn, args, rank, scenes, targets, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()


    while count < max_count:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        player.true_eps_len = 0
        new_episode(args, player)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()
        
        

        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False, shared_model)

            episode_states = []
            episode_states.append(player.episode.scene)
            episode_states = episode_states + player.episode.states

            
            # Compute the loss.
            if not player.done:
                reset_player(player)


        spl, best_path_length = compute_spl(player, player_start_state)
        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

    player.exit()
    res_queue.put({"END": True})
