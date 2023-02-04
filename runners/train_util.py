from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from models.model_io import ModelInput


def update_test_model(args, player, target_action_prob, weight=1):
    action_loss = weight * F.cross_entropy(player.last_action_probs, torch.max(target_action_prob, 1)[1])
    inner_gradient = torch.autograd.grad(
        action_loss,
        [v for _, v in player.model.named_parameters()],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    player.model.load_state_dict(SGD_step(player.model, inner_gradient, args.inner_lr))
    player.episode.model_update = True


def run_episode(player, args, total_reward, model_options, training, shared_model=None):
    num_steps = args.num_steps

    update_test = False

    count = 0
    while not player.done:
        player.action(model_options, training, update_test)
        count +=1
        if player.done:
            break
    return sum(player.high_rewards)


def new_episode(args, player):
    player.episode.new_episode(args, player.scenes, player.targets)
    player.reset_hidden()
    player.done = False


def a3c_loss(args, player, gpu_id, model_options):
    """ Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """
    R = torch.zeros(1, 1)
    
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    low_pretrain = args.low_pretrain
    # low_pretrain = False
    high_pretrain = args.high_pretrain
    # high_pretrain = True
    seek_policy_loss = 0
    seek_value_loss = 0
    if low_pretrain:
        for i in range(len(player.seek_rewards)):
            entropies = player.seek_entropies[i]
            values = player.seek_values[i]
            log_probs = player.seek_log_probs[i]
            rewards = player.seek_rewards[i]
            values.append(Variable(R))
            if len(rewards) != 0 and len(rewards) != len(values)-1:
                print('seek rewards != values')
                print(rewards, values,Variable(R))
                print('^&'*50)
            p_l, v_l = fragment_loss(args, R, rewards, values, log_probs, entropies, gpu_id)
            seek_policy_loss += p_l
            seek_value_loss += v_l
            if len(rewards) != 0 and len(rewards) != len(values)-1:
                print('seek rewards != values')
            if len(rewards) != len(log_probs):
                print('seek rewards != log_probs')
            if len(rewards) != len(entropies):
                print('seek rewards != entropies')

    
    tune_policy_loss = 0
    tune_value_loss = 0
    if low_pretrain:
        for i in range(len(player.tune_rewards)):
            entropies = player.tune_entropies[i]
            values = player.tune_values[i]
            log_probs = player.tune_log_probs[i]
            rewards = player.tune_rewards[i]
            values.append(Variable(R))
            p_l, v_l = fragment_loss(args, R, rewards, values, log_probs, entropies, gpu_id)
            tune_policy_loss += p_l
            tune_value_loss += v_l
            if len(rewards) != 0 and len(rewards) != len(values)-1:
                print('tune rewards != values')
            if len(rewards) != len(log_probs):
                print('tune rewards != log_probs')
            if len(rewards) != len(entropies):
                print('tune rewards != entropies')

    avoid_policy_loss = 0
    avoid_value_loss = 0

    high_policy_loss = 0
    high_value_loss = 0
    termination_loss = 0
    td_err = 0

    
   
    if high_pretrain:
        action_input = torch.cat(player.Q,dim=0).reshape(-1,4)
        action_target = torch.tensor(player.skills)
        
        with torch.cuda.device(gpu_id):
            action_target = action_target.cuda()

        crossentropyloss = nn.CrossEntropyLoss()
        td_err_il = crossentropyloss(action_input,action_target)
        

    if not low_pretrain:
        entropies = player.high_entropies
        values = player.high_values
        log_probs = player.high_log_probs
        rewards = player.high_rewards
        values.append(Variable(R))
        
        duplicate_reg = 0
        duplicate_reg += (player.duplicate_states_num / len(player.state_buffer))* 2
        term_skill_inconsistent = 0
        low_stop_num = 0
        high_change_num = 0
        for i in range(1,len(player.skills)):
            if player.actions[i-1] == 5:
                low_stop_num += 1
                if player.skills[i]==player.skills[i-1]:
                    term_skill_inconsistent += 1
            if player.skills[i]!=player.skills[i-1] and player.actions[i-1] != 5:
                high_change_num += 1
        low_stop_num = max(low_stop_num, 1)
        term_skill_inconsistent = (term_skill_inconsistent / low_stop_num)*2
        high_change_num *= 0.05
        rewards[-1] = rewards[-1] - duplicate_reg - term_skill_inconsistent

        p_l, v_l = fragment_loss(args, R, rewards, values, log_probs, entropies, gpu_id)
        high_policy_loss += p_l
        high_value_loss += v_l

     
    if low_pretrain:
        policy_loss = seek_policy_loss + tune_policy_loss + avoid_policy_loss + high_policy_loss
        value_loss = seek_value_loss + tune_value_loss + avoid_value_loss + high_value_loss
    elif high_pretrain:
        policy_loss = termination_loss+td_err_il
        value_loss = high_value_loss
    else:
        policy_loss = seek_policy_loss + tune_policy_loss + avoid_policy_loss + high_policy_loss
        value_loss = seek_value_loss + tune_value_loss + avoid_value_loss + high_value_loss


    return policy_loss, value_loss

def  fragment_loss(args, R, rewards, values, log_probs, entropies, gpu_id):
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()
    R = Variable(R)
    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = (
                rewards[i]
                + args.gamma * values[i + 1].data
                - values[i].data
        )

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = (
                policy_loss
                - log_probs[i] * Variable(gae)
                - args.beta * entropies[i]
        )

    return policy_loss, value_loss



def imitation_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.il_update_actions:
        step_optimal_action = torch.tensor(player.il_update_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss

def meta_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.meta_learning_actions:
        step_optimal_action = torch.tensor(player.meta_learning_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        # step_loss = F.cross_entropy(player.meta_predictions[i], step_optimal_action)
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss

def duplicate_states_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.duplicate_states_actions:
        step_optimal_action = torch.tensor(player.duplicate_states_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss


def compute_learned_loss(args, player, gpu_id, model_options):
    loss_hx = torch.cat((player.hidden[0], player.last_action_probs), dim=1)
    learned_loss = {
        "learned_loss": player.model.learned_loss(
            loss_hx, player.learned_input, model_options.params
        )
    }
    player.learned_input = None
    return learned_loss


def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for (param_name, param), (shared_param_name, shared_param) in zip(
            player.model.named_parameters(), shared_model.named_parameters()
    ):
        if shared_param.requires_grad and 'target_' not in shared_param_name:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                if param.grad !=0:
                    shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()
            


def transfer_gradient_to_shared(gradient, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape)
            elif gpu_id < 0:
                param._grad = gradient[i]
            else:
                param._grad = gradient[i].cpu()

        i += 1


def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta


def update_loss(sum_total_loss, total_loss):
    if sum_total_loss is None:
        return total_loss
    else:
        return sum_total_loss + total_loss

def soft_update_params(net, target_net, tau=0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()


def SGD_step(theta, grad, lr):
    theta_i = {}
    j = 0
    for name, param in theta.named_parameters():
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param
        j += 1

    return theta_i


def get_scenes_to_use(player, scenes, args):
    if args.new_scene:
        return scenes
    return [player.episode.environment.scene_name]


def compute_loss(args, player, gpu_id, model_options):
    policy_loss, value_loss = a3c_loss(args, player, gpu_id, model_options)
    loss = {'policy_loss': policy_loss,
            'value_loss': value_loss}
    loss['total_loss'] = loss['policy_loss'] + 0.5 * loss['value_loss']

    return loss


def batch_modleinput(z_pos,gpu_id=-1):
    state = z_pos[0].state
    det_features = z_pos[0].detection_inputs['features'].unsqueeze(dim=0)
    det_scores = z_pos[0].detection_inputs['scores'].unsqueeze(dim=0)
    det_labels = z_pos[0].detection_inputs['labels'].unsqueeze(dim=0)
    det_boxes = z_pos[0].detection_inputs['bboxes'].unsqueeze(dim=0)
    det_target = z_pos[0].detection_inputs['target'].unsqueeze(dim=0)
    det_indicators = z_pos[0].detection_inputs['indicator'].unsqueeze(dim=0)
    action_probs = z_pos[0].action_probs

    for i in range(1, len(z_pos)):
        state = torch.cat((state,z_pos[i].state), dim=0)
        det_features = torch.cat((det_features,z_pos[0].detection_inputs['features'].unsqueeze(dim=0)), dim=0)
        det_scores = torch.cat((det_scores,z_pos[0].detection_inputs['scores'].unsqueeze(dim=0)), dim=0)
        det_labels = torch.cat((det_labels,z_pos[0].detection_inputs['labels'].unsqueeze(dim=0)), dim=0)
        det_boxes = torch.cat((det_boxes,z_pos[0].detection_inputs['bboxes'].unsqueeze(dim=0)), dim=0)
        det_target = torch.cat((det_target,z_pos[0].detection_inputs['target'].unsqueeze(dim=0)), dim=0)
        det_indicators = torch.cat((det_indicators,z_pos[0].detection_inputs['indicator'].unsqueeze(dim=0)), dim=0)
        action_probs = torch.cat((action_probs,z_pos[0].action_probs), dim=0)
    
    
    model_input = ModelInput()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model_input.state = state.detach().cuda()
            detection_inputs = {
                'features': det_features.detach().cuda(),
                'scores': det_scores.detach().cuda(),
                'labels': det_labels.detach().cuda(),
                'bboxes': det_boxes.detach().cuda(),
                'target': det_target.detach().cuda(),
                'indicator': det_indicators.detach().cuda(),
            }
            model_input.detection_inputs = detection_inputs
            model_input.action_probs = action_probs.detach().cuda()
    return model_input



def end_episode(
        player, res_queue, title=None, episode_num=0, include_obj_success=False, **kwargs
):  
    seek_success = []
    for i in range(len(player.seek_rewards)):
        seek_success.append(0 if player.seek_rewards[i][-1]<=0 else 1)
    tune_success = []
    for i in range(len(player.tune_rewards)):
        tune_success.append(0 if player.tune_rewards[i][-1]<=0 else 1)
    avoid_success = []
    
    
    results = {
        'done_count': player.episode.done_count,
        'ep_length': player.eps_len - player.low_done_count,
        'success': int(player.success),
        'low_seek': seek_success,
        'low_tune': tune_success,
        'low_avoid': avoid_success,
        'tools': {
            'scene': player.episode.scene,
            'target': player.episode.task_data,
            'states': player.episode.states,
            'action_outputs': player.episode.action_outputs,
            'high_action_list': player.high_records[1],
            'action_list': [int(item) for item in player.episode.actions_record],
            'detection_results': player.episode.detection_results,
            'success': player.success,
        }
    }

    results.update(**kwargs)
    res_queue.put(results)


def get_bucketed_metrics(spl, best_path_length, success):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
    return out


def compute_spl(player, start_state):
    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            # This is due to a rare known bug.
            continue

    if not player.success:
        return 0, best

    if best < float("inf"):
        return best / float(player.true_eps_len), best

    # This is due to a rare known bug.
    return 0, best


def action_prob_detection(bbox):
    center_point = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    left_prob = np.linalg.norm(center_point - np.array([0, 150]))
    right_prob = np.linalg.norm(center_point - np.array([300, 150]))
    up_prob = np.linalg.norm(center_point - np.array([150, 0]))
    down_prob = np.linalg.norm(center_point - np.array([150, 300]))
    forward_prob = np.linalg.norm(center_point - np.array([150, 150]))

    detection_prob = torch.tensor([forward_prob, left_prob, right_prob, up_prob, down_prob])

    return torch.argmin(detection_prob)
