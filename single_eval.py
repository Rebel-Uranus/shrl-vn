from __future__ import print_function, division
import os
import json
import time

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate

from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"


def single_eval(args=None, train_dir=None):
    if args is None:
        args = command_parser.parse_arguments()

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    args.test_or_val = 'val'

    args.data_dir = os.path.expanduser('~/Data/AI2Thor_offline_data_2.0.2/')

    if args.detection_feature_file_name is None:
        args.detection_feature_file_name = '{}_features_{}cls.hdf5'.format(args.detection_alg, args.num_category)

    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    tb_log_dir = os.path.join(args.work_dir, 'runs', '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.save_model_dir)]
    checkpoints = [
        (f, int(s[-3]))
        for (f, s) in checkpoints
        if len(s) >= 4 and f.startswith(args.title) and int(s[-3]) >= args.test_start_from
    ]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0
    
    #single eval step
    f, train_ep = checkpoints[-1]

    model = os.path.join(args.save_model_dir, f)
    args.load_model = model

    if train_dir is not None:
        filename = 'result.json' + '_' + args.load_model.split('_')[-3]
        args.results_json = os.path.join(train_dir, filename)

    # run eval on model
    args.test_or_val = "val"
    main_eval(args, create_shared_model, init_agent)

    # check if best on val.
    with open(args.results_json, "r") as f:
        results = json.load(f)


    log_writer.add_scalar("val/success", results["success"], train_ep)
    log_writer.add_scalar("val/spl", results["spl"], train_ep)

    args.phase = 'train'
    args.episode_type = 'BasicEpisode'

    return results, model

    



if __name__ == "__main__":
    single_eval()