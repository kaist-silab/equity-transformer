import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
import tsplib95
from utils.problem_augment import augment

# from nce.solver import solve_mTSP
mp = torch.multiprocessing.get_context('spawn')
import random
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def eval_dataset_mp(args):
    (model, dataset_path, width, softmax_temp, opts, i, num_processes) = args

    # model, _ = load_model(opts.model)

    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(model, dataset_path, width, softmax_temp, opts, offset):
    # Even with multiprocessing, we load the model here since it contains the name where to write results

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(model, dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.sample_size, offset=offset)
        results, max_val, start_time = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size
    # parallelism = num_processes
    costs, tours, durations = zip(*results)  # Not really costs since they should be negative
 
    return costs, durations, max_val


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('greedy') else "sampling",
        temp=softmax_temp)
    
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    if opts.N_aug > 1:
        aug = opts.N_aug
    else:
        aug = 1

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        if opts.problem == 'mtsp':
            max_val = batch.max()
            if max_val > 1:
                batch = batch/max_val
        else:
            max_val = None

        # For TSPLIB
        if aug > 1:
            batch = augment(batch, aug)

        # distance_matrix = torch.cdist(batch, batch, p=2)
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy' and opts.N_aug == 8:
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep, agent_num=opts.agent_num, aug=aug)
    
        duration = time.time() - start
        results.append((costs.cpu().numpy(), sequences.cpu().numpy(),duration))
    
    return results, max_val, start


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default="mtsp", type=str, help="problem type")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--decode_type', type=str, default='greedy',
                        help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+', default=[0],
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default='greedy',
                        help='Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--multiprocessing', default=False,
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--agent_num', default=3, type=int, help="decide the number of agent")
    parser.add_argument('--ft',default="Y", type=str)
    parser.add_argument('--is_serial', default='True', type=str, help="whether to use serial augmentation of instance")
    parser.add_argument('--N_aug', default=8, type=int, help="how any augmentation of instance")
    parser.add_argument('--max_calc_batch_size', default=100000, type=int, help="max batch size for calculation")
    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"
    is_serial = opts.is_serial.lower() == 'true'
    if is_serial:
        num_iter = opts.val_size
        opts.sample_size = 1
    else:
        num_iter = 1
        opts.sample_size = opts.val_size
    widths = opts.width if opts.width is not None else [0]

    Performance = []
    Time = []
    agent_num = opts.agent_num
    model, _ = load_model(opts.model, agent_num=agent_num, ft=opts.ft)
    model.agent_num = opts.agent_num

    for width in widths:
        for dataset_path in opts.datasets:
            for i in range(num_iter):
                cost, duration, max_val = eval_dataset(model, dataset_path, width, opts.softmax_temperature, opts, offset=i)
                Performance.append(cost)
                Time.append(duration)
    Performance = np.array(Performance)
    # For TSPLIB
    if max_val is not None:
        if max_val > 1:
            Performance = Performance * max_val
    Time = np.array(Time)
    print("Average-Performance : ", np.mean(Performance))
    print("Average-Time : ", np.mean(Time))

    
                    

