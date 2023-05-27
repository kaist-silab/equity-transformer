import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from utils.problem_augment import augment
import random

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    # print(cost.shape)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat, batch_size, aug=8):
        with torch.no_grad():
            cost, _  = model(move_to(bat, opts.device))
            # print(cost.shape)
            cost, _ = cost.view(aug, -1).min(0, keepdim=True)
            cost = cost.transpose(0, 1)
            
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(augment(bat, opts.N_aug), batch_size=opts.eval_batch_size, aug=opts.N_aug)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, problem, opts):
    
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    graph_size = opts.graph_size
    
    training_dataset = problem.make_dataset(
        size=graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    model = get_inner_model(model)
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        agent_num = random.sample(range(opts.agent_min, opts.agent_max), 1)[0]
        model.agent_num = agent_num
        model.embedder.agent_num = agent_num
        
        train_batch(
            model,
            optimizer,
            batch,
            opts
        )

        if batch_id > 0 and batch_id % 100 == 0:
            print('Saving model and state...')
            torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all()
        },
        os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))
        
                
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
        epoch_duration = time.time() - start_time
    
    step += 1
        
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    lr_scheduler.step()
    
    # print("Validating...")
    # avg_reward = validate(model, val_dataset, opts)
    # model.agent_num = 40
    # print("Validation: {}".format(avg_reward))


def train_batch(
        model,
        optimizer,
        batch,
        opts
):
    info = {}

    x = move_to(batch, opts.device)

    # Evaluate model, get costs and log probabilities
    x_aug = augment(x, opts.N_aug)
    cost, log_likelihood = model(x_aug)
    
    # Calculate loss
    cost = cost.view(opts.N_aug,-1).permute(1,0)
    log_likelihood = log_likelihood.view(opts.N_aug,-1).permute(1,0)

    advantage = (cost - cost.mean(dim=1).view(-1,1))
    loss = ((advantage) * log_likelihood).mean()

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    
    return info