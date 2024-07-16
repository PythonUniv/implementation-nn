# simple launch
# python train.py

# DDP launch (for many GPUs)
# torchrun --standalone --nproc_per_node=8 train.py


import os
import math
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

from dataset import DataLoaderLite
from gpt2 import GPT2, GPTConfig


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


class CosineLearningRate:
    def __init__(self, max_lr: float, min_lr_scale: float, warmup_steps: int, max_steps: int):
        assert warmup_steps < max_steps
        
        self.max_lr = max_lr
        self.min_lr = min_lr_scale * max_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
    def __call__(self, iteration: int):
        """
            Get learning rate.
        """
        
        if iteration < self.warmup_steps:
            return (iteration + 1) * self.max_lr / self.warmup_steps
        elif iteration >= self.max_steps:
            return self.min_lr
        else:
            decay_ratio = (iteration - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coef = 0.5 * (1 + math.cos(decay_ratio * math.pi))
            return self.min_lr + (self.max_lr - self.min_lr) * coef


def train(
    data_root: str, total_batch_size: int = 524_288, batch_size: int = 64, max_seq_len: int = 1024, max_lr: float = 6e-4,
    min_lr_scale: float = 0.1, warmup_steps: int = 715, max_steps: int = 19073, device_type: str = 'cuda', auto_cast_dtype='float16'
):
    ddp = os.environ.get('RANK') is not None
    if ddp:
        assert torch.cuda.is_available()
        
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        print(f'Using device: {device}')
    
    assert total_batch_size % (batch_size * max_seq_len * ddp_world_size) == 0
    grad_accumulation_steps = total_batch_size // (batch_size * max_seq_len * ddp_world_size)
    
    if master_process:
        print(f'Total batch size: {total_batch_size:,}')
        print(f'Gradient accumulation steps: {grad_accumulation_steps:,}')
    
    train_data_loader = DataLoaderLite(data_root, batch_size, max_seq_len, ddp_rank, ddp_world_size, 'train')
    val_data_loader = DataLoaderLite(data_root, batch_size, max_seq_len, ddp_rank, ddp_world_size, 'test')
    
    torch.set_float32_matmul_precision('high')
    
    # create model
    config = GPTConfig()
    model = GPT2(config)
    model.to(device)
    
    use_compile = False
    if use_compile:
        model = torch.compile(model)
        
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)
    
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, 'log.txt')
    with open(log_file_path, 'w') as file:
        pass
    
    cosine_learning_rate = CosineLearningRate(max_lr, min_lr_scale, warmup_steps, max_steps)
    dtype = getattr(torch, auto_cast_dtype)
    
    for step in range(max_steps):
        t_0 = time.time()
        last_step = step == max_steps - 1
        
        if step % 250 == 0 or last_step:
            model.eval()
            val_data_loader.reset()
            with torch.no_grad():
                val_loss_accumulation = 0.0
                val_loss_steps = 20
                for idx in range(val_loss_steps):
                    x, y = val_data_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type, dtype=dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accumulation += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accumulation, op=dist.ReduceOp.AVG)
            if master_process:
                print(f'Validation loss: {val_loss_accumulation:.4f}')
                with open(log_file_path, 'a') as file:
                    file.write(f'Step: {step}, validation loss: {val_loss_accumulation:.4f}')
            
        if step and step % 5000 == 0 or last_step and master_process:
            checkpoint_path = os.path.join(log_dir, f'model_{step}.pt')
            print(f'Model is saved on step: {step}')
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accumulation
            }
            
            torch.save(checkpoint, checkpoint_path)
            
        model.train()
        optimizer.zero_grad()
        
        loss_accumulated = 0
        for micro_step in range(grad_accumulation_steps):
            x, y = train_data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accumulation_steps - 1
            
            with torch.autocast(device_type, dtype=dtype):
                logits, loss = model(x, y)
            
            loss /= grad_accumulation_steps
            loss_accumulated += loss.detach()
            loss.backward()
            
        if ddp:
            dist.all_reduce(loss_accumulated, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        lr = cosine_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        
        if device_type == 'cuda':
            torch.cuda.synchronize()
        
        t_1 = time.time()
        dt = t_1 - t_0
        
        tokens_per_second = total_batch_size / dt
        if master_process:
            print(
                f'step: {step:5d} | loss: {loss_accumulated} | learning rate: {lr:4e} | norm: {norm:4f} | time: {1000 * dt:.1f} ms/step | {tokens_per_second} tokens/sec')
            with open(log_file_path, 'a') as file:
                file.write(f'{step} train {loss_accumulated}\n')
        
    if ddp:
        destroy_process_group()
