import os
import sys
import copy
import time
import gc

import torch
import triton
import torch.distributed as dist
from torch import nn

# Import all modules in order - side effects in 00 set up distributed state
from src.s00_dist_setup import rank, world_size, grad_accum_steps, grad_scale, master_process, device
import src.s01_fp8_ops  # registers custom ops
from src.s05_model import GPT
from src.s06_data import distributed_data_generator
from src.s07_schedule import args, TRAINING_STAGES, training_schedule
from src.s08_training_manager import TrainingManager

# -----------------------------------------------------------------------------
# Logging

logfile = None
run_id = args.run_id
if master_process:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)

def print0(s, console=False):
    if not master_process:
        return 

    if console:
        print(s)
    

    with open(logfile, "a") as f:
        print(s, file=f)
    
    

# -----------------------------------------------------------------------------

def log_env_info():
    print0("="*100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"Running Triton version {triton.__version__}")
    def nvidia_smi():
        import subprocess
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("="*100)

def create_model():
    """Build GPT, cast to bf16, broadcast, compile, wrap in TrainingManager."""
    val_max_seq_len = args.val_batch_size // (grad_accum_steps * world_size)

    model: nn.Module = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=val_max_seq_len,
        bigram_vocab_size=args.bigram_vocab_size,
    ).to(device="cuda")


    model.weights_to_bfloat16()


    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
    training_manager = TrainingManager(model)
    return model, training_manager, val_max_seq_len

def make_loader(files, batch_size, max_seq_len, **kw):
    """Shorthand for distributed_data_generator with shared args."""
    return distributed_data_generator(
        files, batch_size, max_seq_len,
        grad_accum_steps=grad_accum_steps,
        bigram_vocab_size=args.bigram_vocab_size,
        **kw,
    )

def warmup(model, training_manager: TrainingManager, val_max_seq_len):
    """Run representative steps to compile kernels, then reset state."""
    print0("Compiling model and warming up kernels (~7 minutes on first execution)", console=True)
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=training_manager.get_state())
    train_loader = make_loader(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len)
    val_loader = make_loader(args.val_files, args.val_batch_size, -1, align_to_bos=False)

    transition_steps = training_manager.get_transition_steps()
    warmup_steps = sorted({0, 1} | set(s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 0))
    print0(f"Sampling steps {warmup_steps} for warmup", console=True)
    for step in warmup_steps:
        training_manager.advance_schedule(step)
        model.eval()
        with torch.no_grad():
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(), val_max_seq_len=val_max_seq_len)
        model.train()
        for idx in range(grad_accum_steps):
            send_args = training_manager.train_loader_send_args
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(send_args)
            training_manager.sparse_index_update(step, bigram_cpu)
            loss, _ = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * grad_scale
            training_manager.sparse_index_share(step)
            loss.backward()
            del loss
        training_manager.step_optimizers(step)

    print0("Resetting Model", console=True)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    del val_loader, train_loader, initial_state
    model.train()

def validate(model, training_manager, val_max_seq_len):
    """Run val loop, return avg reduced loss."""
    model.eval()
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = make_loader(args.val_files, args.val_batch_size, -1, align_to_bos=False)
    val_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            val_loss_b, _ += model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args(), val_max_seq_len=val_max_seq_len)
            val_loss += val_loss_b
    val_loss /= val_steps
    del val_loader
    dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
    model.train()
    return val_loss

def train_step(model, training_manager, train_loader, step):
    """One training step: grad accum loop + optimizer step."""
    train_loss = 0
    for idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
        training_manager.sparse_index_update(step, bigram_cpu)
        loss, tokens_numel = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) * grad_scale
        training_manager.sparse_index_share(step)
        train_loss += loss.item() / tokens_numel
        loss.backward()
        del loss
    training_manager.step_optimizers(step)
    return train_loss

def main():
    log_env_info()
    model, training_manager, val_max_seq_len = create_model()
    warmup(model, training_manager, val_max_seq_len)

    train_loader = make_loader(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len)
    gc.collect()

    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = training_schedule.total_steps
    for step in range(train_steps + 1):
        last_step = (step == train_steps)
        training_manager.advance_schedule(step)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step:
                training_manager.apply_final_ws_ext()
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            val_loss = validate(model, training_manager, val_max_seq_len)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, model=model.state_dict(), optimizer=training_manager.get_state())
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            break

        # --------------- TRAINING SECTION -----------------
        train_loss = train_step(model, training_manager, train_loader, step)

        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps}, train_time:{approx_training_time_ms:.0f}ms, step_avg:{approx_training_time_ms/(step + 1):.2f}ms, train loss:{train_loss:.4f},  lr:{training_schedule.get_lr(step)}", console=True)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
