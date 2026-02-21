import os
import uuid
from dataclasses import dataclass
from itertools import accumulate, pairwise

import torch

from src.s00_dist_setup import device

# -----------------------------------------------------------------------------
# Training Management

@dataclass
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule
    num_scheduled_iterations: int = 1020  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 200  # number of steps to continue training at final lr and ws
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 50  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = True
    # bigram hash embedding
    bigram_vocab_size: int = 50304 * 5

args = Hyperparameters()

@dataclass
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int
    duration: float = None
    lr_floor: float = 0.15

class TrainingSchedule:
    """
    Training schedule initialized via TRAINING_STAGES
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        3. YaRN updates to RoPE on window changes
        4. Split embed and lm head at 2/3 of training
        5. Batch size schedule of 8 -> 16 -> 24
        6. Post training extension of long windows from 13 to 20
        7. Seq len updates from 896 to 2048 at 1/3 of training
    """

    def __init__(self, stages: list[TrainingStage], scheduled_iterations: int, extension_iterations: int,
                 cooldown_frac: float = 0.5, split_embed_stage: int = 2, ws_post_yarn_ext: int = 20):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        # increase final validation ws, used for YaRN extension and short window size @classiclarryd
        self.ws_post_yarn_ext = ws_post_yarn_ext

        self.total_steps = self.scheduled_iterations + extension_iterations

        # Build stage boundaries (last is extension stage)
        ends = [0] + [round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])] + [self.total_steps]
        assert self.scheduled_iterations == ends[-2]
        self.boundaries = list(pairwise(ends))

        # Split embed at specified stage (ensure odd step for Adam)
        self.split_step = self.boundaries[split_embed_stage][0] | 1

        # Precompute stage 3 start LR (using original cooldown formula)
        s3_start = self.boundaries[2][0]
        lr = self.stages[2].lr_mul
        cd_start = int(scheduled_iterations * (1 - cooldown_frac))
        if s3_start >= cd_start:
            t = min(1.0, (s3_start - cd_start) / (scheduled_iterations - cd_start))
            lr = lr * (1 - t) + 0.15 * t
        self._stage3_start_lr = lr

        # Precompute MTP weights for all steps
        self.mtp_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        # Returns stage and % of the way through that stage
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr(self, step: int) -> float:
        stage, _ = self.lookup(step)

        # Stage 3: linear from original start LR to lr_floor
        s3_start, s3_end = self.boundaries[2]
        if s3_start <= step < s3_end:
            t = (step - s3_start) / (s3_end - s3_start)
            return self._stage3_start_lr * (1 - t) + stage.lr_floor * t

        # Extension: linear from lr_mul to lr_floor
        ext_start, ext_end = self.boundaries[3]
        if step >= ext_start:
            t = (step - ext_start) / (ext_end - ext_start)
            return self.stages[-1].lr_mul * (1 - t) + self.stages[-1].lr_floor * t

        # Stages 0 & 1: original cooldown behavior
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            lr = lr * (1 - t) + 0.15 * t
        return lr

# window_sizes are in units of `block_size` tokens (defined in TrainingManager)
TRAINING_STAGES = [
    TrainingStage(duration=1/3, train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=16 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.52,  # (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.73, lr_floor=0.30,  # (24/8)**0.5
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
    # extension stage
    TrainingStage(train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(6, 13), lr_mul=0.3, lr_floor=0.01,
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
]

training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=0.55)

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = training_schedule.total_steps - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum
