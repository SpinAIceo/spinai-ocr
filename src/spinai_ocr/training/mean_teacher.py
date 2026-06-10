"""Mean Teacher with dynamic EMA decay (0.996 → 0.9999).

Student: standard gradient-trained network.
Teacher: EMA of student weights. Used to produce consistency targets on
unlabeled images.

Adapted from the medical imaging project's `train.py` EMA schedule.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def dynamic_decay(step: int, total_steps: int, start: float = 0.996, end: float = 0.9999) -> float:
    """Cosine-smooth ramp from `start` → `end` over `total_steps`."""
    if total_steps <= 0:
        return end
    t = min(step / total_steps, 1.0)
    # cosine ease-in
    return end + (start - end) * (0.5 * (1 + math.cos(math.pi * t)))


class MeanTeacher(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        decay_start: float = 0.996,
        decay_end: float = 0.9999,
        total_steps: int = 100_000,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.total_steps = total_steps
        self._step = 0

    @torch.no_grad()
    def update_teacher(self) -> float:
        decay = dynamic_decay(self._step, self.total_steps, self.decay_start, self.decay_end)
        for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)
        # also EMA the buffers (BN running stats)
        for sb, tb in zip(self.student.buffers(), self.teacher.buffers()):
            tb.data.copy_(sb.data)
        self._step += 1
        return decay

    def forward(self, x: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        if use_teacher:
            self.teacher.eval()
            with torch.no_grad():
                return self.teacher(x)
        return self.student(x)
