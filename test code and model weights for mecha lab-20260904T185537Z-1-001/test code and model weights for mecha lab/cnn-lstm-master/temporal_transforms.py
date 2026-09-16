"""Exact-length temporal crops that do not modify dataset frame indices."""
import random


def _fixed_window(frame_indices, size):
    if size <= 0:
        raise ValueError("Temporal sample duration must be positive.")
    indices = list(frame_indices)
    if not indices:
        raise ValueError("Cannot sample a video without frames.")
    return [indices[i % len(indices)] for i in range(size)]


class LoopPadding:
    """Take the first size indices; repeat short clips to exactly size."""
    def __init__(self, size):
        if size <= 0:
            raise ValueError("Temporal sample duration must be positive.")
        self.size = size

    def __call__(self, frame_indices):
        return _fixed_window(frame_indices, self.size)


class TemporalBeginCrop(LoopPadding):
    """Take an exact-length window from the beginning, looping if short."""


class TemporalCenterCrop(LoopPadding):
    """Take a centered exact-length window, looping if short."""
    def __call__(self, frame_indices):
        begin = max(0, (len(frame_indices) - self.size) // 2)
        return _fixed_window(frame_indices[begin:begin + self.size], self.size)


class TemporalRandomCrop(LoopPadding):
    """Take any valid exact-length window, including the final window."""
    def __call__(self, frame_indices):
        begin = random.randint(0, max(0, len(frame_indices) - self.size))
        return _fixed_window(frame_indices[begin:begin + self.size], self.size)
