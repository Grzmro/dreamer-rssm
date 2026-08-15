"""Shared training budget (Phase 4): stop on env steps *or* wall-clock.

The repo's primary x-axis is environment interactions (see
``train/common_logger.py``), and that stays the default: every agent gets the
same number of samples and Dreamer is expected to win. This module adds the
*other* comparison — the same number of **seconds** — because "which agent
should I run on one GPU overnight" is a different question from "which agent
squeezes the most out of 400k frames", and the two have opposite answers here.

A time budget is a **hard stop layered on top of a planned horizon**, never a
replacement for it. Agents anneal their schedules over
``baselines.total_env_steps``: PPO decays its learning rate over the planned
iteration count, DQN decays epsilon over ``exploration_fraction * total``. If
the horizon were simply set to infinity and the run cut off by a timer, those
schedules would silently stretch out and the comparison would measure the
mangled schedule rather than the algorithm. So a time-matched sweep sets a
per-agent step target it expects to fit in the budget (see
``benchmark.agent_env_steps``), and the timer only guarantees that no agent
gets more wall-clock than the others.

Measured throughput on one A100 (ALE/Pong-v5, 2026-08-10 benchmark) for
sizing those targets: dreamer ~11.8 env steps/s, dqn ~397/s, ppo ~980/s.
"""

from __future__ import annotations

import time


class TrainingBudget:
    """Whichever comes first: ``max_env_steps`` or ``time_budget_s`` seconds.

    The clock starts when the object is constructed, which every caller does
    immediately before its training loop — matching ``wall_time_s`` in the
    benchmark CSVs, which is also measured from loop start.
    """

    def __init__(self, max_env_steps: int, time_budget_s: float | None = None):
        self.max_env_steps = int(max_env_steps)
        self.time_budget_s = (
            None if time_budget_s is None or float(time_budget_s) <= 0
            else float(time_budget_s)
        )
        self.start = time.monotonic()
        self.stopped_on: str | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining_s(self) -> float | None:
        if self.time_budget_s is None:
            return None
        return self.time_budget_s - self.elapsed()

    def out_of_time(self) -> bool:
        """True once the wall-clock budget is spent (always False without one).

        For loops already bounded by a step counter (``for global_step in
        range(...)``): breaking on this alone keeps the existing step
        semantics exactly, with no off-by-one on the final step.
        """
        if self.time_budget_s is not None and self.elapsed() >= self.time_budget_s:
            self.stopped_on = "time"
            return True
        return False

    def exhausted(self, env_step: int) -> bool:
        """True once either limit is hit; records which one for reporting.

        For ``while`` loops that carry no step bound of their own.
        """
        if self.out_of_time():
            return True
        if env_step >= self.max_env_steps:
            self.stopped_on = "steps"
            return True
        return False

    def summary(self, env_step: int) -> str:
        """One line for the run log: what stopped it, and whether that is fair.

        A time-matched run that stops on ``steps`` did not actually spend its
        budget — the planned horizon was too small — and a step-matched run
        that stops on ``time`` was truncated. Both cases make a comparison
        unequal, so they are called out explicitly rather than left to be
        inferred from the CSV.
        """
        mins = self.elapsed() / 60
        if self.stopped_on == "time":
            note = (
                "" if self.time_budget_s is None
                else f" (budget {self.time_budget_s / 60:.1f} min)"
            )
            return (
                f"stopped on TIME after {mins:.1f} min{note} at {env_step} env "
                f"steps of {self.max_env_steps} planned"
            )
        return (
            f"stopped on STEPS at {env_step} in {mins:.1f} min"
            + (
                ""
                if self.time_budget_s is None
                else f" — WARNING: {self.time_budget_s / 60:.1f} min budget was "
                     f"not spent, raise the step target for a time-matched run"
            )
        )
