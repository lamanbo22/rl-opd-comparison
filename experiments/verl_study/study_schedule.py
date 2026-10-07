"""Pure, testable budget scheduling and deterministic training-data cursors."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

METHODS = ("rl", "opd", "joint", "opd_then_rl", "rl_then_opd")


@dataclass
class BudgetSchedule:
    method: str
    mode: str
    budget_seconds: float
    endpoint_policy: str = "guarded_ceiling"
    reserve_seconds: float = 300.0
    overrun_tolerance_fraction: float = 0.05
    validation_rounds: int = 2
    validation_switch_after: int = 1
    charged_seconds: float = 0.0
    completed_rounds: int = 0
    actual_updates: int = 0
    phase1_seconds: float = 0.0
    phase2_seconds: float = 0.0
    switch_round: int | None = None
    switch_at_seconds: float | None = None

    def __post_init__(self):
        assert self.method in METHODS and self.mode in ("validation", "budgeted")
        assert self.endpoint_policy in ("guarded_ceiling", "last_at_or_below", "first_at_or_above")
        assert math.isfinite(self.budget_seconds) and self.budget_seconds > 0
        assert math.isfinite(self.reserve_seconds) and self.reserve_seconds > 0
        if self.mode == "budgeted" and self.endpoint_policy == "guarded_ceiling":
            assert self.reserve_seconds < self.budget_seconds / 2
        assert 0 <= self.overrun_tolerance_fraction <= 1
        assert 2 <= self.validation_rounds <= 4
        assert 1 <= self.validation_switch_after < self.validation_rounds
        assert math.isfinite(self.charged_seconds) and self.charged_seconds >= 0
        assert self.completed_rounds >= 0 and 0 <= self.actual_updates <= self.completed_rounds
        assert math.isclose(self.phase1_seconds + self.phase2_seconds, self.charged_seconds, abs_tol=1e-6)

    @property
    def sequential(self):
        return "_then_" in self.method

    def phase_index(self):
        if not self.sequential:
            return 0
        if self.mode == "validation":
            return int(self.completed_rounds >= self.validation_switch_after)
        return int(self.charged_seconds >= self.budget_seconds / 2)

    def active_method(self):
        if not self.sequential:
            return self.method
        return self.method.split("_then_")[self.phase_index()]

    def should_stop(self):
        if self.mode == "validation":
            return self.completed_rounds >= self.validation_rounds
        if self.charged_seconds >= self.budget_seconds:
            return True
        return (self.endpoint_policy == "guarded_ceiling" and
                self.budget_seconds - self.charged_seconds < self.reserve_seconds)

    def primary_eligible(self):
        if self.mode == "validation":
            return True
        if self.endpoint_policy in ("guarded_ceiling", "last_at_or_below"):
            return self.charged_seconds <= self.budget_seconds
        return self.charged_seconds <= self.budget_seconds * (1 + self.overrun_tolerance_fraction)

    def commit(self, charged_iteration_seconds, optimizer_steps):
        assert not self.should_stop(), "No further batch may begin after the stopping boundary."
        assert math.isfinite(charged_iteration_seconds) and charged_iteration_seconds > 0
        assert optimizer_steps in (0, 1)
        before, phase = self.charged_seconds, self.phase_index()
        previous_method = self.active_method()
        self.charged_seconds += charged_iteration_seconds
        self.completed_rounds += 1
        self.actual_updates += optimizer_steps
        if phase == 0:
            self.phase1_seconds += charged_iteration_seconds
        else:
            self.phase2_seconds += charged_iteration_seconds
        next_method = self.active_method()
        switched = next_method != previous_method
        if switched:
            assert self.switch_round is None, "Sequential schedule must switch exactly once."
            self.switch_round = self.completed_rounds
            self.switch_at_seconds = self.charged_seconds
        return {"charged_iteration_seconds": charged_iteration_seconds,
                "charged_seconds_before": before, "charged_seconds_after": self.charged_seconds,
                "active_method": previous_method, "next_active_method": next_method,
                "phase_index": phase, "phase_switch_at_boundary": switched,
                "crossed_midpoint": before < self.budget_seconds / 2 <= self.charged_seconds,
                "crossed_budget": before < self.budget_seconds <= self.charged_seconds,
                "primary_endpoint_eligible": self.primary_eligible(),
                "overrun_seconds": max(0.0, self.charged_seconds - self.budget_seconds) if self.mode == "budgeted" else 0.0}

    def state_dict(self):
        return asdict(self)

    def summary(self):
        total = self.charged_seconds
        overrun = max(0.0, total - self.budget_seconds) if self.mode == "budgeted" else 0.0
        return {**self.state_dict(), "active_method_next": self.active_method(),
                "overrun_seconds": overrun,
                "overrun_fraction": overrun / self.budget_seconds,
                "unused_allowance_seconds": max(0.0, self.budget_seconds - total) if self.mode == "budgeted" else None,
                "budget_status": ("validation_only" if self.mode == "validation" else
                                  "overrun_invalid" if overrun > 0 and self.endpoint_policy == "guarded_ceiling" else
                                  "overrun_exceeds_tolerance" if overrun > self.budget_seconds * self.overrun_tolerance_fraction else
                                  "within_allowance" if total <= self.budget_seconds else "crossing_within_tolerance"),
                "overrun_exceeds_tolerance": overrun > self.budget_seconds * self.overrun_tolerance_fraction,
                "realized_phase1_fraction": self.phase1_seconds / total if total else None,
                "midpoint_overshoot_seconds": max(0.0, self.switch_at_seconds - self.budget_seconds / 2)
                    if self.mode == "budgeted" and self.switch_at_seconds is not None else None,
                "budgeted_run": self.mode == "budgeted",
                "within_budget": total <= self.budget_seconds if self.mode == "budgeted" else None,
                "both_sequential_phases_executed": (self.phase1_seconds > 0 and self.phase2_seconds > 0)
                    if self.sequential else None}


class TrainingCursor:
    """Common deterministic per-seed shuffle; reshuffle each exhausted epoch."""
    def __init__(self, rows, seed, epoch=0, offset=0, questions_seen=0):
        self.rows, self.seed = rows, int(seed)
        assert rows and len({row["question_sha256"] for row in rows}) == len(rows)
        self.epoch, self.offset, self.questions_seen = epoch, offset, questions_seen
        self._cached_epoch, self._cached_order = None, None
        assert epoch >= 0 and 0 <= offset < len(rows)
        assert questions_seen == epoch * len(rows) + offset

    def order(self):
        if self._cached_epoch != self.epoch:
            indices = list(range(len(self.rows)))
            random.Random(f"rl-opd-study:{self.seed}:{self.epoch}").shuffle(indices)
            self._cached_epoch, self._cached_order = self.epoch, indices
        return self._cached_order

    def state_dict(self):
        order = self.order()
        return {"seed": self.seed, "epoch": self.epoch, "offset": self.offset,
                "questions_seen": self.questions_seen, "dataset_size": len(self.rows),
                "order_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
                "shuffle_rule": "Python random.Random('rl-opd-study:{seed}:{epoch}').shuffle(range(N))"}

    @classmethod
    def from_state(cls, rows, state):
        cursor = cls(rows, state["seed"], state["epoch"], state["offset"], state["questions_seen"])
        assert cursor.state_dict() == state
        return cursor

    def take(self, count):
        assert count > 0
        selected = []
        for _ in range(count):
            index = self.order()[self.offset]
            selected.append({**self.rows[index], "study_training_index": index,
                             "study_epoch": self.epoch, "study_epoch_offset": self.offset,
                             "study_question_index": self.questions_seen})
            self.offset += 1
            self.questions_seen += 1
            if self.offset == len(self.rows):
                self.epoch += 1
                self.offset = 0
        return selected
