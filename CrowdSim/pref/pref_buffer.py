"""Preference buffer for CrowdSim — same-route paired runs, fixed-length segments.

Design
------
Each car env runs the *same route* twice in a row (run-A then run-B).
A "route" is a (start_xy, goal_xy) pair sampled once and replayed immediately.
At the end of run-B the two fixed-length segments are compared with the
``avoid`` rule and stored as a preference pair (ep_A, ep_B, label).

Why same-route pairing?
  Comparing two trajectories that solve the same navigation task removes
  confounds from route difficulty.  The only variable is how well each run
  avoided dynamic obstacles.

Segment extraction
  ``segment_len`` is a *required* parameter.  From each episode the segment
  is anchored at the highest-danger moment (minimum moving-obstacle distance
  rolling window) and extended forward for ``segment_len`` steps.  Both
  episodes are processed independently; the anchor is the closest-interaction
  window of *that* run, so the comparison is always about the most critical
  encounter in each attempt.

Labelling (avoid only)
  step score  = tanh((min_dist_mov − MARGIN) × SCALE)
  episode score = Σ step scores          (higher → safer trajectory)
  label = 1  if run-A safer,  0  if run-B safer,  discarded if tie.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

# ANSI colour helpers
_C = {
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
    "cyan":    "\033[96m",
    "red":     "\033[91m",
    "bold":    "\033[1m",
    "reset":   "\033[0m",
}


# ═══════════════════════════════════════════════════════════════════
# Avoid preference rule (only rule supported)
# ═══════════════════════════════════════════════════════════════════

class PreferenceRule:
    """Prefer the trajectory that maintains greater clearance from moving obstacles.

    step score  = tanh((min_dist_mov − MARGIN) × SCALE)
    episode score = sum of all step scores  (higher → safer)

    ``eps`` is the indifference margin: if |score_A − score_B| < eps the
    pair is treated as a tie and discarded.
    """

    MARGIN: float = 1.2   # metres — preferred safety boundary
    SCALE:  float = 2.0   # sharpness of the tanh transition

    def __init__(self, eps: float = 0.1):
        self.eps = float(eps)

    def compute_label(self, ep_a: dict, ep_b: dict) -> int:
        """Return 1 if ep_a preferred, 0 if ep_b preferred, -1 if tie/invalid."""
        ra = self._score(ep_a)
        rb = self._score(ep_b)
        if ra is None or rb is None or abs(ra - rb) < self.eps:
            return -1
        return int(ra > rb)

    def _score(self, ep: dict) -> float | None:
        steps = ep.get("steps", [])
        if not steps:
            return None
        vals = [
            float(np.tanh(
                (float(s.get("min_dist_mov", s.get("min_dist_static", 10.0))) - self.MARGIN)
                * self.SCALE
            ))
            for s in steps
        ]
        return float(np.sum(vals))


# ═══════════════════════════════════════════════════════════════════
# Parallel preference buffer
# ═══════════════════════════════════════════════════════════════════

class CrowdSimPrefBuffer:
    """Multi-env preference buffer using same-route paired runs.

    Each env alternates between run-A and run-B over the *same* route.
    After run-B finishes, the two fixed-length segments are compared and
    stored as one preference pair.

    Usage::

        buf = CrowdSimPrefBuffer(num_envs=4, segment_len=50)
        while running:
            buf.add_step(env_id, step_data)         # every step
            if done:
                should_repeat = buf.end_episode(env_id, goal_xy)
                nav_manager.reset_robot_rl_episodes(done, repeat_mask)
        buf.save("output/pref_data/buffer.pkl")
    """

    def __init__(
        self,
        num_envs: int,
        segment_len: int,            # required — both runs trimmed to this length
        max_pairs: int = 10_000,
        eps: float = 0.1,            # tie indifference margin for the avoid rule
    ):
        if not isinstance(segment_len, int) or segment_len <= 0:
            raise ValueError(
                f"segment_len must be a positive integer, got {segment_len!r}. "
                "Pass --segment-len <N> on the command line."
            )
        self.num_envs    = int(num_envs)
        self.segment_len = int(segment_len)
        self.max_pairs   = int(max_pairs)
        self.rule        = PreferenceRule(eps=eps)

        # Per-env live steps (current run being recorded)
        self._live: list[list[dict]] = [[] for _ in range(num_envs)]

        # Per-env pairing state
        # _run_idx: 0 = waiting for run-A to finish, 1 = waiting for run-B to finish
        self._run_idx:   list[int]            = [0]    * num_envs
        # _route_ep: the extracted segment from run-A (None until run-A finishes)
        self._route_ep:  list[dict | None]    = [None] * num_envs
        # _route_goal: goal_xy used for run-A (for logging only)
        self._route_goal: list[np.ndarray | None] = [None] * num_envs

        # Stored preference pairs: (seg_A, seg_B, label)
        self.pairs: list[tuple[dict, dict, int]] = []

        # Stats
        self.episode_count = 0

    # ── segment extraction ─────────────────────────────────────

    def _extract_segment(self, steps: list[dict]) -> dict | None:
        """Extract a fixed-length segment from *steps*.

        The segment is anchored at the rolling-window minimum of
        ``min_dist_mov`` (highest-danger moment).  Returns None if the
        episode is shorter than ``segment_len``.
        """
        T = self.segment_len
        if len(steps) < T:
            return None
        dists = np.array([s.get("min_dist_mov", 10.0) for s in steps], dtype=np.float32)
        kernel = np.ones(T, dtype=np.float32) / T
        anchor = int(np.argmin(np.convolve(dists, kernel, mode="valid")))
        return {"steps": steps[anchor: anchor + T]}

    # ── step / episode ─────────────────────────────────────────

    def add_step(self, env_id: int, step: dict) -> None:
        """Record one simulation step for *env_id*."""
        self._live[env_id].append(step)

    def end_episode(
        self,
        env_id: int,
        goal_xy: np.ndarray | None = None,
    ) -> bool:
        """Finish the current episode for *env_id* and attempt pairing.

        Returns
        -------
        should_repeat : bool
            True  → run-A just finished; the caller should reset this env
                    to the *same* start and goal (run-B begins).
            False → run-B just finished (or run-A was discarded); the caller
                    should sample a fresh route for this env.
        """
        steps = list(self._live[env_id])
        self._live[env_id].clear()
        self.episode_count += 1

        if self._run_idx[env_id] == 0:
            # ── Run A finished ──────────────────────────────────
            seg = self._extract_segment(steps)
            if seg is None:
                print(
                    f"  {_C['red']}DISCARD run-A{_C['reset']} "
                    f"env={env_id}  too_short={len(steps)} (need {self.segment_len})"
                )
                # Keep _run_idx at 0 and sample a fresh route
                return False

            self._route_ep[env_id]   = seg
            self._route_goal[env_id] = np.asarray(goal_xy, dtype=np.float32) if goal_xy is not None else None
            self._run_idx[env_id]    = 1
            goal_str = (
                f"goal=({goal_xy[0]:.1f},{goal_xy[1]:.1f})"
                if goal_xy is not None else "goal=?"
            )
            print(
                f"  {_C['cyan']}RUN-A{_C['reset']} "
                f"env={env_id}  seg_len={len(seg['steps'])}  {goal_str}  → repeat"
            )
            return True   # signal: replay same route

        else:
            # ── Run B finished ──────────────────────────────────
            self._run_idx[env_id] = 0   # reset regardless of outcome
            seg_a = self._route_ep[env_id]
            self._route_ep[env_id]   = None
            self._route_goal[env_id] = None

            if seg_a is None:
                # Run-A was previously discarded; nothing to pair against
                return False

            seg_b = self._extract_segment(steps)
            if seg_b is None:
                print(
                    f"  {_C['red']}DISCARD run-B{_C['reset']} "
                    f"env={env_id}  too_short={len(steps)} (need {self.segment_len})"
                )
                return False

            label = self.rule.compute_label(seg_a, seg_b)
            if label < 0:
                score_a = self.rule._score(seg_a)
                score_b = self.rule._score(seg_b)
                print(
                    f"  {_C['red']}DISCARD tie{_C['reset']} "
                    f"env={env_id}  score_A={score_a:.3f}  score_B={score_b:.3f}"
                )
                return False

            self.pairs.append((seg_a, seg_b, label))
            while len(self.pairs) > self.max_pairs:
                self.pairs.pop(0)

            lbl_str   = "A≻B" if label == 1 else "B≻A"
            score_a   = self.rule._score(seg_a)
            score_b   = self.rule._score(seg_b)
            print(
                f"  {_C['green']}PAIR{_C['reset']} "
                f"env={env_id}  {lbl_str}  "
                f"score_A={score_a:.3f}  score_B={score_b:.3f}  "
                f"total={len(self.pairs)}"
            )
            return False   # sample a fresh route

    def finish(self) -> None:
        """Discard all in-progress live steps (call at collection end)."""
        for env_id in range(self.num_envs):
            self._live[env_id].clear()
            # Incomplete run-A or run-B — abandon without pairing
            self._route_ep[env_id]   = None
            self._run_idx[env_id]    = 0

    # ── sampling ───────────────────────────────────────────────

    def sample_pairs(self, batch_size: int) -> list[tuple[dict, dict, int]]:
        """Randomly sample *batch_size* labelled pairs (with replacement)."""
        if not self.pairs:
            return []
        indices = [random.randrange(len(self.pairs)) for _ in range(batch_size)]
        return [self.pairs[i] for i in indices]

    # ── persistence ────────────────────────────────────────────

    def save(self, path: str, finish: bool = False) -> None:
        """Serialise the buffer to a pickle file.

        Args:
            finish: If True, call finish() first to flush in-progress episodes.
                    Only set True for the final save at collection end — periodic
                    saves must use finish=False to avoid discarding live data.
        """
        if finish:
            self.finish()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_envs":      self.num_envs,
            "segment_len":   self.segment_len,
            "episode_count": self.episode_count,
            "pairs":         [(a, b, lbl) for a, b, lbl in self.pairs],
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        dist = self.label_distribution()
        print(
            f"{_C['yellow']}[PrefBuffer] Saved {len(self.pairs)} pairs{_C['reset']}"
            f"  seg_len={self.segment_len}"
            f"  label_0={dist[0]:.0f}%  label_1={dist[1]:.0f}%  → {p}"
        )

    @classmethod
    def load(cls, path: str, max_pairs: int = 10_000) -> "CrowdSimPrefBuffer":
        """Deserialise a buffer from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        # segment_len is required in the new format; fall back to 50 for legacy buffers.
        seg_len = data.get("segment_len", 50)
        buf = cls(
            num_envs=data["num_envs"],
            segment_len=seg_len,
            max_pairs=max_pairs,
        )
        buf.episode_count = data["episode_count"]
        buf.pairs         = data["pairs"]
        return buf

    # ── stats ──────────────────────────────────────────────────

    def label_distribution(self) -> dict[int, float]:
        counts: dict[int, int] = {0: 0, 1: 0}
        for _, _, lbl in self.pairs:
            counts[lbl] = counts.get(lbl, 0) + 1
        total = sum(counts.values()) or 1
        return {k: v / total * 100 for k, v in counts.items()}

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        dist = self.label_distribution()
        return (
            f"CrowdSimPrefBuffer("
            f"pairs={len(self.pairs)}, "
            f"episodes={self.episode_count}, "
            f"seg_len={self.segment_len}, "
            f"label_0={dist[0]:.0f}%  label_1={dist[1]:.0f}%)"
        )


# ═══════════════════════════════════════════════════════════════════
# Format converter
# ═══════════════════════════════════════════════════════════════════

def pack_crowdsim_step(
    robot_obs: np.ndarray,        # (obs_dim,) full RL observation
    robot_action: np.ndarray,     # (2,)   [linear, angular]
    robot_xy: np.ndarray,         # (2,)   [x, y]
    robot_yaw: float,
    goal_xy: np.ndarray,          # (2,)   [gx, gy]
    min_dist: float,
    neighbors_xy: list,           # [[x, y], ...]
    depth: np.ndarray | None = None,  # (H, W) depth image, or None
    depth_size: int = 224,        # must match nav_manager.config.rl_depth_size
    min_dist_static: float | None = None,  # nearest static-obstacle clearance (m)
) -> dict:
    """Convert CrowdSim per-step data → buffer step dict.

    ``depth_size`` must equal ``nav_manager.config.rl_depth_size`` so that the
    stored depth shape is identical to what the policy network expects.

    ``min_dist_static`` is the robot's clearance to the nearest *static*
    obstacle (wall/box/cylinder) in metres, computed from the occupancy map
    (nav_manager._min_static_obstacle_dist).  If omitted, it falls back to
    ``min_dist`` (the nearest *dynamic* agent distance) for backward
    compatibility — but callers that have the map should pass it explicitly,
    otherwise ``min_dist_static`` is mislabelled and preference anchoring
    treats a wall near-miss as "safe".
    """
    step = {
        "sys_state":       robot_obs.astype(np.float32),
        "action":          robot_action.astype(np.float32),
        "robot_pose":      [float(robot_xy[0]), float(robot_xy[1]), float(robot_yaw)],
        "goal_position":   [float(goal_xy[0]), float(goal_xy[1])],
        "min_dist_static": float(min_dist_static if min_dist_static is not None else min_dist),
        "min_dist_mov":    float(min_dist),
        "mov_obs":         [[float(x), float(y)] for x, y in neighbors_xy],
    }
    step["depth"] = (
        depth.astype(np.float32)
        if depth is not None
        else np.zeros((depth_size, depth_size), dtype=np.float32)
    )
    return step
