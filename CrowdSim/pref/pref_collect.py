"""Offline preference-data collection for CrowdSim.

Collection strategy
-------------------
Each robot env runs the same route *twice* (run-A then run-B).
After run-B the two fixed-length segments are compared with the ``avoid``
rule and stored as one labelled pair.  The segment length is fixed up-front
via ``--segment-len`` (required).

Usage::

    python CrowdSim/pref_collect.py \\
        --num-envs 4 --headless \\
        --segment-len 50 \\
        --num-episodes 1000 \\
        --save-path output/pref_data/buffer.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.crowd_sim import cfg_path, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrowdSim offline preference collection (avoid rule, same-route pairs)"
    )
    p.add_argument("--env-config",   default="CrowdSim/config/env.yaml")
    p.add_argument("--num-envs",     type=int, default=4)
    p.add_argument("--headless",     action="store_true")
    p.add_argument(
        "--segment-len",
        type=int,
        required=True,
        help=(
            "Fixed number of steps in each trajectory segment.  Both run-A and "
            "run-B are trimmed to this length (anchored at the highest-danger "
            "moment).  Episodes shorter than this value are discarded."
        ),
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Total robot episodes to simulate (each *route* consumes 2 episodes).",
    )
    p.add_argument("--max-pairs",    type=int, default=10_000)
    p.add_argument("--eps",          type=float, default=0.1,
                   help="Tie indifference margin for the avoid rule.")
    p.add_argument("--save-path",    default="output/pref_data/buffer.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config     = load_config(cfg_path(args.env_config))

    from CrowdSim.world.builder import build_env

    print(
        f"[pref_collect] Target: {args.num_episodes} episodes "
        f"({args.num_envs} envs)  seg_len={args.segment_len}"
    )
    t0 = time.time()

    result = build_env(
        config,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    if result is None:
        return
    _, _, nav_manager, runtime = result

    from datetime import datetime
    from CrowdSim.pref.pref_buffer import CrowdSimPrefBuffer

    save_path = args.save_path
    p = Path(save_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_ts = str(p.parent / f"{p.stem}_{ts}{p.suffix}")

    pref_collector = CrowdSimPrefBuffer(
        num_envs=nav_manager.config.num_robots,
        segment_len=args.segment_len,
        max_pairs=args.max_pairs,
        eps=args.eps,
    )

    from CrowdSim.crowd_sim import run_masked_mimic_with_robot_ppo
    run_masked_mimic_with_robot_ppo(
        runtime, nav_manager, config,
        max_episodes=args.num_episodes,
        pref_collector=pref_collector,
        pref_save_path=save_path_ts,
    )

    elapsed = time.time() - t0
    print(f"[pref_collect] Done. {elapsed:.0f}s elapsed.  Data → {save_path_ts}")


if __name__ == "__main__":
    main()
