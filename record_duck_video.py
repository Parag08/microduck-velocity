# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a 720p clip of a MicroDuck velocity checkpoint.

Wraps the Isaac Lab 3.0 video recorder: a headless ``newton_gl`` visualizer renders the
scene at ``--width`` x ``--height`` and ``VideoRecorderCfg`` writes the frames to mp4 with
moviepy (needs the repo's ``video`` extra: ``uv sync --extra video``).

Examples (run from ``~/isaaclab-tutor/repo``):

    # untrained policy: falls over, episodes reset
    uv run python ../record_duck_video.py --checkpoint logs/.../model_0.pt \
        --out videos --name untrained --steps 250

    # trained policy: walks on command
    uv run python ../record_duck_video.py --checkpoint logs/.../model_7500.pt \
        --out videos --name trained --steps 500 --forward_speed 0.3
"""

import argparse
import glob
import os

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg, WindowCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLBackendCfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"

parser = argparse.ArgumentParser(description="Record a MicroDuck checkpoint to mp4.")
add_launcher_args(parser)
# --checkpoint / --experiment_name / --resume / --seed come from the rsl_rl CLI group
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--out", default="videos", help="Output directory for the mp4.")
parser.add_argument("--name", default="duck", help="Output filename stem.")
parser.add_argument("--steps", type=int, default=400, help="Env steps to record (50 Hz policy -> 400 = 8 s).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of ducks in frame.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=50, help="Output frame rate (policy runs at 50 Hz).")
parser.add_argument("--forward_speed", type=float, default=0.3, help="Forward velocity command [m/s].")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Yaw rate command [rad/s].")
parser.add_argument("--eye", type=float, nargs=3, default=(0.85, -0.85, 0.42), help="Camera eye [m].")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.12), help="Camera target [m].")
parser.add_argument("--focal_length", type=float, default=50.0, help="Camera focal length [mm].")
parser.add_argument("--no_follow", action="store_true", help="Keep the camera fixed instead of tracking the duck.")
args_cli = parser.parse_args()

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils import to_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import resolve_task_config

# importing the family registers the gym ids
import isaaclab_tasks.contrib.microduck  # noqa: F401,E402


def main() -> None:
    out_dir = os.path.abspath(args_cli.out)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", play_mode=True, overrides=())
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "seed", None) is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # one env, no random pushes, a steady velocity command: the clip is about the policy
    env_cfg.events.push_robot = None
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_forward_envs = 0.0
    env_cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False

    # headless 720p render: the recorder reuses this configured visualizer of its type
    env_cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            backend=NewtonGLBackendCfg(),
            headless=True,
            window=WindowCfg(size=(args_cli.width, args_cli.height)),
            eye=tuple(args_cli.eye),
            lookat=tuple(args_cli.lookat),
            focal_length=args_cli.focal_length,
            enable_live_plots=False,
        )
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="viz:newton_gl",
            output_dir=out_dir,
            fps=args_cli.fps,
            video_length=args_cli.steps,
            video_interval=0,
            output_filename_prefix=args_cli.name,
        )
    ]

    with launch_simulation(env_cfg, {"visualizer": ["newton_gl"]}):
        from isaaclab.envs import ManagerBasedRLEnv

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        device = env.unwrapped.device

        agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        command = torch.tensor(
            [args_cli.forward_speed, 0.0, args_cli.yaw_rate], device=device, dtype=torch.float
        )

        # the headless newton_gl visualizer created for the recorder; camera is set on it directly
        viz = next(
            v for v in env.unwrapped.sim.visualizers if v.cfg.backend.visualizer_type == "newton_gl"
        )
        offset = torch.tensor(args_cli.eye, device=device)
        target_offset = torch.tensor(args_cli.lookat, device=device)

        def follow() -> None:
            root = env.unwrapped.scene["robot"].data.root_pos_w[0]
            eye = (root + offset).tolist()
            target = (root + target_offset).tolist()
            viz.set_camera_view(eye, target)

        env.reset()
        step = 0
        with torch.inference_mode():
            while step < args_cli.steps:
                if not args_cli.no_follow:
                    follow()
                env.unwrapped.command_manager.get_command("base_velocity")[:] = command
                obs = env.get_observations()
                env.step(policy(obs))
                step += 1
                if step % 50 == 0:
                    print(f"[INFO]: recorded {step}/{args_cli.steps} steps", flush=True)

        env.close()

    clips = sorted(glob.glob(os.path.join(out_dir, f"{args_cli.name}_*.mp4")), key=os.path.getmtime)
    if not clips:
        raise RuntimeError(f"no clip written to {out_dir}")
    clip = clips[-1]
    print(f"[INFO]: wrote {clip} ({os.path.getsize(clip) / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
