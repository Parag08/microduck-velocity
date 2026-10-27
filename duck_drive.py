# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Drive a trained MicroDuck policy interactively with the keyboard.

Modelled on ``examples/demos/h1_locomotion.py`` from the Isaac Lab repo, but pointed at the
MicroDuck velocity task and a local checkpoint.

Keys (click the Newton viewer window first, it acts like any desktop app):

    I : walk forward
    U : walk backward
    J : forward + turn left
    L : forward + turn right
    K : stop
    Esc / close window : quit

With ``--auto`` the script ignores the keyboard and runs a scripted command sequence instead,
which is useful for a headless smoke test of the command -> policy -> sim path.
"""

import argparse
import math
import time

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
RL_LIBRARY = "rsl_rl"

parser = argparse.ArgumentParser(description="Drive a MicroDuck policy interactively.")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--num_envs", type=int, default=2, help="Number of ducks to spawn.")
parser.add_argument("--forward_speed", type=float, default=0.3, help="Forward command [m/s] (task range is +/-0.4).")
parser.add_argument("--yaw_rate", type=float, default=0.6, help="Turn command [rad/s] (task range is +/-1.0).")
parser.add_argument("--auto", action="store_true", help="Ignore the keyboard and run a scripted command sequence.")
parser.add_argument("--log_every", type=int, default=30, help="Print commanded vs measured velocity every N steps.")
add_launcher_args(parser)
# the MicroDuck task ships a Newton MJWarp solver config; a visualizer must be requested explicitly
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import resolve_task_config

# importing the family registers the gym ids (the CLI does this for every task package it finds)
import isaaclab_tasks.contrib.microduck  # noqa: F401,E402


def _t(x):
    """Return the torch tensor behind an Isaac Lab data field."""
    return x.torch if hasattr(x, "torch") else x


class DuckDriver:
    """Hold the env, the policy, and the keyboard-to-command mapping."""

    def __init__(self, env_cfg, checkpoint: str) -> None:
        agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils import to_dict

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        runner = OnPolicyRunner(self.env, to_dict(agent_cfg), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        self._viewers = [
            v
            for v in self.env.unwrapped.sim.visualizers
            if getattr(v.cfg.backend, "visualizer_type", None) in {"newton_gl", "newton_rtx"}
        ]
        self._keys_down: set[str] = set()
        self._manual_command = torch.zeros(3, device=self.device)

    def _key_down(self, key: str) -> bool:
        return any(v.is_key_down(key) for v in self._viewers)

    def _key_pressed(self, key: str) -> bool:
        is_down = self._key_down(key)
        was_down = key in self._keys_down
        self._keys_down.add(key) if is_down else self._keys_down.discard(key)
        return is_down and not was_down

    def command_from_keys(self) -> torch.Tensor:
        """Map held keys to a (vx, vy, wz) command in the task's own units."""
        fwd, yaw = args_cli.forward_speed, args_cli.yaw_rate
        self._manual_command.zero_()
        if self._key_down("i"):
            self._manual_command.copy_(torch.tensor([fwd, 0.0, 0.0], device=self.device))
        if self._key_down("u"):
            self._manual_command.copy_(torch.tensor([-fwd, 0.0, 0.0], device=self.device))
        if self._key_down("j"):
            self._manual_command.copy_(torch.tensor([fwd, 0.0, yaw], device=self.device))
        if self._key_down("l"):
            self._manual_command.copy_(torch.tensor([fwd, 0.0, -yaw], device=self.device))
        if self._key_down("k"):
            self._manual_command.zero_()
        return self._manual_command

    def apply_command(self) -> TensorDict:
        """Write the manual command into the velocity term, then recompute observations."""
        command = self.env.unwrapped.command_manager.get_command("base_velocity")
        command[:] = self._manual_command
        return TensorDict(self.env.unwrapped.observation_manager.compute(), batch_size=[self.env.num_envs])

    def measured_velocity(self) -> torch.Tensor:
        """Body-frame linear velocity of env 0 from the articulation data."""
        data = self.env.unwrapped.scene["robot"].data
        return _t(data.root_lin_vel_b)[0]


def main() -> None:
    env_cfg, _ = resolve_task_config(TASK, f"{RL_LIBRARY}_cfg_entry_point", play_mode=True, overrides=())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 1.0e6  # never time out while driving
    env_cfg.curriculum = None  # freeze the task at its final settings
    # deterministic commands: we write them ourselves every step
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_forward_envs = 0.0
    env_cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, {"visualizer": ["newton_gl"]}):
        driver = DuckDriver(env_cfg, args_cli.checkpoint)
        driver.env.reset()
        sim = driver.env.unwrapped.sim
        print(
            "[INFO]: Driving MicroDuck. Click the viewer window, then use I/U forward-back, "
            "J/L turn, K stop. Close the window or Ctrl+C to quit.",
            flush=True,
        )
        step = 0
        # scripted sequence used by --auto: (first_step, last_step, command)
        fwd, yaw = args_cli.forward_speed, args_cli.yaw_rate
        script = [
            (0, 120, torch.tensor([fwd, 0.0, 0.0])),
            (120, 240, torch.tensor([fwd, 0.0, yaw])),
            (240, 360, torch.tensor([fwd, 0.0, 0.0])),
            (360, 10**9, torch.tensor([0.0, 0.0, 0.0])),
        ]
        try:
            while sim.is_running():
                if args_cli.auto:
                    for start, end, cmd in script:
                        if start <= step < end:
                            driver._manual_command.copy_(cmd.to(driver.device))
                            break
                else:
                    driver.command_from_keys()
                with torch.inference_mode():
                    obs = driver.apply_command()
                    action = driver.policy(obs)
                    driver.env.step(action)
                if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                    vel = driver.measured_velocity()
                    print(
                        f"[step {step:5d}] cmd=({driver._manual_command[0]:+.2f},{driver._manual_command[1]:+.2f},"
                        f"{driver._manual_command[2]:+.2f}) measured vx={vel[0].item():+.3f} "
                        f"vy={vel[1].item():+.3f}",
                        flush=True,
                    )
                step += 1
        finally:
            driver.env.close()


if __name__ == "__main__":
    main()
