# MicroDuck velocity training — full specification of the run on this machine

Task id: `IsaacContrib-Velocity-Flat-MicroDuck`
Run: `repo/logs/rsl_rl/microduck_velocity_flat/2026-10-27_14-41-49/` (created 2026-10-27T06:41:49Z, rsl_rl, version 1)
Robot: Pollen Robotics MicroDuck, 14 Dynamixel XL330 servos, `microduck_walk.usd`
Sources: `source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/flat_env_cfg.py`,
`.../agents/rsl_rl_ppo_cfg.py`, `source/isaaclab_assets/isaaclab_assets/robots/microduck.py`,
and the run's own `params/env.yaml` + `params/agent.yaml`.

## 1. Physics and timing

| quantity | value | where |
|---|---|---|
| physics backend | Newton, MJWarp solver | `flat_env_cfg.py:436-454` |
| `sim.dt` | 0.005 s (200 Hz physics) | `:467` |
| `decimation` | 4 | `:465` |
| policy step | 4 × 0.005 = **0.02 s → 50 Hz** | derived |
| `episode_length_s` | 20.0 s → **1000 policy steps** | `:466` |
| solver | `iterations=100, ls_iterations=50, cone=pyramidal, impratio=1.0, integrator=implicitfast, njmax=96, nconmax=16` | `:438-449` |
| scene | `num_envs=4096` default (run used 1024), `env_spacing=2.0` | `:455` |
| gravity / terrain | plane, friction 1.0/1.0 | `:73-79` |

## 2. Robot and actuators

| item | value |
|---|---|
| joints (14, servo order) | `left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle` |
| head subset | `neck_pitch, head_pitch, head_yaw, head_roll` (4) |
| legs subset | the other 10 |
| actuator model | `BamActuatorCfg`, motor `m6` (XL330), `kp_fw=200.0` |
| electrical | `vin=7.4 V`, `vin_range=(6.5, 8.2)`, `vin_drop_gain_range=(0.0, 0.2)`, `vin_min=6.0` |
| motor constants | `kt=0.366`, `resistance=2.811 Ω`, `max_current=1.75 A`, `max_pwm=1.0`, `error_gain=0.00288` |
| friction model | `friction_base=0.00477`, `friction_viscous=0.00536`, `friction_stribeck=0.00468` |
| command delay | `min_delay=3, max_delay=6` steps |
| soft joint limits | `soft_joint_pos_limit_factor=0.9` |
| initial pose | z = 0.125 m, hip_pitch ±0.4579, ankle ∓0.4530, neck/head_pitch 0.3491, hip_roll ∓0.0873 |
| hardware variant | `presets=backlash` → `passive_<joint>_backlash` play hinge in series with each servo |

## 3. Commands (what the policy is told to do each step)

| command | range | resampling | notes |
|---|---|---|---|
| `base_velocity` | `lin_vel_x (-0.4, 0.4) m/s`, `lin_vel_y (-0.3, 0.3) m/s`, `ang_vel_z (-1.0, 1.0) rad/s` | every 3–8 s | `rel_standing_envs=0.02`, `rel_forward_envs=0.2`, `rel_turn_in_place_envs=0.15`, `debug_vis=True` |
| `head_pose` | ±0.05, ±0.05, ±0.07, ±0.015 (4 DOF) | every 2–5 s | widened by curriculum to ±1.1/±1.4/±0.31 |
| `body_pose` | ±0.005 ×3, ±0.05 ×3 (6 DOF) | every 2–5 s | body-pose reward disabled; kept in obs |

The actor receives `velocity_commands` (3), `head_pose_commands` (4), `body_pose_commands` (6) = **13 of its 61 inputs**.

## 4. Observations — the actor's 61 numbers (deploy contract)

| term | dims | content | noise / delay |
|---|---|---|---|
| `imu` | 6 | angular velocity (3, rad/s) + projected gravity (3) | `ang_vel=0.03`, `gravity=0.01`; `max_angle_deg=6.0`, lag 0–1, `update_period=64` |
| `joint_pos` | 14 | joint positions relative to default, encoder-biased | uniform ±0.001 |
| `joint_vel` | 14 | joint velocities | uniform ±0.25, delay lag 1 |
| `actions` | 14 | previous action | — |
| `velocity_commands` | 3 | (vx, vy, wz) | — |
| `head_pose_commands` | 4 | neck/head pose delta | — |
| `body_pose_commands` | 6 | body pose delta | — |
| **total** | **61** | | `enable_corruption=True` |

### Critic (76, training only, privileged)
`base_lin_vel` 3, `base_ang_vel` 3, `projected_gravity` 3, `joint_pos` 14, `joint_vel` 14, `actions` 14,
`velocity_commands` 3, `foot_height` 2, `foot_air_time` 2, `foot_contact` 2, `foot_contact_forces` 6,
`head_pose_commands` 4, `body_pose_commands` 6 = **76** (`enable_corruption=False`).

## 5. Actions — the 14 numbers the policy outputs

`BiasedJointPositionActionCfg(asset_name="robot", joint_names=<14 servos>, preserve_order=True, scale=1.0, use_default_offset=True)`
→ each action ∈ [-1, 1] scaled by 1.0 rad and added to the joint's default pose, then handed to the BAM servo model.

## 6. Rewards — 15 terms (weight × term value, per policy step)

| term | weight | shape |
|---|---|---|
| `track_lin_vel` | +2.0 | `exp(-(‖v_cmd−v‖² + vz²)/0.1)` |
| `track_ang_vel` | +2.0 | `exp(-((wz_cmd−wz)² + wx²+wy²)/0.5)` |
| `upright` | +2.0 | `exp(-‖gravity_xy‖²/0.05)` on `trunk_base` |
| `pose` | +1.0 | exponential joint-deviation, separate standing/walking stds |
| `head_pose_tracking` | +2.0 | `exp(-err/0.5)` on the 4 head joints |
| `air_time` | +3.0 | feet air time windowed to 0.125–0.3 s |
| `body_ang_vel` | −0.05 | L2 of roll/pitch rates |
| `angular_momentum` | −0.02 | L2 |
| `dof_pos_limits` | −1.0 | soft-limit violation |
| `action_rate_l2` | −0.1 → −1.0 (curriculum) | Δaction² |
| `foot_clearance` | −2.0 | deviation from 0.0425 m target height |
| `foot_swing_height` | −0.25·(0.0425/0.02)² ≈ −1.13 | swing-height error |
| `foot_slip` | −0.1 | foot velocity while loaded and commanded to move |
| `self_collisions` | −1.0 | foot-vs-foot contact |
| `head_pose_bias` | 0.0 → −3.0 (curriculum) | slow drift of head pose |

## 7. Terminations

| term | condition |
|---|---|
| `time_out` | truncation at 20 s (`time_out=True`) |
| `fell_over` | trunk tilt beyond **70°** (`bad_orientation`) |
| `nan_state` | non-finite state in `contact_forces` |

## 8. Randomisation (domain randomisation = sim-to-real surface)

| event | mode | range |
|---|---|---|
| `foot_friction` | startup | static & dynamic 0.7–1.3, 64 buckets |
| `encoder_bias` | startup | ±0.015 rad |
| `mass_inertia` | startup | trunk mass ×0.95–1.05, inertia recomputed |
| `reset_base` | reset | x/y ±0.5 m, z ±0.005 m, yaw ±3.14 rad |
| `reset_robot_joints` | reset | zero offset |
| `randomize_com` | reset | ±0.003 m, widened by curriculum to ±0.015 |
| `randomize_head_com` | reset | ±0.003 m → ±0.01 |
| `randomize_joint_friction` | reset | BAM friction ×0.9–1.1 |
| `randomize_armature` | reset | ×0.9–1.1 |
| `push_robot` | interval 3–6 s | ±0.3 m/s in x and y (disabled in `play_mode()`) |

## 9. Curriculum — the task changes until iteration 2000

| item | 500 | 600 | 750 | 1000 | 1250 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|
| `action_rate_l2.weight` | −0.2 | | −0.4 | −0.6 | −0.8 | −1.0 | |
| `head_pose_bias.weight` | | −1.0 | | −2.0 | | −3.0 | |
| `rel_standing_envs` | 0.05 | | 0.10 | 0.15 | | 0.20 | 0.25 |
| `head_pose.ranges` | ±0.17 | | | ±0.39 | | ±0.72 | ±1.1 |
| `com_range` | ±0.005 | | | ±0.01 | | ±0.015 | |
| `head_com_range` | ±0.005 | | | ±0.01 | | | |

(Values in iterations; internally multiplied by 24 steps per iteration.)

## 10. PPO and training budget (from `params/agent.yaml`)

| setting | value |
|---|---|
| library / runner | rsl_rl `OnPolicyRunner` |
| `num_steps_per_env` | 24 |
| `max_iterations` | 50,000 |
| `save_interval` | 250 (→ 200 checkpoints if finished) |
| actor / critic | MLP `[512, 256, 128]`, ELU, `obs_normalization=True` |
| distribution | Gaussian, `init_std=1.0`, scalar std |
| epochs / minibatches | 5 / 4 |
| learning rate | 1e-3, `schedule="adaptive"`, `desired_kl=0.01` |
| γ / λ / entropy | 0.99 / 0.95 / 0.01 |
| gradient clip | `max_grad_norm=1.0`, Adam |
| value loss | coef 1.0, clipped |
| seed | 42 |

## 11. What the run actually achieved (iteration 7,638, stopped early)

| metric | value |
|---|---|
| Mean reward | 96.48 |
| Mean episode length | 918.4 / 1000 |
| `error_vel_xy` | 0.2141 m/s |
| `error_vel_yaw` | 0.5199 rad/s |
| `fell_over` | 0.1392 |
| `time_out` | 0.8608 |
| Mean action std | 0.21 |
| throughput | ~41,400 steps/s at 1024 envs, 0.60 s/iteration |

## 12. Artifacts

- checkpoints: `model_0.pt` … `model_7500.pt` (every 250 iterations, ~4.8 MB each)
- `params/env.yaml`, `params/agent.yaml` — resolved configs (needed for deployment)
- `run.json`, `git/repo.diff` — provenance
- `exported/policy.pt` (TorchScript, normalizer included), `exported/policy.onnx` (+ `.data`)
- console log: `~/duck_train.log`; driver script: `~/isaaclab-tutor/duck_drive.py`

## 13. How to reproduce

Train (this machine: 8 GB RTX 3070 Laptop, Newton/Warp backs the physics, no Isaac Sim; the PPO
config itself asks for 50,000 iterations, the run above was stopped by hand at 7,638):

```bash
cd ~/isaaclab-tutor/repo
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-MicroDuck \
  --num_envs 1024 --logger tensorboard
# artifacts: logs/rsl_rl/microduck_velocity_flat/<timestamp>/
```

Drive a checkpoint (from the repo venv so `isaaclab` imports resolve; `PYGLFW_LIBRARY_VARIANT=x11`
is required on this Wayland desktop or newton_gl dies with `undefined symbol: glfwGetX11Window`):

```bash
cd ~/isaaclab-tutor/repo
DISPLAY=:0 PYGLFW_LIBRARY_VARIANT=x11 uv run python ../duck_drive.py \
  --checkpoint logs/rsl_rl/microduck_velocity_flat/2026-10-27_14-41-49/model_7500.pt \
  --num_envs 2 --log_every 20
# click the Newton viewer window first, then I/U forward-back, J/L turn, K stop
```

Scripted smoke test (no keyboard): add `--auto` — 120 steps forward, 120 forward+turn, 120 forward,
then stop; it logs commanded vs measured base velocity every `--log_every` steps.
