"""Realtime ONNX Teacher replay in Isaac Lab 3 + Newton/MJWarp."""

from __future__ import annotations

import argparse
import inspect
import sys
import threading
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from collect_isaaclab_teacher import (
    OnnxTeacher,
    load_config,
    validate_config,
)
from supermimic.infantry.spec import INFANTRY_SPEC


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Keep these compatible with collect_isaaclab_teacher.load_config().
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--height",
        type=float,
        default=0.20,
        help="Initial height command.",
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=None,
        help="Neutral linear-velocity command; defaults to the configured range neutral.",
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Neutral yaw-rate command; defaults to the configured range midpoint.",
    )

    parser.add_argument(
        "--slow-scale",
        type=float,
        default=0.5,
        help="Command scale without Shift. Shift uses full configured range.",
    )

    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)

    return parser


class KeypadControls:
    """Realtime keypad command source."""

    DIGITS = {"4", "5", "6", "7", "8", "9"}

    # X11 keysyms emitted by the numeric keypad.  NumLock-off keys use the
    # navigation keysyms, so relying on KeyCode.char silently drops them.
    KEYPAD_VK_TO_DIGIT = {
        65460: "4",  # KP_4
        65430: "4",  # KP_Left
        65461: "5",  # KP_5
        65437: "5",  # KP_Begin
        65462: "6",  # KP_6
        65432: "6",  # KP_Right
        65463: "7",  # KP_7
        65429: "7",  # KP_Home
        65464: "8",  # KP_8
        65431: "8",  # KP_Up
        65465: "9",  # KP_9
        65434: "9",  # KP_Page_Up / KP_Prior
    }

    def __init__(
        self,
        *,
        vx_range: tuple[float, float],
        yaw_range: tuple[float, float],
        height_range: tuple[float, float],
        height: float,
        slow_scale: float,
        vx: float | None = None,
        yaw: float | None = None,
    ):
        from pynput import keyboard

        self.keyboard = keyboard
        self.lock = threading.Lock()

        self.pressed: set[str] = set()

        self.space_down = False

        # Policy always runs.
        # This only enables/disables vx/yaw user commands.
        self.enabled = True

        self.vx_range = tuple(map(float, vx_range))
        self.yaw_range = tuple(map(float, yaw_range))
        self.height_range = tuple(map(float, height_range))

        # Same-sign ranges describe a non-zero operating mode.  Their midpoint
        # is the neutral interactive command (rotation [6, 10] -> yaw=8), so
        # the robot keeps spinning when no translation key is held.
        self.vx_default = self.command_default("vx", vx, self.vx_range)
        self.yaw_default = self.command_default("yaw", yaw, self.yaw_range)

        self.height = min(
            max(float(height), self.height_range[0]),
            self.height_range[1],
        )

        self.slow_scale = min(
            max(float(slow_scale), 0.0),
            1.0,
        )

    def control_token(self, key) -> str | None:
        """Normalize keypad, navigation and ordinary keyboard controls."""
        keyboard = self.keyboard
        char = getattr(key, "char", None)
        if isinstance(char, str):
            char = char.lower()

        char_aliases = {
            "w": "8",
            "s": "5",
            "a": "4",
            "d": "6",
            "r": "7",
            "f": "9",
        }
        if char in self.DIGITS:
            return char
        if char in char_aliases:
            return char_aliases[char]

        key_aliases = {
            keyboard.Key.up: "8",
            keyboard.Key.down: "5",
            keyboard.Key.left: "4",
            keyboard.Key.right: "6",
            keyboard.Key.home: "7",
            keyboard.Key.page_up: "9",
        }
        if key in key_aliases:
            return key_aliases[key]

        return self.KEYPAD_VK_TO_DIGIT.get(getattr(key, "vk", None))

    def on_press(self, key) -> None:
        keyboard = self.keyboard
        char = getattr(key, "char", None)
        token = self.control_token(key)

        print(
            "\nKEY EVENT:",
            "key=",
            repr(key),
            "char=",
            repr(char),
            "vk=",
            repr(getattr(key, "vk", None)),
            flush=True,
        )

        with self.lock:
            if key == keyboard.Key.space or char == " ":
                if not self.space_down:
                    self.enabled = not self.enabled
                    self.space_down = True

                    print(
                        "CONTROL:",
                        "ENABLED" if self.enabled else "DISABLED",
                        flush=True,
                    )

                return

            if key in (
                keyboard.Key.shift,
                keyboard.Key.shift_l,
                keyboard.Key.shift_r,
            ):
                self.pressed.add("shift")
                return

            if token is not None:
                self.pressed.add(token)

    def on_release(self, key) -> None:
        keyboard = self.keyboard
        char = getattr(key, "char", None)
        token = self.control_token(key)

        with self.lock:
            if key == keyboard.Key.space or char == " ":
                self.space_down = False
                return

            if key in (
                keyboard.Key.shift,
                keyboard.Key.shift_l,
                keyboard.Key.shift_r,
            ):
                self.pressed.discard("shift")
                return

            if token is not None:
                self.pressed.discard(token)

    @staticmethod
    def neutral_value(value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low <= 0.0 <= high:
            return 0.0
        return 0.5 * (low + high)

    @classmethod
    def command_default(
        cls,
        label: str,
        value: float | None,
        value_range: tuple[float, float],
    ) -> float:
        result = cls.neutral_value(value_range) if value is None else float(value)
        low, high = value_range
        if not low <= result <= high:
            raise ValueError(
                f"{label} command {result} is outside configured range [{low}, {high}]"
            )
        return result

    @staticmethod
    def directional_value(
        direction: float,
        value_range: tuple[float, float],
        scale: float,
    ) -> float:
        low, high = value_range

        if low == high:
            return low

        # A one-sided interval describes an operating band rather than signed
        # direction.  Select its endpoints without scaling outside the band.
        if low > 0.0 or high < 0.0:
            return high if direction > 0.0 else low

        if direction > 0.0:
            return high * scale

        if direction < 0.0:
            return low * scale

        return 0.0

    def command(
        self,
        dt: float,
    ) -> tuple[float, float, float, bool]:
        with self.lock:
            pressed = self.pressed.copy()
            enabled = self.enabled

            # 7 = height up
            # 9 = height down
            height_direction = (
                float("7" in pressed)
                - float("9" in pressed)
            )

            self.height += (
                0.15
                * dt
                * height_direction
            )

            self.height = min(
                max(
                    self.height,
                    self.height_range[0],
                ),
                self.height_range[1],
            )

        speed_scale = (
            1.0
            if "shift" in pressed
            else self.slow_scale
        )

        if enabled:
            # 8 forward
            # 5 backward
            vx_direction = (
                float("8" in pressed)
                - float("5" in pressed)
            )

            # 4 positive yaw
            # 6 negative yaw
            yaw_direction = (
                float("4" in pressed)
                - float("6" in pressed)
            )

            vx = self.directional_value(
                vx_direction,
                self.vx_range,
                speed_scale,
            ) if vx_direction else self.vx_default

            yaw = self.directional_value(
                yaw_direction,
                self.yaw_range,
                speed_scale,
            ) if yaw_direction else self.yaw_default

        else:
            vx = 0.0
            yaw = 0.0

        return (
            vx,
            yaw,
            self.height,
            enabled,
        )


def vector_list(
    tensor: torch.Tensor,
    digits: int = 4,
) -> list[float]:
    values = (
        tensor
        .detach()
        .cpu()
        .flatten()
        .tolist()
    )

    return [
        round(float(value), digits)
        for value in values
    ]


def main() -> None:
    args = make_parser().parse_args()

    config = load_config(args)

    contract, _, physics = validate_config(
        config
    )

    collection = config["collection"]
    asset = config["asset"]
    teacher_cfg = config["teacher"]

    device = str(
        collection.get(
            "device",
            "cuda:0",
        )
    )

    seed = int(
        collection.get(
            "seed",
            42,
        )
    )

    torch.manual_seed(seed)

    from isaaclab.app import launch_simulation

    from supermimic.infantry.isaaclab_runtime import (
        InfantryCollectionSim,
        make_sim_cfg,
    )

    sim_cfg = make_sim_cfg(
        device,
        INFANTRY_SPEC.control_dt,
    )

    command_cfg = config["commands"]

    vx_range = tuple(
        command_cfg["lin_vel_x"]
    )

    yaw_range = tuple(
        command_cfg["ang_vel_z"]
    )

    # The flat collection preset deliberately disables yaw randomization, but
    # this tool is interactive replay.  Match run_policy.py's slow/fast yaw
    # commands instead of turning keypad 4/6 into a permanent zero command.
    if yaw_range == (0.0, 0.0):
        yaw_range = (-4.0, 4.0)

    height_range = tuple(
        command_cfg["height"]
    )

    controls = KeypadControls(
        vx_range=vx_range,
        yaw_range=yaw_range,
        height_range=height_range,
        height=args.height,
        slow_scale=args.slow_scale,
        vx=args.vx,
        yaw=args.yaw,
    )

    listener = controls.keyboard.Listener(
        on_press=controls.on_press,
        on_release=controls.on_release,
        suppress=False,
    )

    listener.start()

    print()
    print(
        "========== ISAAC LAB REALTIME TEACHER =========="
    )
    print("Controls start ENABLED; Space toggles vx & yaw")
    print("Num 8 / W / Up    : forward")
    print("Num 5 / S / Down  : backward")
    print("Num 4 / A / Left  : positive yaw")
    print("Num 6 / D / Right : negative yaw")
    print("Num 7 / R         : height up")
    print("Num 9 / F         : height down")
    print("NumLock may be on or off")
    print("Shift : full command speed")
    print("Ctrl+C: quit")
    print()
    print(
        "Policy remains active when movement command is OFF."
    )
    print(
        "NO dataset writing."
    )
    print(
        "NO automatic reset after falling."
    )
    print(
        "================================================"
    )
    print()

    try:
        with launch_simulation(
            sim_cfg,
            args,
        ):
            env_kwargs = dict(
                num_envs=1,
                usd_path=asset["usd"],
                contract=contract,
                context=torch.tensor(
                    physics["context"],
                    dtype=torch.float32,
                ),
                command_ranges={
                    key: tuple(value)
                    for key, value
                    in command_cfg.items()
                },
                max_episode_steps=2_000_000_000,
                seed=seed,
            )

            # Some runtime revisions take controller explicitly,
            # others already derive it internally.
            init_parameters = inspect.signature(
                InfantryCollectionSim.__init__
            ).parameters

            if "controller" in init_parameters:
                if "controller" not in physics:
                    raise RuntimeError(
                        "InfantryCollectionSim requires "
                        "controller= but physics metadata "
                        "contains no controller."
                    )

                env_kwargs["controller"] = (
                    physics["controller"]
                )

            env = InfantryCollectionSim(
                sim_cfg,
                **env_kwargs,
            )

            policy = OnnxTeacher(
                teacher_cfg["onnx"],
                str(
                    collection.get(
                        "onnx_provider",
                        "CPUExecutionProvider",
                    )
                ),
            )

            # reset() randomizes the command.
            # Override before first policy observation.
            env.command.zero_()

            env.command[:, 2] = (
                controls.height
            )

            policy_ids_cpu = (
                env.policy_joint_ids
                .detach()
                .cpu()
                .tolist()
            )

            policy_joint_names = [
                env.robot.joint_names[index]
                for index
                in policy_ids_cpu
            ]

            print()
            print("POLICY JOINT MAPPING:")

            for policy_index, (joint_id, name) in enumerate(
                zip(policy_ids_cpu, policy_joint_names, strict=True)
            ):
                print(
                    f"  policy[{policy_index}] -> sim[{joint_id}] {name}"
                )

            print()
            print(
                f"vx range     : {vx_range}"
            )
            print(
                f"yaw range    : {yaw_range}"
            )
            print(
                f"height range : {height_range}"
            )
            print(
                f"initial h    : "
                f"{controls.height:.3f}"
            )
            print(
                f"neutral cmd  : "
                f"vx={controls.vx_default:.3f}, "
                f"yaw={controls.yaw_default:.3f}"
            )
            print()

            control_dt = float(
                INFANTRY_SPEC.control_dt
            )

            next_tick = (
                time.perf_counter()
            )

            last_status = 0.0

            with torch.inference_mode():
                step_index = 0
                while args.steps is None or step_index < args.steps:
                    (
                        vx,
                        yaw,
                        height,
                        enabled,
                    ) = controls.command(
                        control_dt
                    )

                    env.command[0, 0] = vx
                    env.command[0, 1] = yaw
                    env.command[0, 2] = height

                    (
                        current,
                        history,
                        teacher_input,
                    ) = env.observation()

                    teacher_action = policy(
                        teacher_input
                    )

                    transition = env.step(
                        teacher_action,
                        current,
                        history,
                    )
                    step_index += 1

                    # Deliberately ignore:
                    #
                    # transition["terminated"]
                    # transition["truncated"]
                    #
                    # Collection resets the robot here.
                    # Realtime replay does NOT.

                    next_state = (
                        transition[
                            "next_state"
                        ][0]
                    )

                    if not torch.isfinite(
                        next_state
                    ).all():
                        print()
                        print(
                            "ERROR: non-finite "
                            "next_state detected.",
                            flush=True,
                        )
                        break

                    now = (
                        time.perf_counter()
                    )

                    diagnostic_run = args.steps is not None and args.steps <= 30
                    if diagnostic_run or now - last_status >= 0.5:
                        q = (
                            env.robot
                            .data
                            .joint_pos
                            .torch[
                                0,
                                env.policy_joint_ids,
                            ]
                        )

                        qd = (
                            env.robot
                            .data
                            .joint_vel
                            .torch[
                                0,
                                env.policy_joint_ids,
                            ]
                        )

                        raw_action = (
                            teacher_action[0]
                        )

                        executed_action = (
                            transition[
                                "executed_action"
                            ][0]
                        )

                        applied_control = (
                            transition[
                                "applied_control"
                            ][0]
                        )

                        obs = (
                            teacher_input[0]
                        )

                        current_obs = (
                            current[0]
                        )

                        obs_finite = bool(
                            torch.isfinite(
                                obs
                            ).all().item()
                        )

                        action_finite = bool(
                            torch.isfinite(
                                raw_action
                            ).all().item()
                        )

                        state_finite = bool(
                            torch.isfinite(
                                next_state
                            ).all().item()
                        )

                        print()
                        print(
                            "=" * 78
                        )

                        print(
                            "COMMAND"
                            f" | "
                            f"step={step_index:04d}"
                            f" | "
                            f"{'ON ' if enabled else 'OFF'}"
                            f" | vx={vx:+.3f}"
                            f" | yaw={yaw:+.3f}"
                            f" | h={height:.3f}"
                        )

                        print(
                            "BASE"
                            f" | "
                            f"x={float(next_state[0]):+.4f}"
                            f" | "
                            f"y={float(next_state[1]):+.4f}"
                            f" | "
                            f"z={float(next_state[2]):+.4f}"
                            f" | "
                            f"finite={state_finite}"
                        )

                        print(
                            "CURRENT OBS"
                            f" | shape="
                            f"{tuple(current_obs.shape)}"
                            f" | min="
                            f"{float(current_obs.min()):+.4f}"
                            f" | max="
                            f"{float(current_obs.max()):+.4f}"
                            f" | norm="
                            f"{float(torch.linalg.vector_norm(current_obs)):.4f}"
                        )

                        print(
                            "TEACHER INPUT"
                            f" | finite="
                            f"{obs_finite}"
                            f" | shape="
                            f"{tuple(obs.shape)}"
                            f" | min="
                            f"{float(obs.min()):+.4f}"
                            f" | max="
                            f"{float(obs.max()):+.4f}"
                            f" | norm="
                            f"{float(torch.linalg.vector_norm(obs)):.4f}"
                        )

                        print(
                            "ONNX RAW ACTION"
                            f" | finite="
                            f"{action_finite}"
                            f" | "
                            f"{vector_list(raw_action)}"
                        )

                        print(
                            "EXECUTED ACTION"
                            f" | "
                            f"{vector_list(executed_action)}"
                        )

                        print(
                            "APPLIED CONTROL"
                            f" | "
                            f"{vector_list(applied_control)}"
                        )

                        print(
                            "JOINT POSITION:"
                        )

                        q_values = (
                            vector_list(q)
                        )

                        for (
                            name,
                            value,
                        ) in zip(
                            policy_joint_names,
                            q_values,
                            strict=True,
                        ):
                            print(
                                f"  "
                                f"{name:30s}"
                                f" q  = "
                                f"{value:+.4f}"
                            )

                        print(
                            "JOINT VELOCITY:"
                        )

                        qd_values = (
                            vector_list(qd)
                        )

                        for (
                            name,
                            value,
                        ) in zip(
                            policy_joint_names,
                            qd_values,
                            strict=True,
                        ):
                            print(
                                f"  "
                                f"{name:30s}"
                                f" qd = "
                                f"{value:+.4f}"
                            )

                        print(
                            "=" * 78,
                            flush=True,
                        )

                        last_status = now

                    # Approximate wall-clock realtime.
                    next_tick += control_dt

                    delay = (
                        next_tick
                        - time.perf_counter()
                    )

                    if delay > 0.0:
                        time.sleep(
                            delay
                        )

                    elif delay < -0.25:
                        # Avoid accumulating
                        # large timing debt.
                        next_tick = (
                            time.perf_counter()
                        )

    except KeyboardInterrupt:
        print()
        print(
            "Stopped by user."
        )

    finally:
        listener.stop()


if __name__ == "__main__":
    main()
