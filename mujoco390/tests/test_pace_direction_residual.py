"""Ensure directional comparisons require compatible pose and velocity."""

import math

from pace_id.direction_residual import match_error, select_pairs


def block(i, q, v, phase=4, run=0):
  return {"id": i, "run": run, "phase": phase, "moving": True,
          "q": q, "v": v, "L": 0.23, "phi": 0.5}


def test_matching_accepts_wrapped_joint_coordinates():
  a = block(0, [math.pi-0.003, 0.5], [0.3, 0.2])
  b = block(1, [-math.pi+0.003, 0.5], [-0.3, -0.2])
  assert match_error(a, b) is not None


def test_matching_rejects_cross_segment_speed_and_pose_mismatch():
  a = block(0, [0.1, 0.5], [0.3, 0.2])
  b = block(1, [0.1, 0.5], [-0.3, -0.2])
  assert match_error(a, b) is not None
  assert match_error(a, {**b, "run": 1}) is None
  assert match_error(a, {**b, "v": [-0.6, -0.2]}) is None
  assert match_error(a, {**b, "v": [-0.3, 0.2]}) is None
  assert match_error(a, {**b, "q": [0.15, 0.5]}) is None


def test_selected_pairs_do_not_reuse_blocks():
  blocks = [block(i, [0.1, 0.5], v) for i, v in enumerate(
    ([0.3, 0.2], [0.3, 0.2], [-0.3, -0.2])
  )]
  pairs = select_pairs(blocks)
  assert len(pairs) == 1
  assert pairs[0]["positive"] == 0
  assert pairs[0]["negative"] == 2
