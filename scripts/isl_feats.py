"""
Extra per-frame features that describe WHERE A FINGER POINTS and HOW THE HANDS MOVE.

Why: the cross-validation showed that 11 of the 17 mistakes are pointing signs
(I <-> you, he <-> she, you <-> you (plural)). The raw 142 numbers contain the information, but only
indirectly. Here it is made explicit:
  dir   index-finger direction (unit vectors), with and without the wrist as the start point
  palm  which way the palm faces (unit normal of the palm plane)
  pos   where the index fingertip is relative to the nose and to the chest centre
  ext   how far the index and the middle finger are stretched out (pointing = index out, middle curled)
  vel   how fast the wrist and the index fingertip move from frame to frame

The function uses only +, -, *, /, ** and slicing, so the very same code runs on numpy arrays
(tested without PyTorch) and on torch tensors (inside the model, which also keeps ONNX export simple:
the app still sends the same 142 numbers per frame).

Input layout per frame (spec v1): 0-62 hand slot 0 | 63-125 hand slot 1 (21 points x xyz each)
                                  126-139 pose (7 points x xy) | 140-141 hand-present flags.
"""
EPS = 1e-6
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP = 0, 4, 5, 8, 9, 12, 17
NOSE = 0  # first pose point
GROUPS = ("dir", "palm", "pos", "ext", "vel")


def _pt(x, slot, i):
    o = slot * 63 + 3 * i
    return x[..., o:o + 3]


def _norm(v):                      # length of 3-vectors, shape (..., 1)
    return ((v * v).sum(-1)[..., None] + EPS) ** 0.5


def _unit(v):
    return v / _norm(v)


def _cross(a, b, cat):
    return cat([a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
                a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
                a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]], -1)


def n_features(groups):
    per_hand = {"dir": 6, "palm": 3, "pos": 3, "ext": 2, "vel": 6}
    return 2 * sum(per_hand[g] for g in groups)


def features(x, groups, cat, shift_time):
    """x: (B, T, 142) numpy array or torch tensor -> (B, T, n_features(groups)).
    cat(list, axis) concatenates; shift_time(a) returns a delayed by one frame (first frame copies itself)."""
    out = []
    nose_xy = x[..., 126 + 2 * NOSE:126 + 2 * NOSE + 2]
    for slot in (0, 1):
        present = x[..., 140 + slot:141 + slot]                     # (B, T, 1)
        wrist, tip, mcp = _pt(x, slot, WRIST), _pt(x, slot, INDEX_TIP), _pt(x, slot, INDEX_MCP)
        mid_tip, mid_mcp, pinky = _pt(x, slot, MIDDLE_TIP), _pt(x, slot, MIDDLE_MCP), _pt(x, slot, PINKY_MCP)
        if "dir" in groups:
            out += [_unit(tip - mcp) * present, _unit(tip - wrist) * present]
        if "palm" in groups:
            out += [_unit(_cross(mcp - wrist, pinky - wrist, cat)) * present]
        if "pos" in groups:
            to_nose = tip[..., 0:2] - nose_xy
            chest = _norm(cat([tip[..., 0:2], tip[..., 0:1] * 0], -1))
            out += [to_nose * present, chest * present]
        if "ext" in groups:
            base = _norm(mcp - wrist)
            out += [_norm(tip - wrist) / base * present, _norm(mid_tip - wrist) / _norm(mid_mcp - wrist) * present]
        if "vel" in groups:
            prev_present = shift_time(present)
            both = present * prev_present                         # only when the hand is seen in both frames
            out += [(wrist - shift_time(wrist)) * both, (tip - shift_time(tip)) * both]
    return cat(out, -1)


# ---- numpy versions of the two helpers (the torch versions live in cv_eval.py) -------------------------
def np_cat(lst, axis):
    import numpy as np
    return np.concatenate(lst, axis=axis)


def np_shift(a):
    import numpy as np
    return np.concatenate([a[:, :1], a[:, :-1]], axis=1)
