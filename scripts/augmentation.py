"""
Landmark-space data augmentation that imitates different cameras, angles and framing.

Works on batches of clips shaped (B, T, 142) in the spec-v1 layout (see docs/spec.md):
  0-62 hand slot 0 (21 x xyz) | 63-125 hand slot 1 | 126-139 pose (7 x xy) | 140-141 hand-present flags

Key idea: a real phone camera sees the body from another angle or distance, and the app then
re-normalises by the shoulders. So we (1) transform the points like a camera move would,
(2) re-normalise by the shoulders per frame, exactly as the extractor and the app do.
Everything uses numpy only, so it can be tested without PyTorch.
"""
import numpy as np

POSE_PERM = [0, 2, 1, 4, 3, 6, 5]      # swap left/right pose points (nose stays)

STRONG = dict(
    mirror_p=0.2,                 # left-handed signer: mirror the clip
    rot=12, yaw=25, pitch=15,     # degrees: phone tilt, body turned left/right, camera above/below
    sx=(0.85, 1.15), sy=(0.9, 1.1), shear=0.1,   # lens / aspect / posture differences
    noise_hand=0.01, noise_pose=0.02,            # landmark jitter (pose jitter also changes the scale)
    drop_rate=0.2,                # random missed hand detections (max per-clip rate)
    chunk_p=0.3,                  # a hand lost for 3-8 frames in a row
    top_p=0.25,                   # hand leaves the top of the frame
    speed=(0.75, 1.3), pad=0.15,  # window length factor (speed / trimming) and idle padding at the ends
)
NONE = dict(
    mirror_p=0, rot=0, yaw=0, pitch=0, sx=(1, 1), sy=(1, 1), shear=0,
    noise_hand=0, noise_pose=0, drop_rate=0, chunk_p=0, top_p=0, speed=(1, 1), pad=0,
)


def _u(rng, n, lo, hi):
    return rng.uniform(lo, hi, size=n)


def mirror(x):
    """Flip left-right: negate x, swap hand slots, swap left/right pose points and flags."""
    B, T, _ = x.shape
    m = x.copy()
    m[:, :, 0:63], m[:, :, 63:126] = x[:, :, 63:126], x[:, :, 0:63]
    m[:, :, 0:126:3] *= -1
    pose = x[:, :, 126:140].reshape(B, T, 7, 2)[:, :, POSE_PERM].copy()
    pose[..., 0] *= -1
    m[:, :, 126:140] = pose.reshape(B, T, 14)
    m[:, :, 140:142] = x[:, :, [141, 140]]
    return m


def augment(x, rng, cfg=None):
    c = dict(STRONG)
    if cfg:
        c.update(cfg)
    x = np.asarray(x, dtype=np.float32).copy()
    B, T, _ = x.shape

    sel = rng.random(B) < c["mirror_p"]
    if sel.any():
        x[sel] = mirror(x[sel])
    flags = x[:, :, 140:142] > 0                                   # (B,T,2)
    hands = x[:, :, 0:126].reshape(B, T, 2, 21, 3).copy()
    pose = x[:, :, 126:140].reshape(B, T, 7, 2).copy()

    # ---- camera-like transform: yaw, pitch, in-plane tilt, anisotropic scale, shear ----
    yaw = np.deg2rad(_u(rng, B, -c["yaw"], c["yaw"]))
    pit = np.deg2rad(_u(rng, B, -c["pitch"], c["pitch"]))
    rot = np.deg2rad(_u(rng, B, -c["rot"], c["rot"]))
    sx, sy = _u(rng, B, *c["sx"]), _u(rng, B, *c["sy"])
    sh = _u(rng, B, -c["shear"], c["shear"])

    def tf(X, Y, Z):
        p = lambda a: a.reshape((B,) + (1,) * (X.ndim - 1))
        x1 = X * p(np.cos(yaw)) + Z * p(np.sin(yaw))
        z1 = -X * p(np.sin(yaw)) + Z * p(np.cos(yaw))
        y2 = Y * p(np.cos(pit)) - z1 * p(np.sin(pit))
        z2 = Y * p(np.sin(pit)) + z1 * p(np.cos(pit))
        x3 = x1 * p(np.cos(rot)) - y2 * p(np.sin(rot))
        y3 = x1 * p(np.sin(rot)) + y2 * p(np.cos(rot))
        return p(sx) * (x3 + p(sh) * y3), p(sy) * y3, z2

    hx, hy, hz = tf(hands[..., 0], hands[..., 1], hands[..., 2])
    hands = np.stack([hx, hy, hz], axis=-1)
    px, py, _ = tf(pose[..., 0], pose[..., 1], np.zeros_like(pose[..., 0]))
    pose = np.stack([px, py], axis=-1)

    hands += rng.normal(0, 1, hands.shape).astype(np.float32) * c["noise_hand"]
    pose += rng.normal(0, 1, pose.shape).astype(np.float32) * c["noise_pose"]

    # ---- re-normalise per frame by the shoulders (same as the extractor / app) ----
    L, R = pose[:, :, 1, :], pose[:, :, 2, :]
    ctr = (L + R) / 2
    scl = np.maximum(np.linalg.norm(L - R, axis=-1), 0.2)           # (B,T)
    hands[..., 0:2] -= ctr[:, :, None, None, :]
    hands /= scl[:, :, None, None, None]
    pose -= ctr[:, :, None, :]
    pose /= scl[:, :, None, None]
    hands *= flags[:, :, :, None, None]                              # absent hands stay zero

    # ---- missed hand detections ----
    drop = rng.random((B, T, 2)) < _u(rng, B, 0, c["drop_rate"])[:, None, None]
    for b in np.where(rng.random(B) < c["chunk_p"])[0]:
        s0, ln, slot = int(rng.integers(0, max(1, T - 3))), int(rng.integers(3, 9)), int(rng.integers(0, 3))
        if slot == 2:
            drop[b, s0:s0 + ln, :] = True
        else:
            drop[b, s0:s0 + ln, slot] = True
    top = rng.random(B) < c["top_p"]
    thr = _u(rng, B, -2.2, -1.2)                                     # wrist above this = out of the frame
    drop |= top[:, None, None] & (hands[:, :, :, 0, 1] < thr[:, None, None]) & flags
    keep = ~drop
    hands *= keep[:, :, :, None, None]
    flags = flags & keep

    x[:, :, 0:126] = hands.reshape(B, T, 126)
    x[:, :, 126:140] = pose.reshape(B, T, 14)
    x[:, :, 140:142] = flags.astype(np.float32)

    # ---- time: speed change, trimming and idle padding (edge frames repeat) ----
    f = _u(rng, B, *c["speed"])
    length = f * (T - 1)
    lo = -c["pad"] * (T - 1)
    hi = np.maximum((T - 1) - length + c["pad"] * (T - 1), lo)
    start = lo + rng.random(B) * (hi - lo)
    idx = np.round(start[:, None] + np.arange(T)[None, :] * (length[:, None] / (T - 1)))
    idx = np.clip(idx, 0, T - 1).astype(int)
    return x[np.arange(B)[:, None], idx]
