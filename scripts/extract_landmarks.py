"""
Extract normalized hand + upper-body landmarks from INCLUDE videos.

Output per video: .npy of shape (T, 142), T = 32 by default.
Feature layout per frame (see spec.md):
  [0:63]    hand slot 0 ("Left" label)  21 points x (x, y, z)
  [63:126]  hand slot 1 ("Right" label) 21 points x (x, y, z)
  [126:140] pose: nose, L/R shoulder, L/R elbow, L/R wrist -> 7 points x (x, y)
  [140:142] hand presence flags (slot 0, slot 1)

Usage:
  python extract_landmarks.py --videos F:\\include_data\\videos --out F:\\include_data\\landmarks
  python extract_landmarks.py --videos ... --out ... --limit 20      # quick test
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# MediaPipe pose indices: nose, l/r shoulder, l/r elbow, l/r wrist
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
L_SH, R_SH = 1, 2  # positions inside POSE_IDX for the shoulders
N_FEATURES = 126 + 14 + 2
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}


def ensure_model(path: Path, url: str):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def make_detectors(model_dir: Path):
    hand_path = ensure_model(model_dir / "hand_landmarker.task", HAND_URL)
    pose_path = ensure_model(model_dir / "pose_landmarker_lite.task", POSE_URL)
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        ))
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        ))
    return hand, pose


def frame_features(hand_res, pose_res):
    """Return (feature_vector[142], has_pose). Not mirrored: raw camera frame."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    # ---- pose (also gives the normalization reference) ----
    if not pose_res.pose_landmarks:
        return None  # cannot normalize this frame
    pl = pose_res.pose_landmarks[0]
    pose_xy = np.array([[pl[i].x, pl[i].y] for i in POSE_IDX], dtype=np.float32)
    center = (pose_xy[L_SH] + pose_xy[R_SH]) / 2.0
    scale = float(np.linalg.norm(pose_xy[L_SH] - pose_xy[R_SH]))
    if scale < 1e-4:
        return None
    feat[126:140] = ((pose_xy - center) / scale).reshape(-1)

    # ---- hands ----
    slots = {}
    hands = []
    for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        hands.append((handed[0].category_name, pts))
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        # same label twice: order by wrist x in the image
        hands.sort(key=lambda h: h[1][0, 0])
        hands = [("Left", hands[0][1]), ("Right", hands[1][1])]
    for label, pts in hands:
        slot = 0 if label == "Left" else 1
        slots[slot] = pts
    for slot, pts in slots.items():
        norm = pts.copy()
        norm[:, 0] = (pts[:, 0] - center[0]) / scale
        norm[:, 1] = (pts[:, 1] - center[1]) / scale
        norm[:, 2] = pts[:, 2] / scale
        feat[slot * 63:(slot + 1) * 63] = norm.reshape(-1)
        feat[140 + slot] = 1.0
    return feat


def process_video(path: Path, model_dir: Path, T: int, stride: int, width: int):
    hand, pose = make_detectors(model_dir)  # new detectors reset tracking per video
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    feats, total, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        if (i := i + 1) % stride:
            continue
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((total - 1) * 1000.0 / fps)
        f = frame_features(hand.detect_for_video(img, ts),
                           pose.detect_for_video(img, ts))
        if f is not None:
            feats.append(f)
    cap.release()
    hand.close()
    pose.close()
    if not feats:
        return None, total, 0.0
    arr = np.stack(feats)
    idx = np.linspace(0, len(arr) - 1, T).round().astype(int)  # nearest-frame resample
    hand_rate = float((arr[:, 140:142].max(axis=1) > 0).mean())
    return arr[idx], total, hand_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--width", type=int, default=640, help="downscale frames to this width")
    ap.add_argument("--limit", type=int, default=0, help="process only N videos (test)")
    args = ap.parse_args()

    vids = sorted(p for p in args.videos.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if args.limit:
        vids = vids[:args.limit]
    print(f"{len(vids)} videos found")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    rows = []
    for v in tqdm(vids):
        rel = v.relative_to(args.videos)
        npy = (args.out / rel).with_suffix(".npy")
        category = rel.parts[0] if len(rel.parts) > 2 else ""
        label = rel.parts[-2] if len(rel.parts) > 1 else ""
        if npy.exists():
            arr = np.load(npy)
            rows.append([str(rel), category, label, str(npy.relative_to(args.out)),
                         "", "%.3f" % float((arr[:, 140:142].max(axis=1) > 0).mean()), "cached"])
            continue
        res = process_video(v, args.models, args.frames, args.stride, args.width)
        if res is None or res[0] is None:
            rows.append([str(rel), category, label, "", "", "0", "FAILED"])
            continue
        arr, total, hand_rate = res
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, arr)
        rows.append([str(rel), category, label, str(npy.relative_to(args.out)),
                     total, "%.3f" % hand_rate, "ok"])
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "category", "label", "npy_path",
                    "n_frames", "hand_frame_rate", "status"])
        w.writerows(rows)
    failed = sum(r[-1] == "FAILED" for r in rows)
    print(f"Done. {len(rows) - failed} saved, {failed} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
