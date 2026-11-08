# Landmark format spec (v1)

Both the Python extractor and the web app MUST follow this exactly.

## Input
- Use the RAW camera frame. Never mirror the frame before the landmarker.
  (The app may mirror only the preview shown to the user.)
- Models: MediaPipe Tasks `hand_landmarker` (2 hands) and `pose_landmarker_lite` (1 pose),
  same .task files in Python and in the browser.

## Per-frame vector: 142 floats
| Index | Content |
|---|---|
| 0-62 | Hand slot 0 = MediaPipe handedness "Left": 21 points x (x, y, z) |
| 63-125 | Hand slot 1 = handedness "Right": 21 points x (x, y, z) |
| 126-139 | Pose, 7 points x (x, y): nose, L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist (MediaPipe pose indices 0, 11, 12, 13, 14, 15, 16) |
| 140 | Hand slot 0 present (1/0) |
| 141 | Hand slot 1 present (1/0) |

- If both detected hands have the same label, order them by wrist x in the image (smaller x -> slot 0).
- Missing hand: 63 zeros and flag 0.
- Frames where no pose is found (cannot normalize) are dropped.

## Normalization
- center = midpoint of the two shoulders (x, y)
- scale = distance between the shoulders (x, y)
- x' = (x - center_x) / scale, y' = (y - center_y) / scale; hand z' = z / scale
- Pose points use the same center/scale (x, y only).

## Clip
- Each sign video -> 32 frames by nearest-frame resampling over time
  (np.linspace(0, n-1, 32).round()). Output shape (32, 142).

## Labels
- Label = word folder name from INCLUDE (e.g. "1. Hello"); category = top-level folder.
- Final label map goes in `label_map.json` (word -> integer), shared by both of you.

## Test vectors
- Keep 5 reference videos + their expected .npy in `docs/testvectors/`.
  The JavaScript implementation must reproduce them within 1e-3.
