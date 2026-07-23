"""Extract frames from a video at a configurable sampling rate."""
import argparse
import cv2
import os


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames")
    parser.add_argument("--every_n", type=int, default=2,
                        help="Save every Nth frame (default: 2, i.e. 30fps→15fps)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    i = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % args.every_n == 0:
            cv2.imwrite(os.path.join(args.output_dir, f"{saved:06d}.jpg"), frame)
            saved += 1

        i += 1

    cap.release()
    print(f"Done: read {i} frames, saved {saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
