"""
YOLOv8 Object Detection — CPU-optimized for Intel UHD / No-CUDA systems
Author  : Production-ready template
Python  : 3.13+
"""

import cv2
import time
import sys
import os
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def print_banner():
    banner = r"""
         YOLOv8 Object Detector  |  CPU-Optimized
    """
    print(banner)


def ask_user_choice(prompt: str, valid_options: list[str]) -> str:
    """
    Prompt user until they enter a valid option (case-insensitive).
    """
    while True:
        answer = input(prompt).strip().lower()
        if answer in valid_options:
            return answer
        print(f"  ⚠  Invalid choice. Please enter one of: {valid_options}")


def ask_file_path(prompt: str) -> str:
    """
    Prompt user for a file path and validate it exists.
    """
    while True:
        path = input(prompt).strip().strip('"').strip("'")
        if Path(path).is_file():
            return path
        print(f"  ⚠  File not found: '{path}'. Please try again.")


def choose_model() -> str:
    """
    Let user choose YOLO model size.
    Smaller = faster on CPU; larger = more accurate.
    """
    print("\n  Available model sizes (all run on CPU):")
    print("    [n] yolov8n  — nano   (~3 MB)  fastest, least accurate")
    print("    [s] yolov8s  — small  (~11 MB) good balance  ← recommended")
    print("    [m] yolov8m  — medium (~26 MB) slower, more accurate")
    choice = ask_user_choice(
        "\n  Choose model size [n/s/m] (default=s): ",
        ["n", "s", "m", ""]
    )
    mapping = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "": "yolov8s.pt"}
    selected = mapping[choice]
    print(f"  ✔  Using model: {selected}")
    return selected


def ask_save_video() -> tuple[bool, str]:
    """
    Ask whether the user wants to save the output video.
    Returns (save_flag, output_path).
    """
    save = ask_user_choice(
        "\n  Save output video? [y/n]: ",
        ["y", "n"]
    )
    if save == "y":
        out_path = input("  Output file path (e.g. output.mp4): ").strip() or "output.mp4"
        return True, out_path
    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
# Core detector class
# ──────────────────────────────────────────────────────────────────────────────

class YOLOv8Detector:
    """
    Wraps an Ultralytics YOLOv8 model for real-time object detection.
    Runs fully on CPU with optional frame-skip and resize optimizations.
    """

    # Colour palette — one vibrant colour per class (cycles if > 80 classes)
    PALETTE = [
        (255,  56,  56), (255, 157,  51), (255, 212,  51), (51, 255, 255),
        ( 51, 153, 255), (153,  51, 255), (255,  51, 153), ( 51, 255, 153),
        (255, 102,   0), (  0, 204, 102), (  0, 102, 204), (204,   0, 102),
    ]

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.45,
        inference_size: int = 416,   # lower = faster on CPU
        frame_skip: int = 0,         # process every N+1 frames (0 = every frame)
    ):
        from ultralytics import YOLO  # deferred import for cleaner startup

        print(f"\n  Loading model '{model_path}' …", end=" ", flush=True)
        self.model = YOLO(model_path)
        # Force CPU — important for Intel UHD / no-CUDA systems
        self.model.to("cpu")
        print("done ✔")

        self.conf = conf_threshold
        self.infer_size = inference_size
        self.frame_skip = frame_skip
        self._skip_counter = 0
        self._last_results = None

        # Class names from the COCO dataset
        self.class_names = self.model.names  # dict {int: str}

    def _colour_for(self, class_id: int) -> tuple[int, int, int]:
        return self.PALETTE[class_id % len(self.PALETTE)]

    def detect(self, frame):
        """
        Run inference on a frame.
        Respects frame_skip: returns cached results on skipped frames.
        """
        self._skip_counter += 1
        if self.frame_skip > 0 and (self._skip_counter % (self.frame_skip + 1) != 0):
            # Return the previous detections (saves CPU time)
            return self._last_results

        results = self.model(
            frame,
            imgsz=self.infer_size,
            conf=self.conf,
            device="cpu",
            verbose=False,
        )
        self._last_results = results
        return results

    def draw(self, frame, results) -> int:
        """
        Draw bounding boxes, class labels and confidence on the frame.
        Returns the number of detections drawn.
        """
        if results is None:
            return 0

        det_count = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                label  = self.class_names.get(cls_id, f"cls{cls_id}")
                colour = self._colour_for(cls_id)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Label background pill
                text = f"{label}  {conf:.0%}"
                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                tag_y = max(y1 - 6, th + 4)
                cv2.rectangle(
                    frame,
                    (x1, tag_y - th - 4),
                    (x1 + tw + 6, tag_y + baseline),
                    colour,
                    cv2.FILLED,
                )
                # White text over coloured pill
                cv2.putText(
                    frame, text,
                    (x1 + 3, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                det_count += 1

        return det_count


# ──────────────────────────────────────────────────────────────────────────────
# FPS tracker
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Exponential moving-average FPS tracker."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._fps = 0.0
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        instant = 1.0 / max(now - self._prev, 1e-9)
        self._fps = self.alpha * instant + (1 - self.alpha) * self._fps
        self._prev = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ──────────────────────────────────────────────────────────────────────────────
# Video writer helper
# ──────────────────────────────────────────────────────────────────────────────

def create_writer(output_path: str, fps: float, width: int, height: int):
    """Create an OpenCV VideoWriter (MP4 via H264 / XVID fallback)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        base = Path(output_path).stem
        output_path = base + ".avi"
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer, output_path


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame, fps: float, det_count: int, source_label: str):
    """Draw semi-transparent HUD bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detections: {det_count}",
                (130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"[{source_label}]  Press Q to quit",
                (w - 260, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

def run(
    source,           # 0 for webcam OR file path string
    model_path: str,
    conf: float,
    infer_size: int,
    frame_skip: int,
    save: bool,
    save_path: str,
    display_width: int = 960,
):
    """
    Open video source, run detection loop, optionally save output.
    """
    # ── Open capture ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n  ✘  Could not open source: {source}")
        print("     Check camera index / file path and try again.")
        sys.exit(1)

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_label = "WEBCAM" if source == 0 else Path(str(source)).name

    print(f"\n  Source   : {source_label}")
    print(f"  Original : {src_w}×{src_h}  @ {src_fps:.1f} FPS")
    print(f"  Inference size: {infer_size}px  |  Frame skip: {frame_skip}")
    print(f"  Press 'Q' in the window to quit.\n")

    # ── Detector & FPS counter ────────────────────────────────────────────────
    detector = YOLOv8Detector(
        model_path=model_path,
        conf_threshold=conf,
        inference_size=infer_size,
        frame_skip=frame_skip,
    )
    fps_counter = FPSCounter()

    # ── Optional VideoWriter ──────────────────────────────────────────────────
    writer = None
    if save:
        # Calculate display dimensions for writer
        scale   = display_width / src_w if src_w > display_width else 1.0
        out_w   = int(src_w * scale)
        out_h   = int(src_h * scale)
        writer, save_path = create_writer(save_path, src_fps, out_w, out_h)
        print(f"  Saving output to: {save_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n  End of stream or read error.")
                break

            # Resize for display (keeps original for inference if infer_size set)
            if src_w > display_width:
                scale   = display_width / src_w
                disp_w  = display_width
                disp_h  = int(src_h * scale)
                frame   = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # Inference
            results = detector.detect(frame)
            det_count = detector.draw(frame, results)

            # FPS + overlay
            fps = fps_counter.tick()
            draw_overlay(frame, fps, det_count, source_label)

            # Save
            if writer is not None:
                writer.write(frame)

            # Display
            cv2.imshow("YOLOv8 Detector  —  Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  'Q' pressed — stopping.")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"  ✔  Output saved to: {save_path}")
        cv2.destroyAllWindows()
        print("  Done.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    # ── CLI overrides (optional, handy for automation) ────────────────────────
    parser = argparse.ArgumentParser(
        description="YOLOv8 CPU object detector",
        add_help=True,
    )
    parser.add_argument("--source",      default=None,  help="0=webcam or path to video file")
    parser.add_argument("--model",       default=None,  help="Model file e.g. yolov8n.pt")
    parser.add_argument("--conf",        type=float, default=0.45, help="Confidence threshold (0-1)")
    parser.add_argument("--infer-size",  type=int,   default=416,  help="Inference image size (px)")
    parser.add_argument("--frame-skip",  type=int,   default=0,    help="Skip N frames between inferences")
    parser.add_argument("--save",        default=None, help="Path to save output video")
    parser.add_argument("--display-width", type=int, default=960,  help="Max display width in pixels")
    args = parser.parse_args()

    # ── Interactive prompts (only if not provided via CLI) ────────────────────

    # 1. Source: webcam or video file?
    if args.source is None:
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │  Input Source                           │")
        print("  └─────────────────────────────────────────┘")
        choice = ask_user_choice(
            "  Use webcam or video file? [webcam/video]: ",
            ["webcam", "video", "w", "v"]
        )
        if choice in ("webcam", "w"):
            source = 0
        else:
            source = ask_file_path("  Enter video file path: ")
    else:
        source = 0 if args.source == "0" else args.source

    # 2. Model size?
    model_path = args.model if args.model else choose_model()

    # 3. Save output?
    if args.save is None:
        save_flag, save_path = ask_save_video()
    else:
        save_flag, save_path = True, args.save

    # 4. Performance settings (show defaults, let user accept or change)
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Performance Settings (CPU-optimized)   │")
    print("  └─────────────────────────────────────────┘")
    print(f"  Inference size : {args.infer_size}px  (smaller = faster)")
    print(f"  Frame skip     : {args.frame_skip}  (0=every frame, 1=every other…)")
    tweak = ask_user_choice("  Change these? [y/n]: ", ["y", "n"])
    infer_size  = args.infer_size
    frame_skip  = args.frame_skip
    if tweak == "y":
        try:
            infer_size = int(input(f"  Inference size [{args.infer_size}]: ").strip() or args.infer_size)
            frame_skip = int(input(f"  Frame skip     [{args.frame_skip}]: ").strip() or args.frame_skip)
        except ValueError:
            print("  Invalid input — using defaults.")

    # ── Launch ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 52)
    run(
        source=source,
        model_path=model_path,
        conf=args.conf,
        infer_size=infer_size,
        frame_skip=frame_skip,
        save=save_flag,
        save_path=save_path,
        display_width=args.display_width,
    )


if __name__ == "__main__":
    main()
