import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess

from pathlib import Path
import camera_handler as ch
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BASE_DIR = Path(__file__).resolve().parent
BG_COLOR = (192, 192, 192)


def load_background(path, width=1920, height=1080):
    background = cv2.imread(path)

    if background is None:
        raise FileNotFoundError(f"Background file not found: {path}")

    return cv2.resize(background, (width, height))


def resize_to_height(image, target_height):
    h, w = image.shape[:2]
    scale = target_height / h

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(image, (new_width, new_height))


def resize_to_width(image, target_width):
    h, w = image.shape[:2]
    scale = target_width / w

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(image, (new_width, new_height))


def cover_to(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = max(target_width / w, target_height / w)

    new_width = int(w * scale)
    new_height = int(h * scale)

    image = cv2.resize(image, (new_width, new_height))

    x = (new_width - target_width) // 2
    y = (new_height - target_height) // 2

    return image[
        y : y + target_height,
        x : x + target_width
    ]


def contain_to(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)

    new_width = int(w * scale)
    new_height = int(h * scale)

    image = cv2.resize(image, (new_width, new_height))

    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    return image, x, y


def calculate_contain(image_width, image_height, target_width, target_height):
    scale = min(target_width / image_width, target_height / image_height)

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    return new_width, new_height, x, y


def main():
    ap = argparse.ArgumentParser(description="Virtual Background for replacing background virtually with image or blur the background")
    ap.add_argument("--list-devices", action="store_true", help="Show all available devices")
    ap.add_argument("--input-device", default=0, type=int, help="Webcam input device. Use --list-devices to list all available devices")
    ap.add_argument("--output-device", default=0, type=int, help="Virtual webcam device to show the output video. Use --list-device to list all available devices")
    ap.add_argument("--background", type=str, help="Background image to be placed or use blur to blur the background")
    ap.add_argument("--smoothing", type=str, help="Smoothing the segmentation result. This may increase CPU consumption. Use alpha or alpha-temporal")
    ap.add_argument("--canvas-dim", default="1920x1080", type=str, help="Canvas dimension. For example 1920x1080")
    ap.add_argument("--show-debug", action="store_true")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--show-image", action="store_true")

    args = ap.parse_args()

    input_devices, output_devices = ch.list_cameras()
    if args.list_devices:
        ch.print_devices(input_devices, output_devices)
        return

    canvas_dim = args.canvas_dim
    if canvas_dim:
        canvas_width, canvas_height = canvas_dim.split("x")

        canvas_width = int(canvas_width)
        canvas_height = int(canvas_height)

    input_dev = ch.get_video_device(input_devices, args.input_device)
    output_dev = ch.get_video_device(output_devices, args.output_device)

    cap = cv2.VideoCapture(input_dev)

    success, camera = cap.read()
    if not success:
        raise RuntimeError("Cannot read camera frame")

    cam_height, cam_width = camera.shape[:2]

    bg_image = None
    if args.background:
        if args.background != "blur":
            filename = args.background
            bg_path = BASE_DIR / "backgrounds" / filename
            bg_image = load_background(bg_path, canvas_width, canvas_height)

        else:
            bg_image = "blur"
            canvas_width = cam_width
            canvas_height = cam_height

    new_width, new_height, x, y = calculate_contain(cam_width, cam_height, canvas_width, canvas_height)
    print(f"\n\nCamera\t\t: {cam_width} x {cam_height}\n"
          f"Layer\t\t: {new_width} x {new_height}\n"
          f"Position\t: ({x}, {y})\n\n")

    virtual_cam = subprocess.Popen([
        "ffmpeg",
        "-loglevel", "error",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-video_size", f"{canvas_width}x{canvas_height}",
        "-framerate", "30",
        "-i", "-",

        "-pix_fmt", "yuv420p",
        "-f", "v4l2",
        f"{output_dev}"
    ], stdin=subprocess.PIPE)

    prev_t = time.perf_counter()

    try:
        if args.show_image:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            
        with mp_selfie_segmentation.SelfieSegmentation(model_selection = 1) as selfie_segmentation:
            smoothing = None
            if args.smoothing == "alpha":
                smoothing = 1
                prev_mask = None
                background_weight = np.empty((new_height, new_width), dtype=np.float32)
                
            elif args.smoothing == "alpha-temporal":
                smoothing = 2
                prev_mask = None
                background_weight = np.empty((new_height, new_width), dtype=np.float32)
                smoothed_mask = np.empty((new_height, new_width), dtype=np.float32)
                temporal_alpha = 0.4

            while cap.isOpened():
                success, camera = cap.read()
                if not success:
                    print("Ignoring empty camera frame")
                    continue

                camera = cv2.cvtColor(cv2.flip(camera, 1), cv2.COLOR_BGR2RGB)

                camera.flags.writeable = False
                results = selfie_segmentation.process(camera)
                camera.flags.writeable = True

                camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
                mask = results.segmentation_mask

                if args.background != "blur":
                    camera = cv2.resize(camera, (new_width, new_height))
                    mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                if smoothing is not None:
                    if smoothing == 2:
                        if prev_mask is None:
                            prev_mask = mask.copy()

                        else:
                            cv2.addWeighted(mask, temporal_alpha, prev_mask, 1.0 - temporal_alpha, 0, smoothed_mask)
                            prev_mask, smoothed_mask = smoothed_mask, prev_mask

                        mask = prev_mask

                    foreground_weight = mask
                    np.subtract(1.0, mask, out=background_weight)

                else:
                    mask_binary = (mask > 0.1).astype(np.uint8) * 255

                if bg_image is None:
                    bg_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    bg_image[:] = BG_COLOR

                if args.background == "blur":
                    blurred_frame = cv2.GaussianBlur(camera, (31, 31), 0)
                    output_image = blurred_frame.copy()
                    cv2.copyTo(camera, mask_binary, output_image)

                elif args.background != "blur" or args.background is None:
                    output_image = bg_image.copy()
                    roi = output_image[
                        y:y + camera.shape[0],
                        x:x + camera.shape[1]
                    ]

                    if smoothing:
                        blended = cv2.blendLinear(camera, roi, foreground_weight, background_weight)
                        roi[:] = blended

                    else:
                        cv2.copyTo(camera, mask_binary, roi)

                if args.show_fps:
                    now = time.perf_counter()
                    fps = 1 / (now - prev_t)
                    prev_t = now

                    cv2.putText(
                        output_image,
                        f"FPS: {int(fps)}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                virtual_cam.stdin.write(output_image.tobytes())

                if args.show_image:
                    cv2.imshow("Camera", output_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        virtual_cam.stdin.close()
        virtual_cam.wait()


if __name__ == '__main__':
    main()