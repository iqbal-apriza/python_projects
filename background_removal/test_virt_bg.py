import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess

from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BASE_DIR = Path(__file__).resolve().parent

BG_COLOR = (192, 192, 192)

BG_FILENAME = "Home.png"
BG_PATH = BASE_DIR / "backgrounds" / BG_FILENAME

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

cap = cv2.VideoCapture(0)


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
    bg_image = load_background(BG_PATH, CANVAS_WIDTH, CANVAS_HEIGHT)
    prev_t = time.perf_counter()

    success, camera = cap.read()
    if not success:
        raise RuntimeError("Cannot read camera frame")

    cam_height, cam_width = camera.shape[:2]
    new_width, new_height, x, y = calculate_contain(cam_width, cam_height, CANVAS_WIDTH, CANVAS_HEIGHT)
    print(f"\n\nCamera\t\t: {cam_width} x {cam_height}\n"
          f"Layer\t\t: {new_width} x {new_height}\n"
          f"Position\t: ({x}, {y})\n\n")

    with mp_selfie_segmentation.SelfieSegmentation(model_selection = 1) as selfie_segmentation:
        bg_image = load_background(BG_PATH)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

        alpha_smooth = None

        while cap.isOpened():
            success, camera = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            t0 = time.perf_counter()
            camera = cv2.cvtColor(cv2.flip(camera, 1), cv2.COLOR_BGR2RGB)

            t1 = time.perf_counter()
            camera.flags.writeable = False
            results = selfie_segmentation.process(camera)
            camera.flags.writeable = True

            t2 = time.perf_counter()
            camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
            camera = cv2.resize(camera, (new_width, new_height))
            mask = cv2.resize(results.segmentation_mask, (new_width, new_height))

            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            alpha = np.clip(mask, 0.0, 1.0).astype(np.float32)

            if alpha_smooth is None:
                alpha_smooth = alpha.copy()
            else:
                beta = 0.3
                alpha_smooth = ((1.0 - beta) * alpha_smooth + beta * alpha)

            t3 = time.perf_counter()
            t31 = time.perf_counter()

            if bg_image is None:
                bg_image = np.zeros(camera.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR

            output_image = bg_image.copy()
            t32 = time.perf_counter()

            roi = output_image[
                y:y + camera.shape[0],
                x:x + camera.shape[1]
            ]

            t33 = time.perf_counter()
            blended = cv2.blendLinear(camera, roi, alpha_smooth, 1.0 - alpha_smooth)
            roi[:] = blended

            t34 = time.perf_counter()

            t4 = time.perf_counter()
            preprocessing_ms = (t1 - t0) * 1000
            mediapipe_ms = (t2 - t1) * 1000
            resize_ms = (t3 - t2) * 1000
            composite_ms = (t4 - t3) * 1000

            mask_ms = (t31 - t3) * 1000
            copy_ms = (t32 - t31) * 1000
            roi_ms = (t33 - t32) * 1000
            blend_ms = (t34 - t33) * 1000
            assignment_ms = (t4 - t34) * 1000

            total_ms = (t4 - t0) * 1000

            print(
                f"Prepocessing : {preprocessing_ms:.2f} ms\n"
                f"MediaPipe    : {mediapipe_ms:.2f} ms\n"
                f"Resize       : {resize_ms:.2f} ms\n"
                f"Composite    : {composite_ms:.2f} ms\n"
                f"    Mask     : {mask_ms:.2f} ms\n"
                f"    Copy     : {copy_ms:.2f} ms\n"
                f"    ROI      : {roi_ms:.2f} ms\n"
                f"    Blend    : {blend_ms:.2f} ms\n"
                f"    Assign   : {assignment_ms:.2f} ms\n"
                f"Total        : {total_ms:.2f}\n"
            )

            cv2.imshow("Camera", output_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            curr_t = time.perf_counter()
            fps = 1 / (curr_t - prev_t)
            prev_t = curr_t

            print(f"FPS: {fps:.2f}\n\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()