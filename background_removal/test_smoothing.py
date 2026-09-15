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

        background_weight = np.empty((new_height, new_width), dtype=np.float32)
        prev_mask = None
        smoothed_mask = np.empty((new_height, new_width), dtype=np.float32)
        temporal_alpha = 0.4

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

            t3 = time.perf_counter()
            camera = cv2.resize(camera, (new_width, new_height))

            t4 = time.perf_counter()
            mask = cv2.resize(results.segmentation_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            if prev_mask is None:
                prev_mask = mask.copy()
            else:
                cv2.addWeighted(mask, temporal_alpha, prev_mask, 1.0 - temporal_alpha, 0, smoothed_mask)
                prev_mask, smoothed_mask = smoothed_mask, prev_mask

            mask = prev_mask

            t5 = time.perf_counter()
            # mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

            t6 = time.perf_counter()

            if bg_image is None:
                bg_image = np.zeros(camera.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR

            output_image = bg_image.copy()
            t7 = time.perf_counter()

            roi = output_image[
                y:y + camera.shape[0],
                x:x + camera.shape[1]
            ]
            t8 = time.perf_counter()

            foreground_weight = mask
            np.subtract(1.0, mask, out=background_weight)
            t9 = time.perf_counter()

            blended = cv2.blendLinear(camera, roi, foreground_weight, background_weight)
            t10 = time.perf_counter()

            roi[:] = blended
            t11 = time.perf_counter()

            preprocessing_ms = (t1 - t0) * 1000
            mediapipe_ms = (t2 - t1) * 1000
            cvt_color = (t3 - t2) * 1000
            cam_resize = (t4 - t3) * 1000
            mask_resize_ms = (t5 - t4) * 1000
            mask_clip_ms = (t6 - t5) * 1000
            image_copy_ms = (t7 - t6) * 1000
            roi_ms = (t8 - t7) * 1000
            weight_ms = (t9 - t8) * 1000
            blend_ms = (t10 - t9) * 1000
            assign_ms = (t11 - t10) * 1000

            total_ms = (t11 - t0) * 1000

            print(
                f"Prepocessing : {preprocessing_ms:.2f} ms\n"
                f"MediaPipe    : {mediapipe_ms:.2f} ms\n"
                f"Cam cvt color: {cvt_color:.2f} ms\n"
                f"Cam resize   : {cam_resize:.2f} ms\n"
                f"Mask resize  : {mask_resize_ms:.2f} ms\n"
                f"Mask clip    : {mask_clip_ms:.2f} ms\n"
                f"Image copy   : {image_copy_ms:.2f} ms\n"
                f"ROI          : {roi_ms:.2f} ms\n"
                f"Weight       : {weight_ms:.2f} ms\n"
                f"Blend        : {blend_ms:.2f} ms\n"
                f"Assignment   : {assign_ms:.2f} ms\n"
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