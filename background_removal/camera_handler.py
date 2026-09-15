import argparse
import cv2
import subprocess


def list_cameras():
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        capture_output=True,
        text=True
    )

    input_devices = []
    output_devies = []
    dev_name = None
    vid_dev = []

    for line in result.stdout.splitlines():
        if line and not line.startswith("\t"):
            dev_name = line.rstrip(":")

        elif line.startswith("\t/dev/video"):
            vid_dev = line.strip()

            if check_input_compatibility(vid_dev):
                input_devices.append({
                    "name" : dev_name,
                    "device" : vid_dev
                })

            elif check_output_compatibility(vid_dev):
                output_devies.append({
                    "name" : dev_name,
                    "device" : vid_dev
                })

    return input_devices, output_devies


def check_input_compatibility(device:str):
    v4l2 = subprocess.Popen(
        ["v4l2-ctl", "-d", device, "--info"],
        stdout=subprocess.PIPE,
        text=True
    )

    grep = subprocess.run(
        ["grep", "-A4", "Device Caps"],
        stdin=v4l2.stdout,
        capture_output=True,
        text=True
    )

    if "Video Capture" in grep.stdout:
        return True

    return False


def check_output_compatibility(device:str):
    v4l2 = subprocess.Popen(
        ["v4l2-ctl", "-d", device, "--info"],
        stdout=subprocess.PIPE,
        text=True
    )

    grep = subprocess.run(
        ["grep", "-A4", "Device Caps"],
        stdin=v4l2.stdout,
        capture_output=True,
        text=True
    )

    if "Video Output" in grep.stdout:
        return True

    return False


def print_devices(input_devices, output_devices):
    print("==== Available camera devices ====\n")

    n_dev = len(input_devices)

    print("--- Input ---")
    for i in range(n_dev):
        name = input_devices[i]['name']
        vid = input_devices[i]['device']

        print(f"[{i}] {name}\n"
              f"    >> {vid}\n")

    n_dev = len(output_devices)

    print("--- Output ---")
    for i in range(n_dev):
        name = output_devices[i]['name']
        vid = output_devices[i]['device']

        print(f"[{i}] {name}\n"
                f"    >> {vid}\n")


def get_video_device(dev, idx):
    return dev[idx]['device']


def main():
    input_devices, output_devices = list_cameras()
    port = get_video_device(output_devices, 0)

    print(port)

    
if __name__ == '__main__':
    main()