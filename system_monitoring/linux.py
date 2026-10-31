import psutil
import numpy as np
from pathlib import Path

HWMON_PATH      = Path("/sys/class/hwmon")
DRM_PATH        = Path("/sys/class/drm")

GPU_TEMP_NAME   = "amdgpu"
CPU_TEMP_NAME   = "k10temp"

def read_ram():
    ram = psutil.virtual_memory()

    return {
        "usage" : ram.percent,
        "used"  : ram.used,
        "total" : ram.total
    }

class Temperature:
    def __init__(self):
        self.chip_dir = {}

        for device in HWMON_PATH.glob("hwmon*"):
            try:
                chip = (device / "name").read_text().strip()
                self.chip_dir[chip] = device

            except OSError:
                continue


    def read_temp(self):
        dir = self.chip_dir[CPU_TEMP_NAME]
        temps = []

        for input_file in dir.glob("temp*_input"):
            try:
                raw = int(input_file.read_text().strip())
                temps.append(raw)

            except(OSError, ValueError):
                continue

        return np.average(temps)


class CPU:
    def __init__(self):
        self.read_cpu()

    def read_cpu(self):
        return psutil.cpu_percent(interval=None)

    def cleanup(self):
        pass


class GPU:
    def __init__(self):
        self.dir = DRM_PATH / "card1" / "device" / "gpu_busy_percent"

    def read_gpu(self):
        return int(self.dir.read_text().strip())

    def cleanup(self):
        pass