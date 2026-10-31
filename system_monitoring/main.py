import time
import psutil
import numpy as np
import serial
from pathlib import Path

HWMON_PATH      = Path("/sys/class/hwmon")
DRM_PATH        = Path("/sys/class/drm")

GPU_TEMP_NAME   = "amdgpu"
CPU_TEMP_NAME   = "k10temp"

chip_dir = {}

def scan_hardware(ouptut):
    for device in HWMON_PATH.glob("hwmon*"):
        try:
            chip = (device / "name").read_text().strip()
            ouptut[chip] = device

        except OSError:
            continue


def read_temp(chip_name, chip_dir):
    dir = chip_dir[chip_name]
    temps = []

    for input_file in dir.glob("temp*_input"):
        try:
            raw = int(input_file.read_text().strip())
            temps.append(raw)

        except(OSError, ValueError):
            continue

    return np.average(temps)


def read_cpu():
    return psutil.cpu_percent(interval=None)


def read_ram():
    ram = psutil.virtual_memory()

    return {
        "usage" : ram.percent,
        "used"  : ram.used,
        "total" : ram.total
    }


def read_gpu():
    dir = DRM_PATH / "card1" / "device" / "gpu_busy_percent"
    return int(dir.read_text().strip())


class Arduino:
    def __init__(self, port, baud):
        self.arduino = serial.Serial(port=port, baudrate=baud, timeout=1.0)

    def handshake(self):
        self.connected = False

        while not self.connected:
            if self.arduino.in_waiting >= 1:
                data = self.arduino.read(1)

                if data[0] == 0xAA:
                    self.connected = True
                    self.arduino.write(b'o')


    def read_request(self):
        while self.connected:
            data = self.arduino.read(1)

            if not data:
                continue
            
            if data[0] == 0xAB:
                return True


    def send_data(self, cpu, ram, gpu, temp):
        TxData = bytearray(9)

        cpu = int(round(cpu))
        ram = int(round(ram))
        gpu = int(round(gpu))
        temp = int(temp)

        TxData[0] = 0xFE
        TxData[1] = 0xFD

        TxData[2] = cpu & 0xFF
        TxData[3] = ram & 0xFF
        TxData[4] = gpu & 0xFF

        TxData[5] = temp & 0xFF
        TxData[6] = (temp >> 8) & 0xFF
        TxData[7] = (temp >> 16) & 0xFF

        checksum = 0
        for i in range(2, 8):
            checksum += TxData[i]

        TxData[8] = (~checksum) & 0xFF

        # print(temp)

        self.arduino.write(TxData)


    def cleanup(self):
        self.arduino.close()


def main():
    scan_hardware(chip_dir)
    psutil.cpu_percent(interval=None)

    arduino = Arduino("/dev/ttyUSB0", 115200)
    arduino.handshake()

    try:
        print("Arduino ready, monitoring started...")

        while True:
            if arduino.read_request():
                cpu = read_cpu()
                gpu = read_gpu()
                ram = read_ram()

                cpu_temp = read_temp(CPU_TEMP_NAME, chip_dir)
                gpu_temp = read_temp(GPU_TEMP_NAME, chip_dir)

                arduino.send_data(cpu, ram["usage"], gpu, cpu_temp)


    except KeyboardInterrupt:
        print("\nMonitoring stopped")

    finally:
        arduino.cleanup()


if __name__ == '__main__':
    main()