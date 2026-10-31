import psutil
import wmi
import win32pdh

import numpy as np
import time
import serial

import re
from collections import defaultdict


def read_ram():
    ram = psutil.virtual_memory()

    return {
        "usage" : ram.percent,
        "used"  : ram.used,
        "total" : ram.total
    }


class CPU:
    def __init__(self):
        self.query = win32pdh.OpenQuery()

        self.counter = win32pdh.AddEnglishCounter(
            self.query,
            r"\Processor Information(_Total)\% Processor Utility")

        win32pdh.CollectQueryData(self.query)


    def read_cpu(self):
        win32pdh.CollectQueryData(self.query)

        return win32pdh.GetFormattedCounterValue(
            self.counter,
            win32pdh.PDH_FMT_DOUBLE)[1]


    def cleanup(self):
        win32pdh.CloseQuery(self.query)


class GPU:
    def __init__(self):
        self.query = win32pdh.OpenQuery()
        self.counter = win32pdh.AddEnglishCounter(
            self.query,
            r"\GPU Engine(*)\Utilization Percentage")

        self.pattern = re.compile(
            r"luid_(.*?)_phys_(\d+)_eng_(\d+)_engtype_(.+)")

        win32pdh.CollectQueryData(self.query)


    def read_gpu(self):
        win32pdh.CollectQueryData(self.query)

        values = win32pdh.GetFormattedCounterArray(
            self.counter, win32pdh.PDH_FMT_DOUBLE)

        engine_usage = defaultdict(float)
        for instance, usage in values.items():
            match = self.pattern.search(instance)

            if not match:
                continue

            luid, phys, engine, engine_type = match.groups()
            key = (luid, phys, engine)
            engine_usage[key] += usage

        adapter_usage = defaultdict(float)
        for(luid, phys, engine), usage in engine_usage.items():
            adapter_usage[luid] = max(adapter_usage[luid],
                                      usage)

        overall_usage = max(adapter_usage.values(), default=0.0)
        return min(overall_usage, 100.0)


    def cleanup(self):
        win32pdh.CloseQuery(self.query)


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
    cpu = CPU()
    gpu = GPU()

    arduino = Arduino("COM5", 115200)
    arduino.handshake()
    
    try:
        print("Arduino ready, monitoring started...")

        while True:
            if arduino.read_request():
                cpu_usage = cpu.read_cpu()
                ram_usage = read_ram()
                gpu_usage = gpu.read_gpu()

                arduino.send_data(cpu_usage, ram_usage["usage"], gpu_usage, 0)

            # start_time = time.perf_counter()

            # cpu_usage = cpu.read_cpu()
            # ram_usage = read_ram()
            # gpu_usage = gpu.read_gpu()
            
            # print(
            #     f"CPU: {cpu_usage:.1f}%\n"
            #     f"RAM: {ram_usage}\n"
            #     f"GPU: {gpu_usage:.1f}%\n\n",
            # )

            # elapsed = time.perf_counter() - start_time
            # time.sleep(max(0, 1.0 - elapsed))

    except KeyboardInterrupt:
        print("\nMonitoring stopped")

    finally:
        gpu.cleanup()
        cpu.cleanup()
        arduino.cleanup()
    

if __name__ == "__main__":
    main()