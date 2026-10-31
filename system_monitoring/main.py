import sys
import arduino as ad
import argparse

if sys.platform == "win32":
    import windows as mon

elif sys.platform == "linux":
    import linux as mon

PORT    = "/dev/ttyUSB0"
BAUD    = 115200

def main():
    ap = argparse.ArgumentParser(description="Simple program to monitoring the CPU, RAM, GPU, and CPU temperature to Arduino LCD")
    ap.add_argument("--port", type=str, required=True, help="Arduino port")

    args = ap.parse_args()

    arduino_port = args.port

    cpu = mon.CPU()
    gpu = mon.GPU()
    temp = mon.Temperature()

    arduino = ad.Arduino(arduino_port, BAUD)
    arduino.handshake()

    try:
        print("Arduino ready, monitoring started...")

        while True:
            if arduino.read_request():
                cpu_usage = cpu.read_cpu()
                ram_usage = mon.read_ram()
                gpu_usage = gpu.read_gpu()

                cpu_temp = temp.read_temp()

                arduino.send_data(cpu_usage, ram_usage["usage"], gpu_usage, cpu_temp)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")

    finally:
        gpu.cleanup()
        cpu.cleanup()
        arduino.cleanup()


if __name__ == "__main__":
    main()