import sys
import arduino as ad

if sys.platform == "win32":
    import windows as mon

elif sys.platform == "linux":
    import linux as mon


PORT    = "COM5"
BAUD    = 115200

def main():
    cpu = mon.CPU()
    gpu = mon.GPU()
    temp = mon.Temperature()

    arduino = ad.Arduino(PORT, BAUD)
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