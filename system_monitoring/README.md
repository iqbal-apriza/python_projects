# Performance Monitoring
The python program for monitoring a CPU, RAM, and GPU usage as well as CPU temperature of the system. The result will then sent into Arduino with serial and displaying it into LCD.

This program originally developed on linux as it reads directly from linux kernel. For windows, the reading is limited to CPU, RAM, and GPU usage only. For temperature reading is still under developed

<!-- This program currently only works in Linux since it uses the reading directly from the `/sys/class/hwmon/` and `/sys/class/drm/` directory for GPU and temperature reading. Support for other operating systems may be added in future updates. -->

## Prerequisite
For linux, before running this program, please make sure the directory on `/sys/class/hwmon/` and `/sys/class/drm/` as well as the device name under `/sys/class/hwmon/hwmon*/name` and change the name under `GPU_TEMP_NAME` and `CPU_TEMP_NAME`

Please see also the [Arduino Program](https://github.com/iqbal-apriza/arduino-project/tree/main/monitoring) related for this project

## Requirements
- `numpy`
- `psutil`
- `pyserial`
- `pathlib`
- `pywin32` (For Windows)

## Running the Program
After clone the repository, make sure the workspace has a file as shown

    system_monitoring
    ├── arduino.py
    ├── linux.py
    ├── main.py
    ├── windows.py
    └── README.md

To run the program

    $ python main.py --port <Arduino Port>

To get an arduino port, please check under device manager for Windows. The arduino port should be something like `COM3` `COM4` etc

For linux, to check arduino port, run

    $ ls /dev/tty*

Look for something like `/dev/ttyACM0` `/dev/ttyACM1` or `/dev/ttyUSB0` `/dev/ttyUSB1`

If you have `arduino-cli`, you can also check arduino port using `arduino-cli`

    $ arduino-cli board list

## Developed Environment
- **OS:** Ubuntu 22.04, Windows 10 22H2
- **Linux Kernel:** 6.8.0-138-generic
- **Python:** 3.10.12
