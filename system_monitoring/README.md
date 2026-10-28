# Performance Monitoring
The python program for monitoring a CPU, RAM, and GPU usage as well as CPU temperature of the system. The result will then sent into Arduino with serial and displaying it into LCD.

This program currently only works in Linux since it uses the reading directly from the `/sys/class/hwmon/` and `/sys/class/drm/` directory for GPU and temperature reading. Support for other operating systems may be added in future updates.

## Prerequisite
Before running this program, please make sure the directory on `/sys/class/hwmon/` and `/sys/class/drm/` as well as the device name under `/sys/class/hwmon/hwmon*/name` and change the name under `GPU_TEMP_NAME` and `CPU_TEMP_NAME`

Please see also the [Arduino Program](https://github.com/iqbal-apriza/arduino-project/tree/main/monitoring) related for this project

## Requirements
- `numpy`
- `psutil`
- `pyserial`
- `pathlib`

## Developed Environment
- **OS:** Ubuntu 22.04
- **Kernel:** 6.8.0-138-generic
- **Python:** 3.10.12