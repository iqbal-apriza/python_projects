import psutil
import win32pdh

import re
from collections import defaultdict

def read_ram():
    ram = psutil.virtual_memory()

    return {
        "usage" : ram.percent,
        "used"  : ram.used,
        "total" : ram.total
    }

class Temperature:
    def __init__(self):
        pass

    def read_temp(self):
        return None


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