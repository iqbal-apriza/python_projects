import threading
import time

class Timer:
    def __init__(self, period, callback):
        self.period = period
        self.callback = callback

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)


    def start(self):
        self._thread.start()


    def stop(self):
        self._stop_event.set()
        if threading.current_thread() != self._thread:
            self._thread.join()


    def _run(self):
        next_time = time.monotonic() + self.period

        while not self._stop_event.is_set():
            delay = next_time - time.monotonic()

            if delay > 0:
                if self._stop_event.wait(delay):
                    break

            if self._stop_event.is_set():
                break

            self.callback()
            next_time += self.period

            now = time.monotonic()
            if next_time <= now:
                missed = int((now - next_time) // self.period) + 1
                next_time += missed * self.period


def timer_callback():
    print("Timer Callback")


def main():
    timer = Timer(0.05, timer_callback)

    try:
        print("Timer started")
        timer.start()
        threading.Event().wait()

    except KeyboardInterrupt:
        print("\nProgram terminated. Stop the timer")
        timer.stop()


if __name__ == '__main__':
    main()