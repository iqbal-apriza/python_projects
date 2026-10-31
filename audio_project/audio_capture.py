import subprocess
import numpy as np
import time
import argparse

SOURCE          = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo.monitor"
BUFFER_SIZE     = 2048
FRAME_SIZE      = 4

ALPHA           = 0.0
MAX_BAR         = 70
CUTOFF_DB       = -60

BAR_CHAR        = "#"
BAR_PLACE       = "-" * MAX_BAR

CLEAR_SCREEN    = "\033[2J"
CURSOR_HOME     = "\033[H"
CLEAR_LINE      = "\033[K"
RESET_COLOR     = "\033[0m"
COLOR_GREEN     = "\033[32m"
COLOR_YELLOW    = "\033[33m"
COLOR_RED       = "\033[31m"

# MEASURE_MODE    = "linear"

def mapping(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def rms(val):
    data_len = len(val)
    val_sqr = val ** 2

    return np.sqrt(np.sum(val_sqr) / data_len)


def render_terminal(val_l, val_r, mode):
    peak_l = " "
    peak_r = " "
    
    if mode == "decibel":
        bar_l = np.round(mapping(val_l, CUTOFF_DB, 0, 0, MAX_BAR)).astype(int)
        bar_r = np.round(mapping(val_r, CUTOFF_DB, 0, 0, MAX_BAR)).astype(int)

        if val_l >= 0:
            peak_l = "\033[31m#\033[0m"

        if val_r >= 0:
            peak_r = "\033[31m#\033[0m"

        sl, vl = ("-", abs(val_l)) if val_l < 0 else (" ", val_l)
        sr, vr = ("-", abs(val_r)) if val_r < 0 else (" ", val_r)

        unit = "dB"

    if mode == "linear":
        bar_l = np.round(mapping(val_l, 0, 1, 0, MAX_BAR)).astype(int)
        bar_r = np.round(mapping(val_r, 0, 1, 0, MAX_BAR)).astype(int)

        if val_l >= 1.0:
            peak_l = "\033[31m#\033[0m"

        if val_r >= 1.0:
            peak_r = "\033[31m#\033[0m"

        vl = val_l
        vr = val_r

        sl = sr = ''
        unit = ''

    show_bar_l = BAR_CHAR * int(bar_l)
    show_bar_r = BAR_CHAR * int(bar_r)

    print(
        f"{CURSOR_HOME}"
        f"Sound Device\t: {SOURCE}\n"
        f"Audio L\t\t: {sl}{vl:4.2f} {unit}{CLEAR_LINE}\n"
        f"Audio R\t\t: {sr}{vr:4.2f} {unit}{CLEAR_LINE}\n\n"
        f"L: {BAR_PLACE}{peak_l}\r"
        f"L: {show_bar_l}\n"
        f"R: {BAR_PLACE}{peak_r}\r"
        f"R: {show_bar_r}\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Python program for capturing system audio and show it in VU")
    ap.add_argument("-m", type=str, default="decibel", help="Measuring mode. Available mode 'linear' and 'decibel'")

    args = ap.parse_args()
    measure_mode = args.m

    audio_l = 0.0
    audio_r = 0.0

    max_l = 0.0
    max_r = 0.0

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",

        "-f", "pulse",
        "-sample_rate", "44100",
        "-channels", "2",
        "-fragment_size", "1764",
        "-i", SOURCE,

        "-f", "s16le",
        "-ar", "44100",
        "-ac", "2",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=0
    )

    remainder = b""
    prev_time = time.time()

    lower_val = 10 ** (CUTOFF_DB / 20)

    print(CLEAR_SCREEN)

    try:
        while True:
            now = time.time()

            raw = process.stdout.read(BUFFER_SIZE)

            if not raw:
                break

            data = remainder + raw
            usable_size = len(data) - (len(data) % FRAME_SIZE)
            usable = data[:usable_size]

            remainder = data[usable_size:]

            samples = np.frombuffer(usable, dtype=np.int16)
            samples = samples.reshape(-1, 2)

            left = samples[:, 0] / 32767
            right = samples[:, 1] / 32767

            max_l_now = max(np.abs(left))
            max_r_now = max(np.abs(right))

            if max_l_now > max_l:
                max_l = max_l_now

            if max_r_now > max_r:
                max_r = max_r_now

            if now - prev_time >= 0.02:
                audio_l = ALPHA * audio_l + (1 - ALPHA) * max_l
                audio_r = ALPHA * audio_r + (1 - ALPHA) * max_r

                if audio_l < lower_val:
                    audio_l = lower_val

                if audio_r < lower_val:
                    audio_r = lower_val

                if measure_mode == "linear":
                    term_l = audio_l
                    term_r = audio_r

                if measure_mode == "decibel":
                    term_l = 20 * np.log10(audio_l)
                    term_r = 20 * np.log10(audio_r)
                
                render_terminal(term_l, term_r, measure_mode)

                prev_time = now
                max_l = 0.0
                max_r = 0.0

    except KeyboardInterrupt:
        print("\n\n\nStopping...")

    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
