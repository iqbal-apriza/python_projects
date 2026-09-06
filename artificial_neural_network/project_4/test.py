import time

val1 = 0
val2 = 0
val3 = 0

while True:
    print(f"\033[2K\rVal 1: {val1}")
    print(f"\033[2K\rVal 2: {val2}")
    print(f"\033[2K\rVal 3: {val3}")

    val1 += 1
    val2 += 2
    val3 += 3

    time.sleep(0.1)

    print("\033[3A", end="")