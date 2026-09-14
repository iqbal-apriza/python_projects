import os
import fcntl
import ctypes
import numpy as np
import time


DEVICE = "/dev/video2"

# ---------------------------------------------------------------------
# V4L2 constants
# ---------------------------------------------------------------------

VIDIOC_G_FMT = 0xC0D05604
VIDIOC_S_FMT = 0xC0D05605

V4L2_BUF_TYPE_VIDEO_OUTPUT = 2

V4L2_PIX_FMT_BGR32 = 0x34424752   # 'RGB4' in V4L2 naming
# Untuk v4l2loopback, kita akan lihat hasil negosiasi setelah S_FMT.


# ---------------------------------------------------------------------
# V4L2 structures
# ---------------------------------------------------------------------

class V4L2PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class V4L2FormatUnion(ctypes.Union):
    _fields_ = [
        ("pix", V4L2PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class V4L2Format(ctypes.Structure):
    _anonymous_ = ("fmt",)
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("fmt", V4L2FormatUnion),
    ]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fourcc(code):
    return (
        ord(code[0])
        | (ord(code[1]) << 8)
        | (ord(code[2]) << 16)
        | (ord(code[3]) << 24)
    )


def fourcc_string(value):
    return "".join(chr((value >> (8 * i)) & 0xff) for i in range(4))


# ---------------------------------------------------------------------
# Open device
# ---------------------------------------------------------------------

fd = os.open(DEVICE, os.O_RDWR)

print("Device opened:", DEVICE)


# ---------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------

fmt = V4L2Format()
fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT

fmt.pix.width = 640
fmt.pix.height = 480
fmt.pix.pixelformat = fourcc("BGR4")
fmt.pix.field = 1


fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)

print("\nFormat after VIDIOC_S_FMT:")
print("Width      :", fmt.pix.width)
print("Height     :", fmt.pix.height)
print("Pixel      :", fourcc_string(fmt.pix.pixelformat))
print("Bytes/line :", fmt.pix.bytesperline)
print("Size image :", fmt.pix.sizeimage)


# ---------------------------------------------------------------------
# Create test frame
# ---------------------------------------------------------------------

width = fmt.pix.width
height = fmt.pix.height

# BGR image from OpenCV
bgr = np.zeros((height, width, 3), dtype=np.uint8)

# Green
bgr[:, :] = (0, 255, 0)

# Convert BGR -> BGR4
# v4l2 BGR4 means 4 bytes per pixel.
bgra = np.zeros((height, width, 4), dtype=np.uint8)

bgra[:, :, 0] = bgr[:, :, 0]
bgra[:, :, 1] = bgr[:, :, 1]
bgra[:, :, 2] = bgr[:, :, 2]
bgra[:, :, 3] = 255

frame = bgra.tobytes()

print("\nFrame size:", len(frame))
print("Expected :", fmt.pix.sizeimage)


# ---------------------------------------------------------------------
# Send frames
# ---------------------------------------------------------------------

try:
    for i in range(300):

        written = os.write(fd, frame)

        if i % 30 == 0:
            print(f"Frame {i}: {written} bytes")

        time.sleep(1 / 30)

finally:
    os.close(fd)
    print("Device closed")