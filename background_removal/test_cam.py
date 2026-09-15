import cv2

# Buka webcam
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Gagal membuka webcam")
    exit()

while True:
    # Baca frame
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame")
        break

    # Tampilkan frame
    cv2.imshow("Webcam", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam
cap.release()
cv2.destroyAllWindows()