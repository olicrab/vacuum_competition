import cv2

cap = cv2.VideoCapture(0)
counter = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        counter += 1
        cv2.imwrite('screen'+str(counter)+".png", frame)
    elif key == ord('q'):
        break



