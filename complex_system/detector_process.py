import cv2
from ultralytics import YOLO
from global_env import *

detector_model = YOLO('./detector/' + DETECTOR_NAME)
detector_angle = YOLO("./detector/" + ANGLE_DETECTOR_NAME)


def calc_center(xyxy):
    x1, y1, x2, y2 = list(map(int, xyxy.tolist()))

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Вычисляем индексы клетки, в которой находится объект
    grid_row_index = int(center_y // CELL_SZ)
    grid_col_index = int(center_x // CELL_SZ)

    return grid_row_index, grid_col_index


def calculate_angle_with_x_axis(image):
    detections = detector_angle(image, verbose=False)
    black_center = []
    white_center = []
    for result in detections:
        boxes = result.boxes
        class_ids = boxes.cls.tolist()  # Преобразуем тензор в список

        for i in range(len(boxes)):
            x1, y1, x2, y2 = list(map(int, boxes.xyxy[i].tolist()))  # Преобразуем тензор координат в список

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if class_ids[i] == 0:
                black_center = (center_x, center_y)
            elif class_ids[i] == 1:
                white_center = (center_x, center_y)
    if len(black_center) == 0 or len(white_center) == 0:
        return None

    vector = np.array([white_center[0] - black_center[0], white_center[1] - black_center[1]])
    x_axis = np.array([1, 0])  # Ось X
    norm_vector = np.linalg.norm(vector)
    if norm_vector == 0:
        return 0  # Защита от деления на ноль
    cosine_angle = np.dot(vector, x_axis) / norm_vector
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    if black_center[1] < white_center[1]:
        angle_deg = 360 - angle_deg

    angle_deg -= 180

    return angle_deg


def detect(value, angle_dict):
    angle_dict[0] = 0
    angle_dict[1] = 0
    angle_dict[2] = 0
    angle_dict[3] = 0
    cap = cv2.VideoCapture(CAPTURE)
    desired_width = 2560 // 2
    desired_height = 1440 // 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    counter = 0
    index_img = 0

    while True:
        counter += 1
        ret, frame = cap.read()
        frame = frame[:-30, 20:-25]
        frame = cv2.resize(frame, (WIN_W, WIN_H))
        if ret:
            dst = cv2.undistort(frame, camera_matrix, dist_k, None)
            detections = detector_model(dst, verbose=False, conf=0.7)

            result = detections[0]
            boxes = result.boxes
            class_ids = boxes.cls.tolist()  # Преобразуем тензор в список

            for index, detection, confidence in zip(class_ids, result.boxes.xyxy, result.boxes.conf):

                x1, y1, x2, y2 = map(int, detection)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                if confidence > 0.65:
                    # Вырезаем объект из кадра
                    object_frame = dst[y:y + h, x:x + w]
                    # Передаём вырезанный фрейм в другую функцию
                    angle = calculate_angle_with_x_axis(object_frame)
                    angle_dict[index] = angle if angle is not None else angle_dict[index]

            for index, xyxy in zip(class_ids, boxes.xyxy):
                value[index] = calc_center(xyxy)

            cv2.imshow("Detection", result.plot())

            if counter % 10 == 0 and DETECTOR_SAVING_MODE:
                cv2.imwrite('./screens/screen' + str(index_img) + '.jpg', dst)
            index_img += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    d = {}
    d2 = {}
    detect(d, d2)
