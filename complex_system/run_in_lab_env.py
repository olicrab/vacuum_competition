import multiprocessing
import time
import traceback
from multiprocessing import Process
from train_process import train
from detector_process import detect
from get_states_process import get_statuses
from pygame_process import plot_matrix
from global_env import *


def start():
    manager1 = multiprocessing.Manager()
    centres_dict = manager1.dict()

    manager2 = multiprocessing.Manager()
    status_dict = manager2.dict()

    manager3 = multiprocessing.Manager()
    pygame_dict = manager3.dict()

    manager4 = multiprocessing.Manager()
    angle_dict = manager4.dict()

    p1 = Process(target=train, args=(centres_dict, status_dict, pygame_dict, angle_dict))
    p2 = Process(target=detect, args=(centres_dict, angle_dict))
    p3 = Process(target=get_statuses, args=(status_dict,))
    p4 = Process(target=plot_matrix, args=(pygame_dict,))

    p2.start()
    time.sleep(20)
    p3.start()
    time.sleep(5)
    p1.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()


if __name__ == "__main__":
    print('start')
    count = 0
    while True:
        try:
            start()
        except Exception as e:
            count += 1
            if count > 20:
                break
            print(traceback.format_exc())

# todo:процессы:
#   1) Обучение
#   2) Видеопоток
#   3) Запросы состояний роботов
#   4) Отрисовка окна симуляции
