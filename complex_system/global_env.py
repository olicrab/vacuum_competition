from enum import Enum

import numpy as np

CAPTURE = 0

# names: {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
ROBOTS = [{"Name": "Синий",
           "url": "http://192.168.103.237/",
           "player_colors": (0, 0, 255),
           "trace_colors": (0, 0, 102)},
          {"Name": "Зеленый",
           "url": "http://192.168.103.112/",
           "player_colors": (0, 255, 0),
           "trace_colors": (0, 102, 0)},
          {"Name": "Оранжевый",
           "url": "http://192.168.103.151/",
           "player_colors": (255, 140, 0),
           "trace_colors": (102, 70, 0)},
          {"Name": "Красный",
           "url": "http://192.168.103.137/",
           "player_colors": (255, 0, 0),
           "trace_colors": (102, 0, 0)}
          ]

camera_matrix = np.array([[756.84415835, 0.00000000e+00, 635.43097983],
                          [0.00000000e+00, 729.45698572, 369.45182133],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_k = np.array([-0.31113402, 0.16787833, -0.00759438, 0.0114132, -0.05750783])

WIN_W = 1280
WIN_H = 720
CELL_SZ = 80
TABLE_SHAPE = (WIN_W // CELL_SZ, WIN_H // CELL_SZ)
ITERATIONS_THRESHOLD = 250

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

VISION_RADIUS = 1

GAMES_THRESHOLD = 8000
MIN_EXPLORE_PROBABILITY = 10
MAX_EXPLORE_PROBABILITY = 100

N_PLAYERS = 4

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# если меньше этого числа, то он едет на зарядку
MIN_POWER_THRESHOLD = 0.30
# если зарядка больше этого числа, то выезжает с зарядки
MAX_POWER_THRESHOLD = 0.95

DETECTOR_SAVING_MODE = False

DETECTOR_NAME = "new_new_detector_4.pt"
ANGLE_DETECTOR_NAME = "new_angle_detector.pt"
AGENT_BATTERY_STEP = 0.05

GO_TO_CHARGE_PROBABILITY = 0.0001

BACKUP_PATH = "backup.txt"
