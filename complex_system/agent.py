import os
import random
import time
from collections import deque
from enum import Enum

import numpy as np
import requests
import torch
from miio import RoborockVacuum

from complex_system.global_env import *
from complex_system.model import Linear_QNet, QTrainer


class RobotGameState(Enum):
    Work = 0
    Charging = 1
    Dead = 2


class RemoteControl:
    def __init__(self, robot_url: str, id):
        self.robot_url = robot_url
        self.manual_ctl = '/api/v2/robot/capabilities/ManualControlCapability'
        self.home_url = "/api/v2/robot/capabilities/BasicControlCapability"
        self.name = ROBOTS[id]['Name']

    def enter(self):
        r = requests.put(self.robot_url + self.manual_ctl, json={'action': 'enable'})
        print(f"{self.name}: enter", r)

    def exit(self):
        r = requests.put(self.robot_url + self.manual_ctl, json={'action': 'disable'})
        print(f"{self.name}: exit", r)

    def home(self):
        r = requests.put(self.robot_url + self.home_url, json={'action': 'home'})
        print(f"{self.name}: home", r)

    def move(self, direction: str):

        try:
            r = requests.put(self.robot_url + self.manual_ctl, json={'action': 'move', 'movementCommand': direction},
                             timeout=5)
        except:
            print(f"{self.name}: move")
            self.enter()
            self.move(direction)

    def stop(self):
        r = requests.put(self.robot_url + "/api/v2/robot/capabilities/RawCommandCapability",
                         json={'method': 'set_direction', 'args': [5]})
        print(f"{self.name}:stop", r)

    def manual_control(self, action):
        actions = ['forward', 'rotate_counterclockwise', 'rotate_clockwise']
        self.move(actions[np.argmax(action)])
        time.sleep(0.1)

    def manual_stop(self):
        self.exit()


class Agent:
    centers = None
    statuses = None
    angles = None

    next_id = 0

    def __init__(self, url):
        self.robot_game_state = RobotGameState.Work
        self.id = Agent.next_id
        Agent.next_id += 1
        self.robot = RemoteControl(url, self.id)
        self.state = None
        self.n_games = 0
        self.games_threshold = GAMES_THRESHOLD
        self.min_explore_prob = MIN_EXPLORE_PROBABILITY
        self.max_explore_prob = MAX_EXPLORE_PROBABILITY
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.count_errors = 0
        self.count_charging_error = -1

        if os.path.exists('./model/model' + str(self.id) + '.pth'):
            self.model = Linear_QNet.load('./model/model' + str(self.id) + '.pth')
        else:
            self.model = Linear_QNet(35, 1024, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_power(self):
        # получение количества заряда
        return Agent.statuses[self.id]['battery'] / 100

    def get_state_ch(self):

        return Agent.statuses[self.id]['flag']

    def update_state(self, game):
        try:
            curr_power = self.get_power()
            assert curr_power >= 0
        except:
            print(f"Не получили зарядку. Удаляем {ROBOTS[self.id]['Name']}")
            self.robot_game_state = RobotGameState.Dead
            return
        if curr_power > MAX_POWER_THRESHOLD and self.robot_game_state == RobotGameState.Charging:
            self.robot_game_state = RobotGameState.Work
        # elif (not self.force_charging) and curr_power > self.battery_threshold and self.skip:
        #     self.skip = False

        table = game.cells
        i, j = Agent.centers[self.id]
        rows, cols = table.shape

        norm_i, norm_j = i / cols, j / rows

        angle = Agent.angles[self.id]

        is_wall_forward = (i == 0)
        is_wall_backward = (i == rows - 1)
        is_wall_right = (j == cols - 1)
        is_wall_left = (j == 0)

        # радиус клеток вокруг робота, которые входя в обзор
        radius = VISION_RADIUS

        table_slice = _extract_rhombus_center(table, center_row=i, center_col=j, radius=radius)

        mask1 = (table_slice != -1)
        mask2 = (table_slice != self.id)
        mask3 = (table_slice != -2)
        table_is_enemy = table_slice * mask1 * mask2 * mask3
        table_is_enemy_f = np.array([table_is_enemy]).flatten()

        mask3 = (table_slice == -1)
        table_is_empty = table_slice * mask3
        table_is_empty_f = np.array([table_is_empty]).flatten()

        mask4 = (table_slice == self.id)
        table_is_mine = table_slice * mask4
        table_is_mine_f = np.array([table_is_mine]).flatten()

        is_leader = int(np.argmax(game.player_scores) == self.id)

        result = (np.concatenate((np.array([is_wall_forward, is_wall_backward, is_wall_right, is_wall_left]),
                                  table_is_enemy_f, table_is_empty_f, table_is_mine_f,
                                  np.array([norm_i, norm_j]), np.array([((angle / 180) + 1) / 2]), np.array([is_leader]))))

        self.state = result

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action(self):
        def _to_numpy(output):
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy()
            elif isinstance(output, np.ndarray):
                return output
            else:
                raise TypeError(f"Output type not supported: must be np.ndarray or torch.Tensor, got {type(output)}")

        if self.games_threshold != 0:
            explore_probability = self.max_explore_prob + (self.n_games / self.games_threshold) \
                                  * (self.min_explore_prob - self.max_explore_prob)
        else:
            explore_probability = self.min_explore_prob
        self.epsilon = max([self.min_explore_prob, explore_probability])

        prediction = np.array([0, 0, 0])

        if random.randint(0, 100) <= self.epsilon:
            random_numbers = [random.uniform(0, 1) for _ in range(len(prediction))]
            prediction[np.argmax(random_numbers)] = 1
        else:
            state0 = torch.tensor(self.state, dtype=torch.float)
            prediction = self.model(state0)

        prediction = _to_numpy(prediction)
        # Найти индекс максимального элемента
        max_index = np.argmax(prediction)

        # Создать массив из нулей такого же размера
        binary_pred = np.zeros_like(prediction)

        # Установить значение 1 на месте максимального элемента
        binary_pred[max_index] = 1

        flag = False
        try:
            curr_power = self.get_power()
            assert curr_power >= 0
            # проверка на отправку домой
            if curr_power < MIN_POWER_THRESHOLD:
                binary_pred = np.array([0, 0, 0])
                flag = True
                self.force_charging = True
        except Exception as ex:
            print(f"Except 2: {ex}")
            self.robot_game_state = RobotGameState.Dead
            return None

        return binary_pred, flag


def _extract_rhombus_center(array, center_row, center_col, radius):
    # функция возвращает ромбовидный срез с указанным радиусом и центром, если точек нет, то -2
    m, n = array.shape

    rhombus_indices = []

    # Определение индексов элементов в ромбоидальной области
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            row_index = center_row + i
            col_index = center_col + j
            # Проверка на границы массива
            if 0 <= row_index < m and 0 <= col_index < n:
                rhombus_indices.append((row_index, col_index))
            else:
                rhombus_indices.append(None)
    # Извлечение значений по индексам ромбоидальной области
    rhombus_values = [array[idx] if idx is not None else -2 for idx in rhombus_indices]

    # Преобразование в одномерный массив
    rhombus_array = np.array(rhombus_values)

    return rhombus_array
