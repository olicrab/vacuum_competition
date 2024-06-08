import colorsys
import time
from copy import copy

from complex_system.agent import Agent, RobotGameState
from complex_system.global_env import *


def update_pygame_info(n_games, player_scores, cells, centers):
    LabGame.pygame_dict['n_games'] = n_games
    LabGame.pygame_dict['player_scores'] = player_scores
    LabGame.pygame_dict['cells'] = cells
    LabGame.pygame_dict['centres'] = centers


def get_from_backup():
    try:
        f = open(BACKUP_PATH, 'r+')
        curr_game = int(f.readline())
        f.close()
    except:
        print("Error while read backup file")
        curr_game = 0
    return curr_game


class LabGame:
    pygame_dict = {}

    def __init__(self):
        self.w = WIN_W
        self.h = WIN_H
        self.iterations_threshold = ITERATIONS_THRESHOLD
        self.n_games = get_from_backup()
        # задаем размеры сетки исходя из размеров окна и ячейки
        self.cell_size = CELL_SZ
        self.cols = self.w // self.cell_size
        self.rows = self.h // self.cell_size

        # инициализируем параметры игроков
        self.n_players = N_PLAYERS

        # количество сделанных в игре шагов
        self.iterations_count = 0

        # массив с данными о ячейках
        self.cells = None

        # массив меток сколько очков у игрока
        self.player_labels = None

        # очки игроков
        self.player_scores = None

        self.reset()

    def reset(self):
        self.iterations_count = 0
        self.cells = np.full((self.rows, self.cols), fill_value=-1)
        # print(self.cells.shape)
        # print(self.cells)
        # закрашиваем начальные позиции роботов
        for i, pos in sorted(Agent.centers.items(), key=lambda x: x[0]):
            i = int(i)
            self.cells[pos[0], pos[1]] = i

        self.player_labels = np.array(['' for _ in range(self.n_players)])
        self.player_scores = np.ones(shape=(self.n_players,))

    def play_step(self, agents):
        self.iterations_count += 1

        game_over = False
        rewards = np.full_like(self.player_scores, 0)

        raw_actions = []
        for agent in agents:
            if agent.robot_game_state == RobotGameState.Dead:
                action = np.array([0, 0, 0])
                raw_actions.append((action, False))
            else:
                action = agent.get_action()
                raw_actions.append(action)

        actions = [act[0] for act in raw_actions]
        is_go_home = [act[1] for act in raw_actions]

        applied_actions = copy(actions)
        # расчёт наград для каждого игрока
        for i, action in enumerate(actions):
            rewards[i], applied_actions[i] = self._move(agents[i], action, is_go_home[i])

        # обновляем информацию об окне для процесса рендера
        update_pygame_info(self.n_games, self.player_scores, self.cells, Agent.centers)

        # условие завершения игры
        if self._is_cells_filled(eps_k=0.02) or self.iterations_count > self.iterations_threshold:
            game_over = True
            # расчёт наград в конце игры для каждого игрока
            rewards = self._calc_rewards_from_scores()
            return applied_actions, rewards, game_over, self.player_scores

        return applied_actions, rewards, game_over, self.player_scores

    def _calc_rewards_from_scores(self):
        # max_index = np.argmax(self.player_scores)
        # new_arr = self.player_scores.copy()
        # new_arr[max_index] = 10
        # new_arr[new_arr != 10] = -10
        max_score = TABLE_SHAPE[0]*TABLE_SHAPE[1]
        new_scores = []
        for score in self.player_scores:
            new_scores.append(score*10/max_score)
        return new_scores

    def _is_cells_filled(self, eps_k=0.1):
        # eps_k - процент от общего числа ячеек,
        # нужен для определения сколько незначащих незакрашенных ячеек
        count = np.count_nonzero(self.cells == -1)
        res = count < self.cells.size * eps_k
        return res

    def _move(self, agent, action, is_go_home):

        try:
            if agent.robot_game_state == RobotGameState.Charging and agent.get_state_ch() != "charging" and agent.get_state_ch() != 'go_home':
                agent.count_charging_error += 1

                if agent.count_charging_error < 1:
                    agent.robot.home()
                else:
                    if agent.count_charging_error % 20 == 0:
                        agent.count_charging_error = -1

            elif is_go_home and agent.robot_game_state != RobotGameState.Charging:
                agent.robot.home()
                agent.robot_game_state = RobotGameState.Charging

            elif agent.robot_game_state != RobotGameState.Charging and agent.robot_game_state != RobotGameState.Dead:
                try:
                    agent.robot.manual_control(action)
                except Exception as e:
                    print("Command error:", e)
                    agent.count_error += 1
                    assert agent.count_error < 20
        except Exception as ex:
            print(f"Except 3: {ex}")
            agent.robot_game_state = RobotGameState.Dead
            return 0, [0, 0, 0]

        agent.count_error = 0
        time.sleep(0.1)
        center = Agent.centers[agent.id]
        assert center is not None

        new_cell = self.cells[center[0]][center[1]]
        if new_cell == agent.id:
            reward = -1
            dscore = 0
        elif new_cell == -1:
            reward = 1
            dscore = 1
        else:
            reward = 1
            dscore = 1

        self.cells[center[0]][center[1]] = agent.id
        self.player_scores[agent.id] += dscore

        return reward, action
