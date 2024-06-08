import threading
import time
import traceback
from copy import copy

import numpy as np

from complex_system.agent import Agent, RobotGameState
from complex_system.global_env import N_PLAYERS, ROBOTS, BACKUP_PATH
from complex_system.lab_game import LabGame
from complex_system.utils.statistics import GameStat


def connection(agents):
    connection_threads = []

    for agent in agents:
        connection_threads.append(threading.Thread(target=agent.robot.enter()))

    for thread in connection_threads:
        thread.start()

    for thread in connection_threads:
        thread.join()


def train(centres_dict, status_dict, pygame_dict, angle_dict):
    # здесь присваиваем ссылку на общую переменную с процессом детекции
    Agent.centers = centres_dict
    Agent.statuses = status_dict
    Agent.angles = angle_dict
    LabGame.pygame_dict = pygame_dict

    print("1")
    game = LabGame()
    statistics = GameStat()
    print("2")
    agents = [Agent(robot['url']) for robot in ROBOTS]
    print("3")
    records = np.zeros((N_PLAYERS,))

    print("start connect")
    connection(agents)
    print("end connect")
    time.sleep(5)

    count_game = 0
    tic = time.time()
    try:
        while True:
            count_game += 1

            for agent in agents:
                if agent.robot_game_state == RobotGameState.Dead:
                    continue
                agent.update_state(game)

            states_old = []
            for agent in agents:
                if agent.robot_game_state == RobotGameState.Dead:
                    states_old.append(None)
                else:
                    states_old.append(agent.state)

            applied_actions, rewards, done, scores = game.play_step(agents)

            for agent in agents:
                if agent.robot_game_state == RobotGameState.Dead:
                    continue
                agent.update_state(game)

            states_new = [agent.state for agent in agents]

            for i, agent in enumerate(agents):
                if agent.robot_game_state in [RobotGameState.Charging, RobotGameState.Dead]:
                    continue
                # train short memory
                agent.train_short_memory(
                    states_old[i],
                    applied_actions[i],
                    rewards[i],
                    states_new[i],
                    done
                )
                # remember
                agent.remember(
                    states_old[i],
                    applied_actions[i],
                    rewards[i],
                    states_new[i],
                    done
                )

            if done:
                game.n_games += 1
                try:
                    f = open(BACKUP_PATH, 'w+')
                    f.write(str(game.n_games))
                    f.close()
                except:
                    print("Error while read backup file")

                for i, agent in enumerate(agents):
                    if agent.robot_game_state in [RobotGameState.Charging, RobotGameState.Dead]:
                        continue
                    agent.train_long_memory()

                    if scores[i] > records[i]:
                        records[i] = scores[i]
                    agent.model.save('model' + str(agent.id) + '.pth')

                toc = time.time()
                statistics.df.loc[len(statistics.df)] = [
                    game.iterations_count, scores, copy(records), np.argmax(scores), np.argmin(scores), toc - tic
                ]
                tic = time.time()
                statistics.save()
                game.reset()
    except Exception as ex:
        print(f"Except 4: {traceback.format_exc()}")
        for agent in agents:
            agent.robot.manual_stop()




