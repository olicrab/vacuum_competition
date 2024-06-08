import os
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from complex_system.global_env import ROBOTS

matplotlib.use('TkAgg')


class GameStat:
    def __init__(self):
        self._save_folder_name = 'stat'
        self._save_file_name = 'statistics.csv'
        self.df = pd.DataFrame(
            columns=["iterations_count", "players_scores", "players_records", "leader_id", "outsider_id", 'games_time'],
        )

    def save(self):
        statistics_folder_path = self._save_folder_name
        if not os.path.exists(statistics_folder_path):
            os.makedirs(statistics_folder_path)
        file_name = os.path.join('./' + statistics_folder_path, self._save_file_name)
        self.df.to_csv(file_name, index=False)

    def load(self):
        self.df = pd.read_csv('../' + self._save_folder_name + '/' + self._save_file_name)

    def plot(self):
        assert len(self.df) > 0
        X = list(self.df.index)

        scores = self.df['players_scores'].apply(
            lambda x: np.array(list(map(float, x.strip('[]').split())))
        ).to_numpy()

        records = self.df['players_records'].apply(
            lambda x: np.array(list(map(float, x.strip('[]').split())))
        ).to_numpy()

        games_time = self.df['games_time'].to_numpy()
        n_players = len(scores[0])

        fig, axes = plt.subplots(4, figsize=(10, 10))
        for i in range(n_players):
            axes[0].plot(X, [score[i] for score in scores], label=ROBOTS[i]['Name'],
                         color=np.array(ROBOTS[i]['player_colors']) / 255)
            axes[0].set_xlabel('Game')
            axes[0].set_ylabel('Score')
            axes[0].legend()

            axes[1].plot(X, [record[i] for record in records], label=ROBOTS[i]['Name'],
                         color=np.array(ROBOTS[i]['player_colors']) / 255)
            axes[1].set_xlabel('Game')
            axes[1].set_ylabel('Record')
            axes[1].legend()

            r_window_size = 5
            if len(scores) > r_window_size:
                player_score = np.array([score[i] for score in scores])
                rolling_mean = np.convolve(player_score, np.ones(r_window_size) / r_window_size, mode='valid')
                axes[2].plot(X[r_window_size - 1:], rolling_mean, label=ROBOTS[i]['Name'],
                             color=np.array(ROBOTS[i]['player_colors']) / 255)

            axes[2].set_xlabel('Game')
            axes[2].set_ylabel('Mean 5 score')
            axes[2].legend()

        axes[3].plot(X, games_time)
        axes[3].set_xlabel('Game')
        axes[3].set_ylabel('Time')

        plt.savefig('123.jpg')
        # plt.show(block=True)
        # time.sleep(10)
        # plt.close()


if __name__ == "__main__":
    stat = GameStat()

    while True:

        stat.load()
        stat.plot()

        img = cv2.imread('123.jpg')

        cv2.imshow('123', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(20)
        os.remove('123.jpg')

    cv2.destroyAllWindows()

