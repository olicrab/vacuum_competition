import time

import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from global_env import *


# Функция для обновления матрицы и графика
def plot_matrix(global_args):
    # Создание пользовательской цветовой карты
    cmap = ListedColormap(['black', 'blue', 'green', 'orange', 'red'])
    plt.ion()  # Включить интерактивный режим
    fig, ax = plt.subplots(figsize=(16, 9))
    matrix = np.full(shape=(9, 16), fill_value=-1)  # Начальная матрица
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=3)
    plt.colorbar(im, ticks=[-1, 0, 1, 2, 3])

    # Установить квадратные клетки


    # Создание текстовых объектов для отображения счета и текущей игры
    title = ax.set_title('', fontsize=16, color='black')
    while True:
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(8.5, -0.5)
        if len(global_args.items()) == 0:
            time.sleep(0.05)
            continue

        im.set_data(global_args['cells'])  # Обновление данных изображения
        scores_text = ' | '.join([f"{ROBOTS[i]['Name']}: {int(global_args['player_scores'][i])}" for i in
                                  range(N_PLAYERS)])
        title.set_text(f"Game {global_args['n_games']} - Scores: {scores_text}")
        ax.draw_artist(ax.patch)
        ax.draw_artist(im)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

        time.sleep(1)  # Задержка на 1 секунду
