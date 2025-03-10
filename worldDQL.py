import pygame
import time
import numpy as np

from settings import HEIGHT, WIDTH, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED
from pac import Pac
from cell import Cell
from berry import Berry
from ghost import Ghost
from display import Display
from reinforcementDQL import DQNAgent


class World:
    def __init__(self, screen):
        self.screen = screen
        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()
        self.display = Display(self.screen)
        self.game_over = False
        self.reset_pos = False
        self.player_score = 0
        self.game_level = 1
        self.powerups = pygame.sprite.Group()
        # DQN Parameters
        STATE_DIM = 8  # Define in get_state()
        ACTION_DIM = 4  # Up, Down, Left, Right
        self.agent = DQNAgent(STATE_DIM, ACTION_DIM)

        self._generate_world()

    def _dashboard(self):
        """
        Disegna la barra di stato con punteggio, vite e livello.
        """
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)

        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)

    def _generate_world(self):
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == "1":
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                elif char == " ":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
                elif char in ["s", "p", "o", "r"]:
                    self.ghosts.add(Ghost(x_index, y_index, "red"))
                elif char == "P":
                    self.player.add(Pac(x_index, y_index))
        self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]

    def apply_action(self, action_index):
        actions_map = [(0, -PLAYER_SPEED), (0, PLAYER_SPEED), (-PLAYER_SPEED, 0), (PLAYER_SPEED, 0)]
        self.player.sprite.direction = actions_map[action_index]

    def updateRL(self):
        if not self.game_over:
            self.screen.fill("black")

            # Ottieni lo stato
            state = get_state(self.player.sprite, list(self.ghosts.sprites()), list(self.berries.sprites()), list(self.powerups.sprites()))
            action_index = self.agent.act(state)
            self.apply_action(action_index)

            # 🟢 Pac-Man si muove qui!
            self.player.update()

            # 🔹 **AGGIUNGI LA PARTE DI RENDERING QUI**
            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
            self.ghosts.draw(self.screen)
            self.player.update()
            self.player.draw(self.screen)

            # 🟢 Dashboard e aggiornamento schermo
            self.display.game_over() if self.game_over else None
            self._dashboard()
            pygame.display.update()  # AGGIUNTO! Aggiorna il display!

            # Controlla se il gioco è finito
            next_state = get_state(self.player.sprite, self.ghosts.sprites(), self.berries.sprites(), self.powerups.sprites())
            reward = self.compute_reward()
            done = self.player.sprite.life == 0

            # Memorizza e addestra
            self.agent.remember(state, action_index, reward, next_state, done)
            self.agent.replay()
            self.agent.update_epsilon()

    def compute_reward(self):
        reward = 0

        for berry in self.berries.sprites():
            if self.player.sprite.rect.colliderect(berry.rect):
                reward += 10 if not berry.power_up else 50
                berry.kill()

        for ghost in self.ghosts.sprites():
            if self.player.sprite.rect.colliderect(ghost.rect):
                if self.player.sprite.immune:
                    reward += 200  # Incentiva a mangiare i fantasmi
                else:
                    reward -= 300  # Penalizza fortemente se viene colpito

        # Penalità minore per ogni passo
        reward -= 0.1

        print(f"Reward calcolato: {reward}")
        return reward

    def update(self):
        """Aggiorna lo stato del mondo senza RL"""
        if not self.game_over:
            self.screen.fill("black")
            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
            self.ghosts.draw(self.screen)
            self.player.update()
            self.player.draw(self.screen)
            self._dashboard()
            pygame.display.update()


def get_state(pacman, ghosts, food, powerups):
    state = [
        pacman.rect.x, pacman.rect.y
    ]

    # Se ci sono almeno due fantasmi, prendili, altrimenti metti 0
    for i in range(2):
        if i < len(ghosts):
            state.extend([ghosts[i].rect.x, ghosts[i].rect.y])
        else:
            state.extend([0, 0])  # Se mancano fantasmi, riempi con 0

    state.append(len(food))  # Numero di bacche rimaste
    state.append(int(any(powerup.power_up for powerup in powerups)))  # Power-up attivo (1/0)

    return np.array(state, dtype=np.float32)
