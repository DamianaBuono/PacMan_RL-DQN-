import math
import random

import pygame
import numpy as np
from collections import deque
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
        # Nota: lo stato ha 19 dimensioni (include danger_flag e muri)
        self.agent = DQNAgent(state_size=19, action_size=4)
        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()
        self.loss = 0
        self.display = Display(self.screen)
        self.game_over = False
        self.reset_pos = False
        self.game_level = 1
        self.total_reward = 0
        # Storico delle ultime azioni e posizioni per rilevare loop
        self.action_history = deque(maxlen=4)
        self.pos_history = deque(maxlen=8)
        self._generate_world()
        self.total_penalty = 0
        self.total_positive = 0
        self.episode_steps = 0
        self.max_episode_steps = 1500


    def _generate_world(self):
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == "1":
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                elif char == " ":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
                elif char in "spor":
                    colors = {"s": "skyblue", "p": "pink", "o": "orange", "r": "red"}
                    self.ghosts.add(Ghost(x_index, y_index, colors[char]))
                elif char == "P":
                    self.player.add(Pac(x_index, y_index))
        self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]

    def generate_new_level(self):
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == " ":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))

    def restart_level(self):
        self.berries.empty()
        for ghost in self.ghosts.sprites():
            ghost.move_to_start_pos()

        self.game_level = 1
        self.player.sprite.pac_score = 0
        self.player.sprite.n_bacche = 0
        self.player.sprite.life = 3
        self.player.sprite.move_to_start_pos()
        self.player.sprite.direction = (0, 0)
        self.player.sprite.status = "idle"
        self.loss = 0
        self.player.sprite.milestones_rewarded.clear()
        self.agent.reset_episode()
        self.total_reward = 0
        self.action_history.clear()
        self.episode_steps = 0  # reset contatore passi

        self.generate_new_level()

    def _dashboard(self):
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)
        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)
        self.display.show_Nbacche(self.player.sprite.n_bacche)

    def _check_game_state(self):
        if self.player.sprite.life <= 0:
            self.game_over = True
        if not self.berries and self.player.sprite.life > 0:
            self.game_level += 1
            for ghost in self.ghosts.sprites():
                ghost.move_speed += self.game_level
                ghost.move_to_start_pos()
            self.player.sprite.move_to_start_pos()
            self.player.sprite.direction = (0, 0)
            self.player.sprite.status = "idle"
            self.generate_new_level()

    def detect_action_loop(self):
        if len(self.action_history) < 4:
            return False
        a = list(self.action_history)
        return (a == [0, 1, 0, 1] or
                a == [1, 0, 1, 0] or
                a == [2, 3, 2, 3] or
                a == [3, 2, 3, 2])

    def detect_position_loop(self):
        # Se nelle ultime posizioni meno di 4 sono distinte, probabilmente bloccato
        return len(set(self.pos_history)) < 4 if len(self.pos_history) == self.pos_history.maxlen else False

    def get_current_state(self):
        pac_rect = self.player.sprite.rect
        pac_pos = (pac_rect.x, pac_rect.y)
        norm_x = pac_pos[0] / WIDTH
        norm_y = pac_pos[1] / HEIGHT

        pac_direction = self.player.sprite.direction
        norm_dx = pac_direction[0] / PLAYER_SPEED
        norm_dy = pac_direction[1] / PLAYER_SPEED

        max_distance = math.hypot(WIDTH, HEIGHT)

        # Dati fantasmi
        ghosts = self.ghosts.sprites()
        num_ghosts = len(ghosts)
        if num_ghosts > 0:
            dists = [self.get_distance(pac_rect.center, g.rect.center) for g in ghosts]
            mi = np.argmin(dists)
            norm_ghost_distance = dists[mi] / max_distance
            nearest = ghosts[mi]
            ghost_dx = nearest.direction[0] / PLAYER_SPEED
            ghost_dy = nearest.direction[1] / PLAYER_SPEED
        else:
            norm_ghost_distance, ghost_dx, ghost_dy = 1.0, 0.0, 0.0

        # Dati bacche
        berries = [(b.abs_x, b.abs_y) for b in self.berries.sprites()]
        num_berries = len(berries)
        if berries:
            bd = [self.get_distance(pac_rect.center, pos) for pos in berries]
            mi = np.argmin(bd)
            norm_nearest_berry_distance = bd[mi] / max_distance
            berry_dx = (berries[mi][0] - pac_pos[0]) / WIDTH
            berry_dy = (berries[mi][1] - pac_pos[1]) / HEIGHT
        else:
            norm_nearest_berry_distance, berry_dx, berry_dy = 0.0, 0.0, 0.0

        is_immune = 1 if self.player.sprite.immune else 0
        walls = self.check_walls(pac_pos)
        norm_n_bacche = self.player.sprite.n_bacche / self.agent.target_berries
        danger_flag = 1 if norm_ghost_distance < 0.2 else 0

        state = np.array([
            norm_x, norm_y,
            norm_dx, norm_dy,
            num_ghosts, num_berries,
            norm_ghost_distance,
            ghost_dx, ghost_dy,
            norm_nearest_berry_distance,
            berry_dx, berry_dy,
            is_immune,
            walls["up"], walls["down"], walls["left"], walls["right"],
            norm_n_bacche,
            danger_flag
        ])
        return state

    def apply_action(self, action):
        actions_map = {
            0: (0, -PLAYER_SPEED),  # su
            1: (0, PLAYER_SPEED),   # giù
            2: (-PLAYER_SPEED, 0),  # sinistra
            3: (PLAYER_SPEED, 0)    # destra
        }
        self.player.sprite.direction = actions_map[action]

    def get_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def check_walls(self, pac_pos):
        walls = {"up": 0, "down": 0, "left": 0, "right": 0}
        pac_rect = pygame.Rect(pac_pos[0], pac_pos[1], CHAR_SIZE, CHAR_SIZE)
        offset = PLAYER_SPEED
        dirs = {
            "up": (0, -offset),
            "down": (0, offset),
            "left": (-offset, 0),
            "right": (offset, 0)
        }
        for d, (dx, dy) in dirs.items():
            test_rect = pac_rect.move(dx, dy)
            for wall in self.walls:
                if wall.rect.colliderect(test_rect):
                    walls[d] = 1
                    break
        return walls

    def compute_reward(self):
        reward = 0
        pacman = self.player.sprite
        for berry in self.berries.sprites():
            if pacman.rect.colliderect(berry.rect):
                pacman.n_bacche += 1
                reward += 2.0  # ← normalizzato
                self.total_positive += 2.0
                pacman.pac_score += 10
                milestones = {24: 0.5, 49: 1.0} #, 98: 1.5, 147: 2.0
                for m, bonus in milestones.items():
                    if pacman.n_bacche == m and m not in pacman.milestones_rewarded:
                        reward += bonus
                        self.total_positive += bonus
                        pacman.milestones_rewarded.add(m)
                if berry.power_up:
                    reward += 2.5
                    self.total_positive += 2.5
                    pacman.immune = True
                    pacman.immune_time = 150
                    pacman.pac_score += 50
                berry.kill()


        for ghost in self.ghosts.sprites():
            if pacman.rect.colliderect(ghost.rect):
                if pacman.immune:
                    reward += 3.0  # ← normalizzato
                    self.total_positive += 3.0
                else:
                    reward -= 3.0  # ← normalizzato
                self.total_penalty += 3.0

        return int(reward)

    def compute_action_reward(self, action):
        actions_map = {0: (0, -PLAYER_SPEED), 1: (0, PLAYER_SPEED), 2: (-PLAYER_SPEED, 0), 3: (PLAYER_SPEED, 0)}
        pacman = self.player.sprite
        current_pos = (pacman.rect.x, pacman.rect.y)
        dx, dy = actions_map[action]
        new_pos = (pacman.rect.x + dx, pacman.rect.y + dy)
        action_reward = 0.0
        if new_pos == current_pos:
            action_reward -= 0.02
            self.total_penalty += 0.02
        elif new_pos in self.agent.visited_positions:
            action_reward -= 0.05
            self.total_penalty += 0.05
        else:
            action_reward += 0.05
            self.total_positive += 0.05


        self.agent.visited_positions.append(new_pos)
        return int(action_reward)

    def compute_final_bonus(self):
        pacman = self.player.sprite
        if pacman.n_bacche >= self.agent.target_berries and pacman.life > 0:
            self.total_positive += 5.0  # ← normalizzato
            return 5.0
        if pacman.life <= 0 and pacman.n_bacche < self.agent.target_berries:
            self.total_penalty += 5.0  # ← normalizzato
            return -5.0
        if self.episode_steps >= self.max_episode_steps:
            self.total_penalty += 2.5  # ← normalizzato
            return -2.5

        return 0.0

    def updateRL(self):
        if not self.game_over:
            self.episode_steps += 1
            # forzatura termine episodio se troppi passi
            if self.episode_steps >= self.max_episode_steps:
                print("⚠️ Episodio terminato per numero massimo di passi.")
                self.game_over = True

            self.screen.fill("black")
            state = self.get_current_state()
            action = self.agent.act(state)

            # Verifica loop (azioni o posizioni)
            is_loop = self.detect_action_loop() or self.detect_position_loop()
            if is_loop:
                possible_actions = [0, 1, 2, 3]
                if action in possible_actions:
                    possible_actions.remove(action)
                action = random.choice(possible_actions)

            # Registra azione per rilevamento loop
            self.action_history.append(action)
            self.apply_action(action)
            self.player.sprite.animateRL(action, self.walls_collide_list)

            # wrapping orizzontale
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            # collisioni fantasmi
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if self.player.sprite.immune:
                        ghost.move_to_start_pos()
                        self.player.sprite.pac_score += 100
                    else:
                        self.player.sprite.life -= 1
                        self.reset_pos = True
                        if self.player.sprite.life <= 0:
                            self.game_over = True
                        break

            if self.player.sprite.life <= 0 or self.player.sprite.n_bacche == self.agent.target_berries:
                self.game_over = True

            # Reward
            ar = self.compute_action_reward(action)
            rw = self.compute_reward() + ar

            if is_loop:
                rw -= 1.0
                self.total_penalty += 1.0

            if self.game_over:
                rw += self.compute_final_bonus()

            next_state = self.get_current_state()
            done = self.game_over
            self.agent.remember(state, action, rw, next_state, done)
            self.loss = self.agent.replay()
            self.total_reward += rw
            self._check_game_state()

            # Disegno
            for wall in self.walls.sprites(): wall.update(self.screen)
            for berry in self.berries.sprites(): berry.update(self.screen)
            for ghost in self.ghosts.sprites(): ghost.update(self.walls_collide_list)
            self.ghosts.draw(self.screen)
            self.player.update()
            self.player.draw(self.screen)

            if self.game_over:
                self.display.game_over()

            self._dashboard()

            if self.reset_pos and not self.game_over:
                for ghost in self.ghosts.sprites(): ghost.move_to_start_pos()
                self.player.sprite.move_to_start_pos()
                self.player.sprite.direction = (0, 0)
                self.reset_pos = False

    def update(self):
        if not self.game_over:
            pressed_key = pygame.key.get_pressed()
            self.player.sprite.animate(pressed_key, self.walls_collide_list)
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if not self.player.sprite.immune:
                        self.player.sprite.life -= 1
                        self.reset_pos = True
                        break
                    else:
                        ghost.move_to_start_pos()
                        self.player.sprite.pac_score += 100
            if self.player.sprite.life <= 0:
                self.game_over = True
            elif self.player.sprite.n_bacche >= self.agent.target_berries:
                self.game_over = True
            reward = self.compute_reward()
            if self.game_over:
                reward += self.compute_final_bonus()
            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 100
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    self.player.sprite.n_bacche += 1
                    berry.kill()
        self._check_game_state()
        for wall in self.walls.sprites(): wall.update(self.screen)
        for berry in self.berries.sprites(): berry.update(self.screen)
        for ghost in self.ghosts.sprites(): ghost.update(self.walls_collide_list)
        self.ghosts.draw(self.screen)
        self.player.update()
        self.player.draw(self.screen)
        if self.game_over:
            self.display.game_over()
        self._dashboard()
        if self.reset_pos and not self.game_over:
            for ghost in self.ghosts.sprites(): ghost.move_to_start_pos()
            self.player.sprite.move_to_start_pos()
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0, 0)
            self.reset_pos = False
        if self.game_over:
            if pygame.key.get_pressed()[pygame.K_r]:
                self.game_over = False
                self.restart_level()
