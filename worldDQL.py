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
        self.agent = DQNAgent(state_size=6, action_size=4)  # Stato e azioni dell'agente

        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()

        self.display = Display(self.screen)

        self.game_over = False
        self.reset_pos = False
        self.player_score = 0
        self.game_level = 1
        self.total_reward = 0

        self._generate_world()

    def _generate_world(self):
        """Genera il mondo di gioco basandosi sulla mappa"""
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
        """Rigenera le bacche per un nuovo livello"""
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == " ":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
        time.sleep(2)

    def restart_level(self):
        """Resetta il gioco al livello iniziale"""
        self.berries.empty()
        for ghost in self.ghosts.sprites():
            ghost.move_to_start_pos()

        self.game_level = 1
        self.player.sprite.pac_score = 0
        self.player.sprite.life = 3
        self.player.sprite.move_to_start_pos()
        self.player.sprite.direction = (0, 0)
        self.player.sprite.status = "idle"

        self.generate_new_level()

    def _dashboard(self):
        """Mostra le informazioni di gioco sulla barra di navigazione"""
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)

        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)

    def _check_game_state(self):
        """Verifica lo stato del gioco e gestisce la progressione di livello"""
        if self.player.sprite.life == 0:
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

    def get_current_state(self):
        """Restituisce lo stato attuale del gioco"""
        pac_pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)
        pac_direction = self.player.sprite.direction
        ghosts = [(ghost.rect.x, ghost.rect.y) for ghost in self.ghosts.sprites()]
        berries = [(berry.abs_x, berry.abs_y) for berry in self.berries.sprites()]

        return np.array([pac_pos[0], pac_pos[1], pac_direction[0], pac_direction[1], len(ghosts), len(berries)])

    def apply_action(self, action):
        """Applica l'azione scelta dall'agente"""
        actions_map = {
            0: (0, -PLAYER_SPEED),
            1: (0, PLAYER_SPEED),
            2: (-PLAYER_SPEED, 0),
            3: (PLAYER_SPEED, 0)
        }
        self.player.sprite.direction = actions_map[action]

    def get_reward(self):
        """Calcola la ricompensa basata sugli eventi di gioco"""
        reward = 0

        # Ricompensa per la raccolta delle bacche
        for berry in self.berries.sprites():
            if self.player.sprite.rect.colliderect(berry.rect):
                reward += 50 if berry.power_up else 10
                berry.kill()

        # Ricompensa per l'interazione con i fantasmi
        for ghost in self.ghosts.sprites():
            if self.player.sprite.rect.colliderect(ghost.rect):
                if not self.player.sprite.immune:
                    reward -= 30  # Penale per la collisione con un fantasma
                else:
                    ghost.move_to_start_pos()
                    reward += 100  # Ricompensa per aver mangiato un fantasma mentre è immune

        # Ricompensa per il completamento di livello (quando tutte le bacche sono state mangiate)
        if not self.berries and self.player.sprite.life > 0:
            reward += 50

        # Penalità se il gioco è finito
        if self.game_over:
            reward -= 50

        return reward

    def updateRL(self):
        """Aggiorna il gioco durante la fase di reinforcement learning"""
        if not self.game_over:
            self.screen.fill("black")

            current_state = self.get_current_state()
            action = self.agent.act(current_state)

            self.apply_action(action)
            self.player.sprite.animateRL(action, self.walls_collide_list)

            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    reward = self.get_reward()
                    next_state = self.get_current_state()
                    self.agent.remember(current_state, action, reward, next_state, self.game_over)
                    self.agent.replay()
                    self.total_reward += reward
                    print("-----------total reward:", self.total_reward)
                    berry.kill()

            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    print("---Vite di Pacman----", self.player.sprite.life)
                    if not self.player.sprite.immune:
                        self.player.sprite.life -= 1
                        print("---nuove Vite di Pacman----", self.player.sprite.life)
                        reward = self.get_reward()
                        next_state = self.get_current_state()
                        self.agent.remember(current_state, action, reward, next_state, self.game_over)
                        self.agent.replay()
                        self.total_reward += reward
                        print("-----------total reward:", self.total_reward)

                        if self.player.sprite.life > 0:
                            self.reset_pos = True
                            time.sleep(2)  # Attendi prima di riposizionare Pac-Man
                        else:
                            self.game_over = True
                        break
                    else:
                        ghost.move_to_start_pos()
                        reward = self.get_reward()
                        next_state = self.get_current_state()
                        self.agent.remember(current_state, action, reward, next_state, self.game_over)
                        self.agent.replay()
                        self.total_reward += reward
                        print("-----------total reward:", self.total_reward)
                        self.player.sprite.pac_score += 100

            self._check_game_state()

            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
            self.ghosts.draw(self.screen)

            self.player.update()
            self.player.draw(self.screen)

            if self.game_over:
                self.display.game_over()

            self._dashboard()

            if self.reset_pos and not self.game_over:
                [ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
                self.player.sprite.move_to_start_pos()
                self.player.sprite.direction = (0, 0)
                self.reset_pos = False

    # GIOCO NON AUTONOMO
    def update(self):
        if not self.game_over:
            # player movement
            pressed_key = pygame.key.get_pressed()
            self.player.sprite.animate(pressed_key, self.walls_collide_list)

            # teleporting to the other side of the map
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            # PacMan eating-berry effect
            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150  # Timer based from FPS count
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    berry.kill()

            # PacMan bumping into ghosts
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if not self.player.sprite.immune:
                        time.sleep(2)
                        self.player.sprite.life -= 1
                        self.reset_pos = True
                        break
                    else:
                        ghost.move_to_start_pos()
                        self.player.sprite.pac_score += 100

        self._check_game_state()

        # rendering
        [wall.update(self.screen) for wall in self.walls.sprites()]
        [berry.update(self.screen) for berry in self.berries.sprites()]
        [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
        self.ghosts.draw(self.screen)

        self.player.update()
        self.player.draw(self.screen)
        self.display.game_over() if self.game_over else None

        self._dashboard()

        # reset Pac and Ghosts position after PacMan get captured
        if self.reset_pos and not self.game_over:
            [ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
            self.player.sprite.move_to_start_pos()
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0, 0)
            self.reset_pos = False

        # for restart button
        if self.game_over:
            pressed_key = pygame.key.get_pressed()
            if pressed_key[pygame.K_r]:
                self.game_over = False
                self.restart_level()

