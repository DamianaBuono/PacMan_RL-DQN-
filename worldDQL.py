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
        # Aggiorniamo lo state_size se hai modificato lo stato (qui manteniamo 8 se usi le modifiche precedenti)
        self.agent = DQNAgent(state_size=8, action_size=4)

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
        self.last_position = None
        self.combo_counter = 0

        self._generate_world()

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
        time.sleep(2)

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

        self.combo_counter = 0
        self.last_position = None

        self.agent.reset_episode()
        self.total_reward = 0

        self.generate_new_level()

    def _dashboard(self):
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)

        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)
        self.display.show_Nbacche(self.player.sprite.n_bacche)

    def _check_game_state(self):
        # Modifica: usa <= 0 invece di == 0 per il game over
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

    def get_current_state(self):
        pac_pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)
        pac_direction = self.player.sprite.direction
        ghosts = [(ghost.rect.x, ghost.rect.y) for ghost in self.ghosts.sprites()]
        berries = [(berry.abs_x, berry.abs_y) for berry in self.berries.sprites()]

        if ghosts:
            ghost_distances = [self.get_distance(self.player.sprite.rect.center, (g[0], g[1])) for g in ghosts]
            nearest_ghost_distance = min(ghost_distances)
        else:
            nearest_ghost_distance = 999

        if berries:
            berry_distances = [self.get_distance(self.player.sprite.rect.center, (b[0], b[1])) for b in berries]
            nearest_berry_distance = min(berry_distances)
        else:
            nearest_berry_distance = 999

        return np.array([pac_pos[0],
                         pac_pos[1],
                         pac_direction[0],
                         pac_direction[1],
                         len(ghosts),
                         len(berries),
                         nearest_ghost_distance,
                         nearest_berry_distance])

    def apply_action(self, action):
        actions_map = {
            0: (0, -PLAYER_SPEED),
            1: (0, PLAYER_SPEED),
            2: (-PLAYER_SPEED, 0),
            3: (PLAYER_SPEED, 0)
        }
        self.player.sprite.direction = actions_map[action]

    def get_distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def updateRL(self):
        if not self.game_over:
            self.screen.fill("black")

            current_state = self.get_current_state()
            action = self.agent.act(current_state)

            self.apply_action(action)
            self.player.sprite.animateRL(action, self.walls_collide_list)

            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            # Controllo delle condizioni terminali
            if self.player.sprite.life <= 0:
                self.game_over = True
                #print("Game Over: Pac-Man ha esaurito le vite!")
            elif self.player.sprite.n_bacche >= self.agent.target_berries:
                self.game_over = True
               # print(f"Game Over: Pac-Man ha raccolto {self.agent.target_berries} bacche!")

            # Calcolo la ricompensa per questo step
            reward = self.agent.calculate_reward(self)
            # Se l'episodio è finito, aggiungo la ricompensa finale
            if self.game_over:
                final_bonus = self.agent.finalize_episode(self.player.sprite)
                reward += final_bonus

            next_state = self.get_current_state()
            done = self.game_over

            self.agent.remember(current_state, action, reward, next_state, done)
            self.agent.replay()

            self.total_reward += reward

            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if self.player.sprite.immune:
                        ghost.move_to_start_pos()
                        self.player.sprite.pac_score += 100
                    else:
                        self.player.sprite.life -= 1
                        self.combo_counter = 0
                        self.last_position = None

                        if self.player.sprite.life <= 0:
                            self.game_over = True
                        else:
                            self.reset_pos = True
                            time.sleep(2)
                        break

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

    def update(self):
        if not self.game_over:
            pressed_key = pygame.key.get_pressed()
            self.player.sprite.animate(pressed_key, self.walls_collide_list)

            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    self.player.sprite.n_bacche += 1
                    berry.kill()

            # Debug: stampa il valore delle vite prima del controllo
            if self.player.sprite.life <= 0:
                self.game_over = True
                #print("Game Over: Pac-Man ha esaurito le vite!")
            elif self.player.sprite.n_bacche >= self.agent.target_berries:
                self.game_over = True
                #print(f"Game Over: Pac-Man ha raccolto {self.agent.target_berries} bacche!")

            reward = self.agent.calculate_reward(self)

            self.total_reward += reward

            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if not self.player.sprite.immune:
                        time.sleep(2)
                        self.player.sprite.life -= 1
                        #print(f"Vita rimanente dopo collisione (update): {self.player.sprite.life}")  # Debug
                        self.reset_pos = True
                        break
                    else:
                        ghost.move_to_start_pos()
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
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0, 0)
            self.reset_pos = False

        if self.game_over:
            pressed_key = pygame.key.get_pressed()
            if pressed_key[pygame.K_r]:
                self.game_over = False
                self.restart_level()
