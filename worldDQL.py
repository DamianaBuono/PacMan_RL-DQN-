import math
import pygame
import numpy as np
from settings import HEIGHT, WIDTH, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED
from pac import Pac
from cell import Cell
from berry import Berry
from ghost import Ghost
from display import Display
from reinforcementDQL import DQNAgent  # L'agente ora non calcola il reward

class World:
    def __init__(self, screen):
        self.screen = screen
        # Nota: lo stato ha 19 dimensioni se includi informazioni aggiuntive
        self.agent = DQNAgent(state_size=19, action_size=4)
        self.player = pygame.sprite.GroupSingle()
        # Invece di un gruppo, usiamo un singolo fantasma
        self.ghost = None
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()
        self.loss = 0
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
        # Seleziona solo la prima occorrenza di un fantasma
        ghost_created = False
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == "1":
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                elif char == " ":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
                # Se il carattere corrisponde a un fantasma, creiamo il singolo fantasma
                elif char in "spor" and not ghost_created:
                    self.ghost = Ghost(x_index, y_index, "red")
                    ghost_created = True
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
        if self.ghost:
            self.ghost.move_to_start_pos()

        self.game_level = 1
        self.player.sprite.pac_score = 0
        self.player.sprite.n_bacche = 0
        self.player.sprite.life = 3
        self.player.sprite.move_to_start_pos()
        self.player.sprite.direction = (0, 0)
        self.player.sprite.status = "idle"
        self.loss = 0
        self.combo_counter = 0
        self.last_position = None
        self.player.sprite.milestones_rewarded.clear()
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
        if self.player.sprite.life <= 0:
            self.game_over = True
        if not self.berries and self.player.sprite.life > 0:
            self.game_level += 1
            if self.ghost:
                self.ghost.move_speed += self.game_level
                self.ghost.move_to_start_pos()
            self.player.sprite.move_to_start_pos()
            self.player.sprite.direction = (0, 0)
            self.player.sprite.status = "idle"
            self.generate_new_level()

    def get_current_state(self):
        pac_rect = self.player.sprite.rect
        pac_pos = (pac_rect.x, pac_rect.y)
        norm_x = pac_pos[0] / WIDTH
        norm_y = pac_pos[1] / HEIGHT

        pac_direction = self.player.sprite.direction
        norm_dx = pac_direction[0] / PLAYER_SPEED
        norm_dy = pac_direction[1] / PLAYER_SPEED

        max_distance = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        # Gestione del singolo fantasma
        if self.ghost:
            ghost_distance = self.get_distance(pac_rect.center, self.ghost.rect.center)
            norm_ghost_distance = ghost_distance / max_distance
            ghost_dx_norm = self.ghost.direction[0] / PLAYER_SPEED
            ghost_dy_norm = self.ghost.direction[1] / PLAYER_SPEED
            num_ghosts = 1
        else:
            norm_ghost_distance = 1.0
            ghost_dx_norm = 0.0
            ghost_dy_norm = 0.0
            num_ghosts = 0

        # Calcola le informazioni della bacca più vicina
        berries = [(berry.abs_x, berry.abs_y) for berry in self.berries.sprites()]
        if berries:
            berry_distances = [self.get_distance(pac_rect.center, pos) for pos in berries]
            min_distance = min(berry_distances)
            norm_nearest_berry_distance = min_distance / max_distance
            min_index = np.argmin(berry_distances)
            nearest_berry_pos = berries[min_index]
            berry_dx = (nearest_berry_pos[0] - pac_pos[0]) / WIDTH
            berry_dy = (nearest_berry_pos[1] - pac_pos[1]) / HEIGHT
        else:
            norm_nearest_berry_distance = 0.0
            berry_dx = 0.0
            berry_dy = 0.0

        is_immune = 1 if self.player.sprite.immune else 0
        walls = self.check_walls((pac_rect.x, pac_rect.y))
        norm_n_bacche = self.player.sprite.n_bacche / self.agent.target_berries

        # Flag di pericolo se il fantasma è troppo vicino
        danger_threshold = 0.2
        danger_flag = 1 if norm_ghost_distance < danger_threshold else 0

        state = np.array([
            norm_x,
            norm_y,
            norm_dx,
            norm_dy,
            num_ghosts,
            len(self.berries.sprites()),
            norm_ghost_distance,
            ghost_dx_norm,
            ghost_dy_norm,
            norm_nearest_berry_distance,
            berry_dx,
            berry_dy,
            is_immune,
            walls["up"],
            walls["down"],
            walls["left"],
            walls["right"],
            norm_n_bacche,
            danger_flag
        ])

        return state

    def apply_action(self, action):
        actions_map = {
            0: (0, -PLAYER_SPEED),
            1: (0, PLAYER_SPEED),
            2: (-PLAYER_SPEED, 0),
            3: (PLAYER_SPEED, 0)
        }
        self.player.sprite.direction = actions_map[action]

    def get_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def check_walls(self, pac_pos):
        walls = {"up": 0, "down": 0, "left": 0, "right": 0}
        tile_size = CHAR_SIZE
        for wall in self.walls.sprites():
            if (pac_pos[0], pac_pos[1] - tile_size) == (wall.rect.x, wall.rect.y):
                walls["up"] = 1
            if (pac_pos[0], pac_pos[1] + tile_size) == (wall.rect.x, wall.rect.y):
                walls["down"] = 1
            if (pac_pos[0] - tile_size, pac_pos[1]) == (wall.rect.x, wall.rect.y):
                walls["left"] = 1
            if (pac_pos[0] + tile_size, pac_pos[1]) == (wall.rect.x, wall.rect.y):
                walls["right"] = 1
        return walls

    # Funzione per il reward ambientale
    def compute_reward(self):
        reward = 0
        pacman = self.player.sprite

        for berry in self.berries.sprites():
            if pacman.rect.colliderect(berry.rect):
                # Bonus progressivi: assegna solo se non già dato
                milestones = {24: 1, 49: 2, 98: 3, 147: 4}
                if pacman.life > 0:
                    for m, bonus in milestones.items():
                        if pacman.n_bacche == m and m not in pacman.milestones_rewarded:
                            reward += bonus
                            print("BONUS ", reward)
                            pacman.milestones_rewarded.add(m)
                if berry.power_up:
                    reward += 2
                    pacman.immune = True
                    pacman.immune_time = 150
                    pacman.pac_score += 50
                else:
                    reward += 1
                    pacman.pac_score += 10
                pacman.n_bacche += 1
                berry.kill()

        # Gestione della collisione con il fantasma
        ghost = self.ghost
        if pacman.rect.colliderect(ghost.rect):
            if pacman.immune:
                reward += 5
                ghost.move_to_start_pos()
            else:
                reward -= 10

        self.total_reward += reward  # Accumula il reward nell'ambiente
        return reward

    # Bonus finale al termine dell'episodio
    def compute_final_bonus(self):
        pacman = self.player.sprite
        if pacman.n_bacche >= self.agent.target_berries and pacman.life > 0:
            return 50
        elif pacman.life <= 0 and pacman.n_bacche < self.agent.target_berries:
            return -10
        else:
            return 0

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

            # Gestione collisione con il fantasma
            if self.ghost and self.player.sprite.rect.colliderect(self.ghost.rect):
                if self.player.sprite.immune:
                    self.ghost.move_to_start_pos()
                    self.player.sprite.pac_score += 100
                else:
                    self.player.sprite.life -= 1
                    self.combo_counter = 0
                    self.last_position = None
                    if self.player.sprite.life <= 0:
                        self.game_over = True
                    else:
                        self.reset_pos = True

            if self.player.sprite.life <= 0 or self.player.sprite.n_bacche >= self.agent.target_berries:
                self.game_over = True

            # Calcolo del reward e aggiornamento dello stato
            reward = self.compute_reward()
            if self.game_over:
                reward += self.compute_final_bonus()

            next_state = self.get_current_state()
            done = self.game_over

            self.agent.remember(current_state, action, reward, next_state, done)
            self.loss = self.agent.replay()
            # Il reward cumulativo viene gestito via self.total_reward

            self._check_game_state()

            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            if self.ghost:
                self.ghost.update(self.walls_collide_list)
                self.ghost.draw(self.screen)

            self.player.update()
            self.player.draw(self.screen)

            if self.game_over:
                self.display.game_over()
            self._dashboard()

            if self.reset_pos and not self.game_over:
                if self.ghost:
                    self.ghost.move_to_start_pos()
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

            # Controllo collisione con il fantasma
            if self.ghost and self.player.sprite.rect.colliderect(self.ghost.rect):
                if not self.player.sprite.immune:
                    self.player.sprite.life -= 1
                    self.reset_pos = True
                else:
                    self.ghost.move_to_start_pos()
                    self.player.sprite.pac_score += 100

            if self.player.sprite.life <= 0:
                self.game_over = True
            elif self.player.sprite.n_bacche >= self.agent.target_berries:
                self.game_over = True

            # Calcolo del reward
            reward = self.compute_reward()
            if self.game_over:
                reward += self.compute_final_bonus()

            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    self.player.sprite.n_bacche += 1
                    berry.kill()

        self._check_game_state()
        [wall.update(self.screen) for wall in self.walls.sprites()]
        [berry.update(self.screen) for berry in self.berries.sprites()]
        if self.ghost:
            self.ghost.update(self.walls_collide_list)
            self.ghost.draw(self.screen)
        self.player.update()
        self.player.draw(self.screen)
        if self.game_over:
            self.display.game_over()

        self._dashboard()

        if self.reset_pos and not self.game_over:
            if self.ghost:
                self.ghost.move_to_start_pos()
            self.player.sprite.move_to_start_pos()
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0, 0)
            self.reset_pos = False

        if self.game_over:
            pressed_key = pygame.key.get_pressed()
            if pressed_key[pygame.K_r]:
                self.game_over = False
                self.restart_level()
