import pygame
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
        self.agent = DQNAgent(state_size=16, action_size=4)
        self.player = pygame.sprite.GroupSingle()
        # Invece di un gruppo, usiamo un singolo fantasma
        self.ghost = None
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

    def check_walls(self, pac_pos):
        walls = {"up": 0, "down": 0, "left": 0, "right": 0}
        tile_size = CHAR_SIZE  # dimensione di una cella
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

    def get_current_state(self):
        pac_pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)
        pac_direction = self.player.sprite.direction

        # Dati relativi al singolo fantasma
        if self.ghost:
            distance = self.get_distance(self.player.sprite.rect.center, self.ghost.rect.center)
            ghost_data = [distance, self.ghost.direction[0], self.ghost.direction[1]]
        else:
            ghost_data = [999, 0, 0]

        # Informazioni sulle bacche: distanza della bacchetta più vicina
        berries = [(berry.abs_x, berry.abs_y) for berry in self.berries.sprites()]
        if berries:
            berry_distances = [self.get_distance(self.player.sprite.rect.center, (b[0], b[1])) for b in berries]
            nearest_berry_distance = min(berry_distances)
        else:
            nearest_berry_distance = 999

        is_immune = 1 if self.player.sprite.immune else 0
        walls = self.check_walls(pac_pos)
        num_berries = len(self.berries.sprites())

        # Stato esteso (dimensione = 25):
        # [pac_x, pac_y, pac_dx, pac_dy, num_ghosts, num_berries,
        #  ghost_distance, ghost_dx, ghost_dy,
        #  nearest_berry_distance, is_immune,
        #  walls: up, down, left, right, n_bacche]
        state = np.array([
            pac_pos[0],
            pac_pos[1],
            pac_direction[0],
            pac_direction[1],
            1,                # essendoci un solo fantasma, il numero è 1 (oppure 0 se non esiste)
            num_berries,
        ] + ghost_data + [
            nearest_berry_distance,
            is_immune,
            walls["up"],
            walls["down"],
            walls["left"],
            walls["right"],
            self.player.sprite.n_bacche
        ])
        return state

    def apply_action(self, action):
        actions_map = {
            0: (0, -PLAYER_SPEED),   # Su
            1: (0, PLAYER_SPEED),    # Giù
            2: (-PLAYER_SPEED, 0),   # Sinistra
            3: (PLAYER_SPEED, 0)     # Destra
        }
        self.player.sprite.direction = actions_map[action]

    def get_distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def updateRL(self):
        if not self.game_over:
            self.screen.fill("black")
            current_state = self.get_current_state()
            action = self.agent.act(current_state)

            # Logica euristica per fuggire dal singolo fantasma (se non immune)
            if not self.player.sprite.immune and self.ghost:
                pac_center = np.array(self.player.sprite.rect.center)
                ghost_center = np.array(self.ghost.rect.center)
                distance = self.get_distance(pac_center, ghost_center)
                if distance < 200:
                    diff = pac_center - ghost_center
                    if abs(diff[0]) > abs(diff[1]):
                        escape_action = 3 if diff[0] > 0 else 2
                    else:
                        escape_action = 1 if diff[1] > 0 else 0
                    action = escape_action

            self.apply_action(action)
            self.player.sprite.animateRL(action, self.walls_collide_list)

            # Gestione del wrapping orizzontale
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            # Controllo collisione con il singolo fantasma
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

            # Calcolo del reward per questo step
            reward = self.agent.calculate_reward(self)
            next_state = self.get_current_state()
            done = self.game_over

            if self.game_over:
                final_bonus = self.agent.finalize_episode(self.player.sprite)
                reward += final_bonus

            self.agent.remember(current_state, action, reward, next_state, done)
            self.agent.replay()

            self.total_reward += reward

            self._check_game_state()

            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            if self.ghost:
                self.ghost.update(self.walls_collide_list)
            if self.ghost:
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

            # Controllo collisione con il singolo fantasma
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

            reward = self.agent.calculate_reward(self)
            if self.game_over:
                final_bonus = self.agent.finalize_episode(self.player.sprite)
                reward += final_bonus

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
        if self.ghost:
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
