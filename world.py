import math
import random
import pygame
import numpy as np
from collections import deque
from settings import HEIGHT, WIDTH, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED,MAX_STEPS
from pac import Pac
from cell import Cell
from berry import Berry
from ghost import Ghost
from display import Display
from collections import Counter
from reinforcement import SarsaAgent

class World:
    LOOP_WINDOW = 6  # quante mosse/posizioni tenere in memoria
    LOOP_DIST_THRESHOLD = 2
    def __init__(self, screen, agent: SarsaAgent):
        self.screen = screen
        self.agent = agent

        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()

        self.loss = 0
        self.display = Display(self.screen)
        self.game_over = False
        self.episode_won = False
        self.episode_lost = False
        self.reset_pos = False
        self.max_levels = 4
        self.game_level = 1
        self.total_reward = 0
        # Storico per rilevare loop di azioni e posizioni
        self.action_history = deque(maxlen=self.LOOP_WINDOW)
        self.pos_history = deque(maxlen=self.LOOP_WINDOW)

        self.total_penalty = 0
        self.total_positive = 0
        self.consecutive_loops = 0
        self.episode_steps = 0
        self.map = MAP
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.visited_positions = set()
        self._generate_world()

    def _generate_world(self):
        for y_index, row in enumerate(MAP):
            for x_index, char in enumerate(row):
                if char == "1":
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                elif char in "spor":
                    colors = {"s": "skyblue", "p": "pink", "o": "orange", "r": "red"}
                    self.ghosts.add(Ghost(x_index, y_index, colors[char]))
                elif char == "P":
                    self.player.add(Pac(x_index, y_index))

        self._generate_random_berries()
        self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]

    def _generate_random_berries(self, num_berries=5):
        self.berries.empty()  # Pulisce il gruppo di bacche
        player_pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)  # Usa rect.x e rect.y
        ghost_positions = [(g.rect.x, g.rect.y) for g in self.ghosts]

        empty_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.map[y][x] == '-' and
               (x, y) != player_pos and
               (x, y) not in ghost_positions
        ]
        random.shuffle(empty_positions)

        self.berries = pygame.sprite.Group()

        for i, pos in enumerate(empty_positions[:num_berries]):
            is_power_up = (i % 5 == 0)
            # pos = (colonna, riga) nel grid map: è ciò che Berry si aspetta
            berry = Berry(
                pos[0], pos[1],
                CHAR_SIZE // 4 if not is_power_up else CHAR_SIZE // 2,
                is_power_up
            )
            self.berries.add(berry)


    def restart_level(self):
        self.berries.empty()
        self.ghosts.empty()

        self.game_level = 1
        self._generate_world()
        self.player.sprite.pac_score = 0
        self.player.sprite.n_bacche = 0
        self.player.sprite.life = 3
        self.player.sprite.move_to_start_pos()
        self.player.sprite.direction = (0, 0)
        self.player.sprite.status = "idle"

        self.loss = 0
        self.agent.reset_episode()
        self.visited_positions.clear()
        self.total_reward = 0
        self.action_history.clear()
        self.episode_won = False
        self.episode_lost = False
        self.game_over = False

    def _dashboard(self):
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)
        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)
        self.display.show_Nbacche(self.player.sprite.n_bacche)

    def _check_game_state(self):
        if self.player.sprite.life <= 0:
            self.episode_lost = True
            return

        if not self.berries and self.player.sprite.life > 0:
            # Pacman ha completato il livello corrente
            self.total_reward += 10 * self.game_level
            self.total_positive += 10 * self.game_level
            self.game_level += 1

            if self.game_level > self.max_levels:
                self.episode_won = True
                return

            # Aumenta difficoltà e resetta posizioni
            for ghost in self.ghosts.sprites():
                #ghost.move_speed += self.game_level
                ghost.move_to_start_pos()

            self.player.sprite.move_to_start_pos()
            self.player.sprite.direction = (0, 0)
            self.player.sprite.status = "idle"

            if len(self.ghosts) < 4:
                self.increase_difficulty()

            self._generate_random_berries()

    def increase_difficulty(self):
        max_ghosts = 4
        current_ghosts = len(self.ghosts)

        if current_ghosts < max_ghosts:
            for y_index, row in enumerate(MAP):
                for x_index, char in enumerate(row):
                    if char == '-':
                        pos = (x_index * CHAR_SIZE, y_index * CHAR_SIZE)
                        if pos not in [g.rect.topleft for g in self.ghosts.sprites()]:
                            color_list = ["red", "pink", "orange", "skyblue"]
                            new_color = color_list[current_ghosts]  # Assegna colore diverso
                            self.ghosts.add(Ghost(x_index, y_index, new_color))
                            return

    def detect_action_loop(self):
        """
        True se nelle ultime LOOP_WINDOW azioni:
          - alternanza costante di una coppia (es. 0,1,0,1,…)
          - o ripetizione > 50% della stessa azione
        """
        if len(self.action_history) < self.LOOP_WINDOW:
            return False

        a = list(self.action_history)
        # 1) alternanza costante
        pairs = [(a[i], a[i+1]) for i in range(len(a)-1)]
        if all(p == pairs[0] for p in pairs):
            return True

        # 2) troppa ripetizione singola
        most_common, cnt = Counter(a).most_common(1)[0]
        if cnt > self.LOOP_WINDOW // 2:
            return True

        return False

    def detect_position_loop(self):
        """
        True se nelle ultime LOOP_WINDOW passi:
          - tutte le posizioni entro una soglia di pochi pixel
          - o alternanza costante di due posizioni
        """
        if len(self.pos_history) < self.LOOP_WINDOW:
            return False

        poses = list(self.pos_history)
        xs = [p[0] for p in poses]
        ys = [p[1] for p in poses]
        # 1) praticamente immobile
        if max(xs) - min(xs) < self.LOOP_DIST_THRESHOLD and \
           max(ys) - min(ys) < self.LOOP_DIST_THRESHOLD:
            return True

        # 2) alternanza costante di due posizioni
        pairs = [(poses[i], poses[i+1]) for i in range(len(poses)-1)]
        if all(p == pairs[0] for p in pairs):
            return True

        return False
    def get_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def get_current_state(self):
        pac_rect = self.player.sprite.rect
        pac_x, pac_y = pac_rect.x, pac_rect.y
        norm_x = pac_x / WIDTH
        norm_y = pac_y / HEIGHT

        # Direzione di Pac-Man
        pac_dx, pac_dy = self.player.sprite.direction
        norm_dx = pac_dx / PLAYER_SPEED
        norm_dy = pac_dy / PLAYER_SPEED

        max_distance = math.hypot(WIDTH, HEIGHT)

        # Fantasmi
        ghosts = self.ghosts.sprites()
        if ghosts:
            distances = [self.get_distance(pac_rect.center, g.rect.center) for g in ghosts]
            idx_min = np.argmin(distances)
            nearest_ghost = ghosts[idx_min]
            ghost_dist_norm = distances[idx_min] / max_distance
            ghost_dx = nearest_ghost.direction[0] / PLAYER_SPEED
            ghost_dy = nearest_ghost.direction[1] / PLAYER_SPEED
            danger_flag = int(ghost_dist_norm < 0.2)
        else:
            ghost_dist_norm = 1.0
            ghost_dx = ghost_dy = 0.0
            danger_flag = 0

        # Bacca più vicina
        berries = [(b.abs_x, b.abs_y) for b in self.berries.sprites()]
        if berries:
            berry_distances = [self.get_distance(pac_rect.center, pos) for pos in berries]
            nearest_berry_dist = min(berry_distances)
            berry_dist_norm = nearest_berry_dist / max_distance
        else:
            berry_dist_norm = 1.0

        # Stato di immunità
        is_immune = 1 if self.player.sprite.immune else 0

        # Pareti adiacenti
        walls = self.check_walls((pac_x, pac_y))
        wall_up = walls["up"]
        wall_down = walls["down"]
        wall_left = walls["left"]
        wall_right = walls["right"]

        # Stato finale come array
        state = np.array([
            norm_x, norm_y,
            norm_dx, norm_dy,
            ghost_dist_norm,
            ghost_dx, ghost_dy,
            berry_dist_norm,
            is_immune,
            wall_up, wall_down, wall_left, wall_right,
            danger_flag
        ], dtype=np.float32)

        return state

    def apply_action(self, action):
        actions_map = {
            0: (0, -PLAYER_SPEED),
            1: (0, PLAYER_SPEED),
            2: (-PLAYER_SPEED, 0),
            3: (PLAYER_SPEED, 0)
        }
        self.player.sprite.direction = actions_map[action]

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
                reward += 10.0  # Ricompensa base per bacca
                self.total_positive += 10.0
                pacman.pac_score += 10
                if berry.power_up:
                    reward += 20.0
                    self.total_positive += 20.0
                    pacman.immune = True
                    pacman.immune_time = 175
                    pacman.pac_score += 50
                berry.kill()
                pacman.n_bacche += 1
                break


        for ghost in self.ghosts.sprites():
            if pacman.rect.colliderect(ghost.rect):
                if pacman.immune:
                    reward += 20.0
                    self.total_positive += 20.0
                else:
                    reward -= 30.0
                    self.total_penalty += 30.0
        return reward

    def compute_action_reward(self, action):
        actions_map = {0: (0, -PLAYER_SPEED), 1: (0, PLAYER_SPEED), 2: (-PLAYER_SPEED, 0), 3: (PLAYER_SPEED, 0)}
        pacman = self.player.sprite
        current_pos = (pacman.rect.x, pacman.rect.y)
        dx, dy = actions_map[action]
        new_pos = (pacman.rect.x + dx, pacman.rect.y + dy)
        action_reward = 0.0
        if new_pos == current_pos:
            action_reward -= 0.02  # penalizzazione più severa per muro
            self.total_penalty += 0.02
        elif new_pos in self.visited_positions:
            action_reward -= 0.05
            self.total_penalty += 0.05
        else:
            action_reward += 0.2
            self.total_positive += 0.2
        self.visited_positions.add(new_pos)
        return action_reward

    def compute_final_bonus(self):
        if self.episode_won:
            #print("reward positiva fine episodio applicata ")
            bonus = 100.0
            self.total_positive += bonus
            return bonus
        if self.episode_lost:
            #print("reward negativa fine episodio applicata ")
            penalty = -50.0
            self.total_penalty += 50.0
            return penalty
        return 0.0

    def compute_loop_reward(self):
        if self.detect_action_loop() or self.detect_position_loop():
            penalty = -2.0
            self.total_penalty += 2.0
            return penalty
        return 0.0

    def step(self, action):
        if not self.game_over:
            self.episode_steps += 1
            self.screen.fill("black")

            pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)
            self.pos_history.append(pos)

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
                            self.episode_lost = True
                        break

            ar = self.compute_action_reward(action)
            rw = self.compute_reward()
            final_bonus= 0.0
            rwl= self.compute_loop_reward()

            self._check_game_state()
            next_state = self.get_current_state()

            done = self.episode_won or self.episode_lost or (self.episode_steps >= MAX_STEPS)
            if done:
                self.game_over = True
                final_bonus = self.compute_final_bonus()

            rw = rw + ar + rwl + final_bonus
            self.total_reward += rw

            if self.reset_pos and not (self.episode_won and self.episode_lost):
                for ghost in self.ghosts.sprites(): ghost.move_to_start_pos()
                self.player.sprite.move_to_start_pos()
                self.player.sprite.direction = (0, 0)
                self.reset_pos = False

            return next_state, rw, done

    def render(self):
        for wall in self.walls.sprites(): wall.update(self.screen)
        for berry in self.berries.sprites(): berry.update(self.screen)
        for ghost in self.ghosts.sprites(): ghost.update(self.walls_collide_list)
        self.ghosts.draw(self.screen)
        self.player.update()
        self.player.draw(self.screen)

        if self.game_over:
            self.display.game_over()

        self._dashboard()

    def reset(self):
        self.restart_level()
        self.pos_history.clear()
        return self.get_current_state()


    def update(self):

        self.screen.fill("black")

        if not self.game_over:
            pressed_key = pygame.key.get_pressed()
            self.player.sprite.animate(pressed_key, self.walls_collide_list)

            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            rw = self.compute_reward()
            self.total_reward += rw
            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    self.player.sprite.n_bacche += 1
                    berry.kill()
            if self.player.sprite.life <= 0:
                self.game_over = True
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    if not self.player.sprite.immune:
                        self.player.sprite.life -= 1
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
