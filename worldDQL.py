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
        self.agent = DQNAgent(state_size=6, action_size=4)  # Supponiamo che state_size=6 (posizione PacMan, direzione, fantasmi, bacche) e action_size=4 (su, giù, sinistra, destra)

        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.berries = pygame.sprite.Group()

        self.display = Display(self.screen)

        self.game_over = False
        self.reset_pos = False
        self.player_score = 0
        self.game_level = 1

        self._generate_world()


    # Crea e aggiungi il giocatore alla schermata
    def _generate_world(self):
        # Renderizza gli ostacoli dalla mappa
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == "1":  # Per i muri
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                elif char == " ":  # Per i percorsi da riempire con bacche
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":  # Per le grandi bacche
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))

                # Per la posizione iniziale dei fantasmi
                elif char == "s":
                    self.ghosts.add(Ghost(x_index, y_index, "skyblue"))
                elif char == "p":
                    self.ghosts.add(Ghost(x_index, y_index, "pink"))
                elif char == "o":
                    self.ghosts.add(Ghost(x_index, y_index, "orange"))
                elif char == "r":
                    self.ghosts.add(Ghost(x_index, y_index, "red"))

                elif char == "P":  # Per la posizione iniziale di PacMan
                    self.player.add(Pac(x_index, y_index))

        self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]

    def generate_new_level(self):
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == " ":  # Per i percorsi da riempire con bacche
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
                elif char == "B":  # Per le grandi bacche
                    self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
        time.sleep(2)

    def restart_level(self):
        self.berries.empty()
        [ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
        self.game_level = 1
        self.player.sprite.pac_score = 0
        self.player.sprite.life = 3
        self.player.sprite.move_to_start_pos()
        self.player.sprite.direction = (0, 0)
        self.player.sprite.status = "idle"
        self.generate_new_level()

    # Mostra la barra di navigazione
    def _dashboard(self):
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)

        self.display.show_life(self.player.sprite.life)
        self.display.show_level(self.game_level)
        self.display.show_score(self.player.sprite.pac_score)

    def _check_game_state(self):
        # Verifica se il gioco è finito
        if self.player.sprite.life == 0:
            self.game_over = True

        # Genera un nuovo livello
        if len(self.berries) == 0 and self.player.sprite.life > 0:
            self.game_level += 1
            for ghost in self.ghosts.sprites():
                ghost.move_speed += self.game_level
                ghost.move_to_start_pos()

            self.player.sprite.move_to_start_pos()
            self.player.sprite.direction = (0, 0)
            self.player.sprite.status = "idle"
            self.generate_new_level()

    # Restituisce lo stato attuale
    def get_current_state(self):
        pac_pos = (self.player.sprite.rect.x, self.player.sprite.rect.y)
        pac_direction = self.player.sprite.direction
        ghosts = tuple((ghost.rect.x, ghost.rect.y) for ghost in self.ghosts.sprites())
        berries = tuple((berry.abs_x, berry.abs_y) for berry in self.berries.sprites())
        power_up_active = self.player.sprite.immune
        state = np.array([pac_pos[0], pac_pos[1], pac_direction[0], pac_direction[1], len(ghosts), len(berries)])
        return state

    def apply_action(self, action):
        """Applica l'azione scelta dall'agente"""
        actions_map = {
            0: (0, -PLAYER_SPEED),  # Su
            1: (0, PLAYER_SPEED),  # Giù
            2: (-PLAYER_SPEED, 0),  # Sinistra
            3: (PLAYER_SPEED, 0)  # Destra
        }
        self.player.sprite.direction = actions_map[action]

    def get_reward(self):
        reward = 0

        # PacMan colleziona una bacca
        for berry in self.berries.sprites():
            if self.player.sprite.rect.colliderect(berry.rect):
                if berry.power_up:
                    reward += 50  # Ricompensa maggiore per le bacche grandi
                else:
                    reward += 10  # Ricompensa minore per le bacche normali
                berry.kill()  # Rimuovi la bacca dalla mappa

        # PacMan collide con un fantasma
        for ghost in self.ghosts.sprites():
            if self.player.sprite.rect.colliderect(ghost.rect):
                if not self.player.sprite.immune:
                    reward -=30  # Penalità per collisione con il fantasma
                    self.player.sprite.life -= 1  # Riduci una vita
                    if self.player.sprite.life <= 0:
                       self.game_over = True  # Game over quando le vite finiscono
                else:
                    ghost.move_to_start_pos()  # Il fantasma torna alla posizione iniziale
                    reward += 100  # Ricompensa per aver "mangiato" un fantasma se immune

        # Completamento del livello
        if len(self.berries) == 0 and self.player.sprite.life > 0:
            reward += 50  # Ricompensa per completare il livello

        # Game over
        if self.game_over:
            reward -= 50  # Penalità per game over

        return reward

    total_reward = 0

    def updateRL(self):
        """Aggiorna il gioco durante la fase di reinforcement learning"""
        if not self.game_over:
            self.screen.fill("black")

            # Stato attuale
            current_state = self.get_current_state()
            action = self.agent.act(current_state)

            # Applicazione dell'azione e aggiornamento dello stato
            self.apply_action(action)
            self.player.sprite.animateRL(action, self.walls_collide_list)

            # Calcolo ricompensa
            reward = self.get_reward()
            next_state = self.get_current_state()
            done = self.game_over

            # Memorizzazione e apprendimento
            self.agent.remember(current_state, action, reward, next_state, done)
            self.agent.replay()

            # Debug ricompense
            self.total_reward += reward
            print("Total Reward:", self.total_reward)

            # Rendering corretto
            self._dashboard()
            [wall.update(self.screen) for wall in self.walls.sprites()]
            [berry.update(self.screen) for berry in self.berries.sprites()]
            [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
            self.ghosts.draw(self.screen)
            self.player.update()
            self.player.draw(self.screen)
            self.display.game_over() if self.game_over else None

    # GIoco non autonomo
    def update(self):
        if not self.game_over:
            # Movimento del giocatore
            pressed_key = pygame.key.get_pressed()
            self.player.sprite.animate(pressed_key, self.walls_collide_list)

            # Teletrasporto al lato opposto della mappa
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0

            # PacMan mangia le bacche
            for berry in self.berries.sprites():
                if self.player.sprite.rect.colliderect(berry.rect):
                    if berry.power_up:
                        self.player.sprite.immune_time = 150  # Timer basato sul conteggio FPS
                        self.player.sprite.pac_score += 50
                    else:
                        self.player.sprite.pac_score += 10
                    berry.kill()

            # PacMan collide con i fantasmi
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

        # Rendering
        [wall.update(self.screen) for wall in self.walls.sprites()]
        [berry.update(self.screen) for berry in self.berries.sprites()]
        [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
        self.ghosts.draw(self.screen)

        self.player.update()
        self.player.draw(self.screen)
        self.display.game_over() if self.game_over else None

        self._dashboard()

        # Reset posizione di PacMan e dei fantasmi dopo che PacMan viene catturato
        if self.reset_pos and not self.game_over:
            [ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
            self.player.sprite.move_to_start_pos()
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0, 0)
            self.reset_pos = False

        # Per il pulsante di riavvio
        if self.game_over:
            pressed_key = pygame.key.get_pressed()
            if pressed_key[pygame.K_r]:
                self.game_over = False
                self.restart_level()
