import pygame

from settings import CHAR_SIZE, PLAYER_SPEED
from animation import import_sprite


class Pac(pygame.sprite.Sprite):
    def __init__(self, row, col):
        super().__init__()
        self.time_since_last_berry = 0
        self.abs_x = (row * CHAR_SIZE)
        self.abs_y = (col * CHAR_SIZE)

        # pac animation
        self._import_character_assets()
        self.frame_index = 0
        self.animation_speed = 0.5
        self.image = self.animations["idle"][self.frame_index]
        self.rect = self.image.get_rect(topleft=(self.abs_x, self.abs_y))
        self.mask = pygame.mask.from_surface(self.image)

        self.pac_speed = PLAYER_SPEED
        self.immune_time = 0
        self.immune = False

        self.directions = {'left': (-PLAYER_SPEED, 0), 'right': (PLAYER_SPEED, 0), 'up': (0, -PLAYER_SPEED),
                           'down': (0, PLAYER_SPEED)}
        self.keys = {'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'up': pygame.K_UP, 'down': pygame.K_DOWN}
        self.direction = (0, 0)

        # pac status
        self.status = "idle"
        self.life = 3
        self.pac_score = 0
        self.n_bacche = 0
        self.combo_counter = 0
        self.milestones_rewarded = set()
        self.last_position = (self.rect.x, self.rect.y)
        self.last_direction = None

    # gets all the image needed for animating specific player action
    def _import_character_assets(self):
        character_path = "assets/pac/"
        self.animations = {
            "up": [],
            "down": [],
            "left": [],
            "right": [],
            "idle": [],
            "power_up": []
        }
        for animation in self.animations.keys():
            full_path = character_path + animation
            self.animations[animation] = import_sprite(full_path)

    def _is_collide(self, x, y):
        tmp_rect = self.rect.move(x, y)
        if tmp_rect.collidelist(self.walls_collide_list) == -1:
            return False
        return True

    def move_to_start_pos(self):
        self.rect.x = self.abs_x
        self.rect.y = self.abs_y
        self.last_direction = None

    # update with sprite/sheets
    def animate(self, pressed_key, walls_collide_list):
        animation = self.animations[self.status]

        # loop over frame index
        self.frame_index += self.animation_speed
        if self.frame_index >= len(animation):
            self.frame_index = 0
        image = animation[int(self.frame_index)]
        self.image = pygame.transform.scale(image, (CHAR_SIZE, CHAR_SIZE))

        self.walls_collide_list = walls_collide_list
        for key, key_value in self.keys.items():
            if pressed_key[key_value] and not self._is_collide(*self.directions[key]):
                self.direction = self.directions[key]
                self.status = key if not self.immune else "power_up"
                break

        if not self._is_collide(*self.direction):
            self.rect.move_ip(self.direction)
            self.status = self.status if not self.immune else "power_up"
        if self._is_collide(*self.direction):
            self.status = "idle" if not self.immune else "power_up"

    def animateRL(self, action_direction, walls_collide_list):
        # Mappa numeri a direzioni
        action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        # Verifica se l'azione passata è valida
        if action_direction not in action_map:
            print(f"Errore: l'azione {action_direction} non è valida")
            action_direction = 0  # Imposta un valore di default (ad esempio 'up') in caso di errore

        # Imposta la direzione basandoti sull'azione dell'agente RL
        self.direction = self.directions[action_map[action_direction]]

        # Recupera l'animazione corrente
        animation = self.animations[self.status]

        # Ciclo sull'indice dei frame
        self.frame_index += self.animation_speed
        if self.frame_index >= len(self.animations[self.status]):
            self.frame_index = 0
        image = self.animations[self.status][int(self.frame_index)]
        self.image = pygame.transform.scale(image, (CHAR_SIZE, CHAR_SIZE))

        # Verifica collisioni e aggiorna lo stato e la posizione
        self.walls_collide_list = walls_collide_list
        if not self._is_collide(*self.direction):
            self.rect.move_ip(self.direction)
            self.status = self.status if not self.immune else "power_up"
        else:
            self.status = "idle" if not self.immune else "power_up"

        self.last_position = (self.rect.x, self.rect.y)

    def update(self):
        # Timer based from FPS count
        self.immune = True if self.immune_time > 0 else False
        self.immune_time -= 1 if self.immune_time > 0 else 0

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
