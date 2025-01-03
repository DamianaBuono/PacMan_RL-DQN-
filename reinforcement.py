import random
import pickle

from settings import CHAR_SIZE, MAP

def load_q_table(filename="q_table.pkl"):
    """
    Carica la Q-table da un file.
    """
    try:
        with open(filename, "rb") as file:
            q_table = pickle.load(file)
        print(f"Q-table caricata da {filename}.")
        return q_table
    except FileNotFoundError:
        print(f"File {filename} non trovato. Creazione di una nuova Q-table.")
        return {}

# Q-table inizialmente vuota
q_table = load_q_table()



def choose_action(state, epsilon=0.1):
    """
    Sceglie un'azione basandosi sulla politica epsilon-greedy.
    """
    actions = ['right', 'left', 'up', 'down']

    # Controlla se lo stato è nella Q-table; se non c'è, inizializzalo
    if state not in q_table:
        q_table[state] = {action: 0 for action in actions}

    # Politica epsilon-greedy
    if random.random() < epsilon:
        return random.choice(actions)  # Esplora
    else:
        return max(q_table[state], key=q_table[state].get)  # Sfrutta



def update_q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    '''
    Aggiorna la Q-table usando la formula Q-Learning.
    '''

    if next_state not in q_table:
        q_table[next_state] = {a: 0 for a in ['right', 'left', 'up', 'down']}

    max_next_q = max(q_table[next_state].values())
    q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])


def compute_reward(state, action, world):
    reward = 0

    # Rettangolo di Pac-Man
    pac_rect = world.player.sprite.rect
    #print(f"Pac-Man rect: {pac_rect}")

    # Ricompensa per raccogliere bacche
    for berry in world.berries.sprites():
        #print(f"Bacca rect: {berry.rect}")
        if pac_rect.colliderect(berry.rect):
            print("Collisione con bacca!")
            reward += 50 if berry.power_up else 20
            print(f"reward: {reward}")

    # Penalità per collisioni con fantasmi
    for ghost in world.ghosts.sprites():
        #print(f"Fantasma rect: {ghost.rect}")
        if pac_rect.colliderect(ghost.rect):
            print("Collisione con fantasma!")
            if not world.player.sprite.immune:
                reward -= 50
            else:
                reward += 100
            #stampa reward
            print(f"reward: {reward}")

    # reward+=1
        ''' # Penalità per collisione con muri
        if any(pac_rect.colliderect(wall.rect) for wall in world.walls.sprites()):
            print("Collisione con muro!")
            reward -= 5  # Penalità per toccare il muro
            print(f"reward: {reward}")'''

    if world.player.sprite.time_since_last_berry < 10:  # Controlla se il tempo è entro la soglia
        print("Bacca mangiata velocemente!")
        reward += 10  # Ricompensa per aver mangiato entro il tempo richiesto
        print(f"reward: {reward}")
    else: reward -= 12

    '''  # Rettangolo di Pac-Man
    pac_rect = world.player.sprite.rect
    pac_x, pac_y = pac_rect.center
    cell_x, cell_y = pac_x // CHAR_SIZE, pac_y // CHAR_SIZE  # Converte la posizione di Pac-Man in coordinate della mappa

            # Determina la nuova posizione in base alla direzione dell'azione
    dx, dy = world.player.sprite.direction
    next_cell_x = cell_x + dx // CHAR_SIZE
    next_cell_y = cell_y + dy // CHAR_SIZE

            # Penalità per collisioni con i muri
    if MAP[next_cell_y][next_cell_x] == "1":  # Controlla se la prossima posizione è un muro
        print("Pac-Man ha tentato di muoversi verso un muro!")
        reward -= 10  # Penalità per tentativo di attraversare un muro
        print(f"reward: {reward}")'''

    return reward

def save_q_table(filename="q_table.pkl"):
    """
    Salva la Q-table in un file.
    """
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)
    print(f"Q-table salvata in {filename}.")
