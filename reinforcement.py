import random
import pickle

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
    actions = ['up', 'down', 'left', 'right']

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
        q_table[next_state] = {a: 0 for a in ['up', 'down', 'left', 'right']}

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
            reward += 50 if berry.power_up else 10
            print(f"reward: {reward}")

    # Penalità per collisioni con fantasmi
    for ghost in world.ghosts.sprites():
        #print(f"Fantasma rect: {ghost.rect}")
        if pac_rect.colliderect(ghost.rect):
            print("Collisione con fantasma!")
            if not world.player.sprite.immune:
                reward -= 100
            else:
                reward += 100
            #stampa reward
            print(f"reward: {reward}")

    # reward+=1

    return reward


import pickle

def save_q_table(filename="q_table.pkl"):
    """
    Salva la Q-table in un file.
    """
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)
    print(f"Q-table salvata in {filename}.")
