
import math
import random
import pickle
import json

from settings import CHAR_SIZE, MAP

def load_q_table(filename="q_table.pkl"):
    """
    Carica la Q-table da un file. Se il file non esiste o è vuoto, crea una nuova Q-table.
    """
    try:
        with open(filename, "rb") as file:
            if file.peek(1):  # Controlla se il file non è vuoto
                q_table = pickle.load(file)
                print(f"Q-table caricata da {filename}.")
                return q_table
            else:
                print(f"File {filename} è vuoto. Creazione di una nuova Q-table.")
                return {}
    except FileNotFoundError:
        print(f"File {filename} non trovato. Creazione di una nuova Q-table.")
        return {}
    except EOFError:
        print(f"Errore nella lettura del file {filename}. Creazione di una nuova Q-table.")
        return {}

def load_q_table_json(filename="q_table.json"):
    """
    Carica la Q-table da un file JSON in formato leggibile.
    """
    try:
        with open(filename, "r") as file:
            q_table_serializable = json.load(file)
        # Converti le chiavi stringa in tuple
        q_table = {eval(key): value for key, value in q_table_serializable.items()}
        print(f"Q-table caricata da {filename}.")
        return q_table
    except FileNotFoundError:
        print(f"File {filename} non trovato. Creazione di una nuova Q-table.")
        return {}


# Caricamento q table pkl
#q_table = load_q_table()

#caricamento q-table json
q_table = load_q_table_json()




'''def choose_action(state, epsilon=0.3):
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
        return max(q_table[state], key=q_table[state].get)  # Sfrutta'''

def choose_action_epsilon_greedy(state, epsilon=0.3):
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

def choose_action_softmax(state, tau=1.0):
    """
    Sceglie un'azione basandosi sulla Softmax Policy.
    :param state: Lo stato corrente
    :param tau: Il parametro di temperatura (controlla l'esplorazione)
    """
    actions = ['right', 'left', 'up', 'down']

    # Controlla se lo stato è nella Q-table; se non c'è, inizializzalo
    if state not in q_table:
        q_table[state] = {action: 0 for action in actions}

    # Calcola le probabilità Softmax
    q_values = q_table[state]
    exp_q_values = {action: math.exp(q / tau) for action, q in q_values.items()}
    sum_exp_q_values = sum(exp_q_values.values())
    probabilities = {action: exp / sum_exp_q_values for action, exp in exp_q_values.items()}

    # Seleziona un'azione in base alle probabilità
    actions, probs = zip(*probabilities.items())
    chosen_action = random.choices(actions, weights=probs, k=1)[0]

    return chosen_action

def choose_action(state, method="softmax", epsilon=0.3, tau=1.0):
    """
    Sceglie un'azione in base al metodo specificato.
    :param method: Metodo di selezione ("epsilon-greedy" o "softmax")
    :param epsilon: Parametro epsilon per epsilon-greedy
    :param tau: Parametro di temperatura per Softmax
    """
    if method == "epsilon-greedy":
        return choose_action_epsilon_greedy(state, epsilon)
    elif method == "softmax":
        return choose_action_softmax(state, tau)
    else:
        raise ValueError("Metodo sconosciuto per la selezione dell'azione.")

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
            #print("Collisione con bacca!")
            reward += 50 if berry.power_up else 20
            #print(f"reward: {reward}")

    # Penalità per collisioni con fantasmi
    for ghost in world.ghosts.sprites():
        #print(f"Fantasma rect: {ghost.rect}")
        if pac_rect.colliderect(ghost.rect):
            #print("Collisione con fantasma!")
            if not world.player.sprite.immune:
                reward -= 100
            else:
                reward += 100
            #stampa reward
            #print(f"reward: {reward}")

    reward+=5


    if world.player.sprite.time_since_last_berry > 10:  # Controlla se il tempo è entro la soglia
        print("Bacca mangiata velocemente!", reward)
        reward += -10  # Ricompensa per aver mangiato entro il tempo richiesto
        print("NON mangia la bacca in tempo: ", reward)
        #print(f"reward: {reward}")



    return reward

def save_q_table(filename="q_table.pkl"):
    """
    Salva la Q-table in un file.
    """
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)
    print(f"Q-table salvata in {filename}.")

def save_q_table_json(filename="q_table.json"):
    """
    Salva la Q-table in un file JSON in formato leggibile.
    """
    # Converti le chiavi tuple in stringhe
    q_table_serializable = {str(key): value for key, value in q_table.items()}
    with open(filename, "w") as file:
        json.dump(q_table_serializable, file, indent=4)  # Usa `indent` per rendere il file leggibile
    print(f"Q-table salvata in formato leggibile in {filename}.")