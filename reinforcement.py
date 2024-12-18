import random

# Q-table inizialmente vuota
q_table = {}


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
    """
    Calcola la ricompensa basandosi sullo stato e sull'azione.
    """
    reward = 0

    # Ricompensa per raccogliere bacche
    pac_rect = world.player.sprite.rect
    for berry in world.berries.sprites():
        if pac_rect.colliderect(berry.rect):
            reward += 50 if berry.power_up else 10

    # Penalità per collisioni con fantasmi
    for ghost in world.ghosts.sprites():
        if pac_rect.colliderect(ghost.rect):
            if not world.player.sprite.immune:
                reward -= 100
            else:
                reward += 100

    # Bonus per sopravvivenza
    reward += 1

    return reward
