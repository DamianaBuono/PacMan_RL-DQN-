import pygame, sys
import numpy as np
import tensorflow as tf
import torch
import os
from datetime import datetime

from reinforcementDQL import DQNAgent
from settings import WIDTH, HEIGHT, NAV_HEIGHT
from worldDQL import World

# Inizializzazione di pygame
pygame.init()

# Impostazione della finestra del gioco
screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
pygame.display.set_caption("PacMan")

# Cartella base per il log di TensorBoard
BASE_LOG_DIR = r"C:\Users\claud\PycharmProjects\tensorboard"

# Percorso base per il salvataggio dei modelli
BASE_MODEL_DIR = r"C:\Users\claud\Desktop\IA\ModelliSalvati"

class Main:
    def __init__(self, screen, model_path=None):
        self.screen = screen
        self.FPS = pygame.time.Clock()
        # Aggiorno state_size a 8 per essere coerente con world.get_current_state()
        self.agent = DQNAgent(state_size=8, action_size=4)

        # Dati per tracciare ricompensa e loss
        self.episode_rewards = []
        self.episode_losses = []

        # Carica un modello se viene fornito un percorso
        if model_path:
            if os.path.exists(model_path):
                self.agent.model.load_state_dict(torch.load(model_path))
                print(f"Modello caricato con successo da {model_path}")
            else:
                print(f"Impossibile trovare il modello nel percorso {model_path}, inizializzando un nuovo modello.")
        else:
            print("Nessun percorso modello fornito, verrà utilizzato un nuovo modello.")

    def main(self):
        """Modalità manuale di gioco"""
        world = World(self.screen)
        while True:
            self.screen.fill("black")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            world.update()
            pygame.display.update()
            self.FPS.tick(30)

    def simulate_training(self, episodes, training_name):
        """Simula l'allenamento dell'agente per un certo numero di episodi"""

        # Creazione della cartella di log specifica per il training
        self.log_dir = os.path.join(BASE_LOG_DIR, training_name)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"I log verranno salvati in: {self.log_dir}")

        reward_all_episodes = []
        loss_all_episodes = []

        writer = tf.summary.create_file_writer(self.log_dir)  # Creazione del writer per TensorBoard

        for episode in range(episodes):
            world = World(self.screen)
            # Otteniamo lo stato corrente e lo trasformiamo in un array 2D con dimensione (1, 8)
            state = world.get_current_state()
            state = np.reshape(state, [1, self.agent.state_size])

            episode_loss = 0  # Variabile per la loss per episodio
            episode_reward = 0

            while not world.game_over:
                action = self.agent.act(state)
                world.apply_action(action)
                world.updateRL()

                next_state = world.get_current_state()
                next_state = np.reshape(next_state, [1, self.agent.state_size])

                reward = self.agent.calculate_reward(world)
                episode_reward += reward

                self.agent.remember(state, action, reward, next_state, world.game_over)
                loss = self.agent.replay()
                episode_loss += loss

                state = next_state

                pygame.display.update()
                self.FPS.tick(30)

            reward_all_episodes.append(episode_reward)
            loss_all_episodes.append(episode_loss)

            if (episode + 1) % 100 == 0:
                mean_reward = np.mean(reward_all_episodes[-100:])
                print(f"Episode {episode + 1}: Average Reward (last 100 episodes): {mean_reward}")
                with writer.as_default():
                    tf.summary.scalar('mean_reward_last_100', mean_reward, step=episode)

            with writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=episode)
                tf.summary.scalar('loss', episode_loss, step=episode)

        # Salva il modello usando il nome del training e un timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(BASE_MODEL_DIR, f"{training_name}_{timestamp}.pth")
        torch.save(self.agent.model.state_dict(), model_path)
        print(f"Modello salvato con successo in {model_path}")
        writer.close()


# Blocco principale
if __name__ == "__main__":
    # model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_2_20230325-150530.pth"
    # Passa il percorso del modello all'inizializzazione Main(screen, model_path=model_path)

    play = Main(screen)  # Inizializza senza modello (prima volta)
    mode = "training"  # Scegli tra "training" e "game"

    if mode == "training":
        training_name = "Training_Pacman_2"  # Imposta qui il nome del training
        play.simulate_training(episodes=500, training_name=training_name)
    elif mode == "game":
        play.main()
