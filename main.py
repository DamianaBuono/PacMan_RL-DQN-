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
        self.agent = DQNAgent(state_size=8, action_size=4)

        # Carica un modello se fornito
        if model_path and os.path.exists(model_path):
            self.agent.model.load_state_dict(torch.load(model_path))
            print(f"Modello caricato con successo da {model_path}")
        else:
            print("Nessun modello fornito o non trovato, inizializzando un nuovo modello.")
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
        """Simula l'allenamento dell'agente con logging per TensorBoard."""
        self.log_dir = os.path.join(BASE_LOG_DIR, training_name)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"I log verranno salvati in: {self.log_dir}")

        writer = tf.summary.create_file_writer(self.log_dir)

        for episode in range(episodes):
            world = World(self.screen)
            state = np.reshape(world.get_current_state(), [1, self.agent.state_size])

            episode_reward = 0
            episode_loss = 0
            episode_length = 0

            while not world.game_over:
                action = self.agent.act(state)
                world.apply_action(action)
                world.updateRL()

                next_state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                reward = self.agent.calculate_reward(world)
                loss = self.agent.replay()

                self.agent.remember(state, action, reward, next_state, world.game_over)

                state = next_state
                episode_reward += reward
                episode_loss += loss
                episode_length += 1

                pygame.display.update()
                self.FPS.tick(30)

            with writer.as_default():
                tf.summary.scalar('environment/cumulative_reward', episode_reward, step=episode)
                tf.summary.scalar('environment/episode_length', episode_length, step=episode)
                tf.summary.scalar('losses/policy_loss', episode_loss, step=episode)
                tf.summary.scalar('losses/value_loss', episode_loss, step=episode)

            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")

        writer.close()

        # Salva il modello
        model_path = os.path.join(BASE_MODEL_DIR, f"{training_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth")
        torch.save(self.agent.model.state_dict(), model_path)
        print(f"Modello salvato in {model_path}")

if __name__ == "__main__":
    # model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_2_20230325-150530.pth"
    # Passa il percorso del modello all'inizializzazione Main(screen, model_path=model_path)

    play = Main(screen)  # Inizializza senza modello (prima volta)
    mode = "training"  # Scegli tra "training" e "game"

    if mode == "training":
        training_name = "Training_Pacman_5"  # Imposta qui il nome del training
        play.simulate_training(episodes=10, training_name=training_name)
    elif mode == "game":
        play.main()
