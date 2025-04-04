import pygame, sys
import numpy as np
import tensorflow as tf
import torch
import os
import pandas as pd
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
        self.agent = DQNAgent(state_size=16, action_size=4)

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
        """Simula l'allenamento dell'agente con logging per TensorBoard e salvataggio dei risultati in Excel."""
        log_dir = os.path.join(BASE_LOG_DIR, training_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"I log verranno salvati in: {log_dir}")

        writer = tf.summary.create_file_writer(log_dir)
        # Lista per salvare i dati per ogni episodio
        episodes_data = []

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
                loss = self.agent.replay()  # Aggiornamento del modello (learning step)

                # Memorizza l'esperienza dopo il replay
                self.agent.remember(state, action, reward, next_state, world.game_over)

                state = next_state
                episode_reward += reward
                episode_loss += loss
                episode_length += 1

                #pygame.display.update()
                #self.FPS.tick(30)

            # Scrive i dati in TensorBoard
            with writer.as_default():
                tf.summary.scalar('environment/cumulative_reward', episode_reward, step=episode)
                tf.summary.scalar('environment/episode_length', episode_length, step=episode)
                tf.summary.scalar('losses/policy_loss', episode_loss, step=episode)
                tf.summary.scalar('losses/value_loss', episode_loss, step=episode)

            # Al termine dell'episodio, raccogliamo le informazioni:
            lives = world.player.sprite.life
            berries_eaten = world.player.sprite.n_bacche

            print(f"Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}, "
                  f"Vite rimanenti = {lives}, Bacche mangiate = {berries_eaten}")

            episodes_data.append({
                "Episode": episode + 1,
                "Cumulative_Reward": episode_reward,
                "Remaining_Lives": lives,
                "Berries_Eaten": berries_eaten
            })

        writer.close()

        # Salva i risultati in un file Excel
        results_df = pd.DataFrame(episodes_data)
        # Creiamo un nome file con timestamp per evitare sovrascritture
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(BASE_LOG_DIR, training_name, f"training_results_{timestamp}.xlsx")
        results_df.to_excel(results_file, index=False)
        print(f"Risultati salvati in {results_file}")

        # Salva il modello con timestamp per evitare sovrascritture
        model_filename = f"{training_name}_{timestamp}.pth"
        model_path = os.path.join(BASE_MODEL_DIR, model_filename)
        torch.save(self.agent.model.state_dict(), model_path)
        print(f"Modello salvato in {model_path}")

    def simulate_testing(self, episodes):
        """Simula il testing dell'agente con il modello caricato."""
        # Imposta epsilon a 0 per testare la policy deterministica
        self.agent.epsilon = 0.0
        scores = []

        for episode in range(episodes):
            world = World(self.screen)
            state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
            episode_reward = 0

            while not world.game_over:
                action = self.agent.act(state)  # Azione basata esclusivamente sulla policy
                world.apply_action(action)
                world.updateRL()

                next_state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                reward = self.agent.calculate_reward(world)
                state = next_state

                episode_reward += reward

                pygame.display.update()
                self.FPS.tick(30)

            scores.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

        print(f"Average Reward over {episodes} episodes: {np.mean(scores)}")

if __name__ == "__main__":
    # Scegli la modalità: "training", "game" o "testing"
    mode = "training"  # Modifica questa variabile in base alla modalità desiderata

    if mode == "training":
        play = Main(screen)  # Avvio senza modello pre-caricato
        training_name = "Training_Pacman_NoPER_NoNoisy5"  # Nome del training
        play.simulate_training(episodes=10, training_name=training_name)
    elif mode == "testing":
        # Per il testing, è necessario caricare un modello già addestrato
        model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_2_20230325-150530.pth"
        play = Main(screen, model_path=model_path)
        play.simulate_testing(episodes=10)
    elif mode == "game":
        play = Main(screen)  # Avvio senza modello pre-caricato
        play.main()
