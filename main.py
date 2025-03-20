import pygame, sys
import numpy as np
import tensorflow as tf
from reinforcementDQL import DQNAgent
from settings import WIDTH, HEIGHT, NAV_HEIGHT
from worldDQL import World

# Inizializzazione di pygame
pygame.init()

# Impostazione della finestra del gioco
screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
pygame.display.set_caption("PacMan")

# Cartella per il log di TensorBoard
log_dir = r"C:\Users\claud\PycharmProjects\tensorboard"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class Main:
    def __init__(self, screen):
        self.screen = screen
        self.FPS = pygame.time.Clock()
        self.agent = DQNAgent(state_size=6, action_size=4)

        # Dati per tracciare la ricompensa e la loss
        self.episode_rewards = []
        self.episode_losses = []

    def main(self):
        """Modalità manuale di gioco"""
        world = World(self.screen)
        while True:
            self.screen.fill("black")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Aggiorna il mondo
            world.update()
            # Rendering
            pygame.display.update()
            self.FPS.tick(30)

    def simulate_training(self, episodes):
        """Simula l'allenamento dell'agente per un certo numero di episodi"""
        reward_all_episode = 0
        reward_all_episodes = []
        loss_all_episodes = []

        writer = tf.summary.create_file_writer(log_dir)  # Creazione del writer per TensorBoard

        for episode in range(episodes):
            print(f"Training Episode {episode + 1}/{episodes}")

            world = World(self.screen)
            state = world.get_current_state()
            state = np.reshape(state, [1, self.agent.state_size])

            episode_loss = 0  # Variabile per la loss per episodio

            while not world.game_over:
                action = self.agent.act(state)
                world.apply_action(action)
                next_state = world.get_current_state()
                next_state = np.reshape(next_state, [1, self.agent.state_size])

                world.updateRL()

                reward = world.total_reward
                done = world.game_over
                loss = self.agent.replay()  # Ottieni la loss durante il replay

                if loss is not None:  # Evita errori se la loss è None
                    episode_loss += loss

                state = next_state

                pygame.display.update()
                self.FPS.tick(30)

            reward_all_episode += world.total_reward
            reward_all_episodes.append(world.total_reward)
            loss_all_episodes.append(episode_loss)

            print(f"Episode {episode} completed. Total Reward: {world.total_reward}. Cumulative: {reward_all_episode}")

            # Ogni 1000 episodi, calcola la media dei reward e della loss
            if (episode + 1) % 10000 == 0:
                mean_reward = np.mean(reward_all_episodes[-10000:])
                mean_loss = np.mean(loss_all_episodes[-10000:])
                print(f"Average Reward over last 1000 episodes: {mean_reward}")
                print(f"Average Loss over last 1000 episodes: {mean_loss}")

                # Log su TensorBoard
                with writer.as_default():
                    tf.summary.scalar('mean_reward_last_1000', mean_reward, step=episode)
                    tf.summary.scalar('mean_loss_last_1000', mean_loss, step=episode)

            # Log delle metriche di ogni episodio
            with writer.as_default():
                tf.summary.scalar('reward', world.total_reward, step=episode)
                tf.summary.scalar('loss', episode_loss, step=episode)

        writer.close()  # Chiudi il writer dopo l'allenamento


# Blocco principale
if __name__ == "__main__":
    play = Main(screen)

    # Modalità: scegli tra "training" o "game"
    mode = "training"

    if mode == "training":
        play.simulate_training(episodes=50000)  # Esegui l'allenamento per 500.000 episodi
    elif mode == "game":
        play.main()
