import pygame, sys, os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from settings import WIDTH, HEIGHT, NAV_HEIGHT
from worldDQL import World
from reinforcementDQL import DQNAgent  # L'agente ora non gestisce più il reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializzazione di pygame
pygame.init()

# Impostazione della finestra di gioco con area per il navigational bar
screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
pygame.display.set_caption("PacMan")

# Directory per TensorBoard e per il salvataggio dei modelli
BASE_LOG_DIR = r"C:\Users\claud\PycharmProjects\tensorboard"
BASE_MODEL_DIR = r"C:\Users\claud\Desktop\IA\ModelliSalvati"


class Main:
    def __init__(self, screen, model_path=None):
        self.screen = screen
        self.FPS = pygame.time.Clock()
        self.agent = DQNAgent(state_size=19, action_size=4)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Gestione dei diversi formati di salvataggio:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.model.load_state_dict(checkpoint['model_state_dict'])
                self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Modello e ottimizzatore caricati con successo da {model_path}")
            else:
                # Se il checkpoint non contiene le chiavi previste, assumiamo di avere solo lo state_dict del modello
                self.agent.model.load_state_dict(checkpoint)
                print(f"Solo modello caricato da {model_path} (nessun ottimizzatore)")
        else:
            print("Nessun modello fornito o non trovato, inizializzando un nuovo modello.")

    def main(self):
        """Modalità manuale di gioco."""
        world = World(self.screen)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # Modalità manuale: aggiorna l’ambiente (il world gestisce tutta la logica incluso il reward)
            world.update()
            pygame.display.update()
            self.FPS.tick(30)

    def simulate_training(self, episodes, training_name):
        # Prepara directory per log e salvataggio modelli
        save_dir = os.path.join(BASE_MODEL_DIR, training_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Tutti i file verranno salvati in: {save_dir}")

        # Configura il writer per TensorBoard
        log_dir = os.path.join(save_dir, "tensorboard_logs")
        writer = SummaryWriter(log_dir=log_dir)

        episodes_data = []

        # Creazione di un set fisso di stati di validazione
        validation_states = []
        num_val_states = 50
        val_world = World(self.screen)
        for _ in range(num_val_states):
            # Aggiorna l'ambiente per raccogliere stati variegati
            val_world.updateRL()
            validation_states.append(val_world.get_current_state())
        validation_states = np.stack(validation_states)  # shape: (num_val_states, state_size)

        try:
            for episode in range(episodes):
                # Inizializzazione dell'ambiente per l’episodio
                world = World(self.screen)
                self.agent.reset_episode()
                state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                episode_reward = 0
                episode_steps = 0
                episode_losses = []

                # Ciclo dell’episodio gestito interamente dal world (con updateRL che gestisce anche il reward)
                while not world.game_over:
                    action = self.agent.act(state)
                    world.apply_action(action)
                    world.updateRL()

                    next_state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                    state = next_state
                    episode_steps += 1
                    episode_reward = world.total_reward

                    if world.loss:
                        episode_losses.append(world.loss)
                    pygame.display.update()
                    self.FPS.tick(30)

                # Calcolo della media delle reward (reward medio per step)
                avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0

                lives = world.player.sprite.life
                berries = world.player.sprite.n_bacche
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0

                print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {episode_steps}, "
                      f"Lives = {lives}, Berries = {berries}, Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")

                episodes_data.append({
                    "Episode": episode + 1,
                    "Cumulative_Reward": episode_reward,
                    "Average_Reward": avg_reward,  # Aggiunto il reward medio
                    "Steps": episode_steps,
                    "Remaining_Lives": lives,
                    "Berries_Eaten": berries,
                    "Average_Loss": avg_loss
                })

                # Logging su TensorBoard
                writer.add_scalar("Reward/Cumulative", episode_reward, episode)
                writer.add_scalar("Reward/Average", avg_reward, episode)  # Logging della media delle reward
                writer.add_scalar("Agent/Avg_Loss", avg_loss, episode)
                writer.add_scalar("Env/Remaining_Lives", lives, episode)
                writer.add_scalar("Env/Berries_Eaten", berries, episode)
                writer.add_scalar("Episode/Steps", episode_steps, episode)

                # Calcolo e logging della media dei Q-valori sui validation_states
                with torch.no_grad():
                    q_values = self.agent.model(torch.FloatTensor(validation_states).to(device))
                    avg_q = q_values.mean().item()
                writer.add_scalar("Validation/Avg_Q", avg_q, episode)

        except KeyboardInterrupt:
            print("\nTraining interrotto manualmente. Salvataggio dei dati...")
        finally:
            writer.close()

            # Salvataggio dei dati di training
            results_df = pd.DataFrame(episodes_data)
            results_file = os.path.join(save_dir, f"training_results({training_name}).xlsx")
            results_df.to_excel(results_file, index=False)
            print(f"Risultati salvati in {results_file}")

            # Salvataggio dei grafici
            if not results_df.empty:
                # Grafico della reward cumulativa
                plt.figure(figsize=(10, 6))
                plt.plot(results_df["Episode"], results_df["Cumulative_Reward"], marker='o', linestyle='-', color='blue')
                plt.title('Cumulative Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Cumulative Reward')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "training_CumulativeReward.png"))
                plt.close()

                # Grafico della reward media
                plt.figure(figsize=(10, 6))
                plt.plot(results_df["Episode"], results_df["Average_Reward"], marker='o', linestyle='-', color='purple')
                plt.title('Average Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "training_AverageReward.png"))
                plt.close()

                # Grafico delle vite residue
                plt.figure(figsize=(10, 6))
                plt.plot(results_df["Episode"], results_df["Remaining_Lives"], marker='o', linestyle='-', color='orange')
                plt.title('Remaining Lives per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Remaining Lives')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "training_RemainingLives.png"))
                plt.close()

                # Grafico delle bacche mangiate
                plt.figure(figsize=(10, 6))
                plt.plot(results_df["Episode"], results_df["Berries_Eaten"], marker='o', linestyle='-', color='green')
                plt.title('Berries Eaten per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Berries Eaten')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "training_BerriesEaten.png"))
                plt.close()

                # Grafico della loss media
                plt.figure(figsize=(10, 6))
                plt.plot(results_df["Episode"], results_df["Average_Loss"], marker='o', linestyle='-', color='red')
                plt.title('Average Loss per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "training_Loss.png"))
                plt.close()

            # Salvataggio finale del modello e dell'ottimizzatore (checkpoint completo)
            model_filename = f"{training_name}.pth"
            model_path = os.path.join(save_dir, model_filename)
            torch.save({
                'model_state_dict': self.agent.model.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict()
            }, model_path)
            print(f"Checkpoint completo salvato in {model_path}")

    def simulate_testing(self, episodes):
        """Simula il testing dell'agente con il modello caricato."""
        self.agent.epsilon = 0.0  # Policy deterministica durante il testing
        scores = []
        for episode in range(episodes):
            world = World(self.screen)
            state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
            episode_reward = 0
            while not world.game_over:
                action = self.agent.act(state)
                world.apply_action(action)
                world.updateRL()
                next_state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                state = next_state
                pygame.display.update()
                self.FPS.tick(30)
            scores.append(world.total_reward)
            print(f"Test Episode {episode + 1}: Reward = {world.total_reward}")
        print(f"Average Reward over {episodes} episodes: {np.mean(scores)}")


if __name__ == "__main__":
    # Modalità: "training", "testing" o "game"
    mode = "training"  # Modifica in base alla modalità desiderata

    if mode == "training":
        # Esempio: specifica il path al modello se vuoi continuare un training
        # model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_7.3\Training_Pacman_7.3.pth"
        main_obj = Main(screen)
        training_name = "Training_Pacman_7.5"
        main_obj.simulate_training(episodes=500, training_name=training_name)
    elif mode == "testing":
        model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_Improved\Training_Pacman_Improved.pth"
        main_obj = Main(screen, model_path=model_path)
        main_obj.simulate_testing(episodes=10)
    elif mode == "game":
        main_obj = Main(screen)
        main_obj.main()
