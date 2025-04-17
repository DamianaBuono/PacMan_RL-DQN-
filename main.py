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
        save_dir = os.path.join(BASE_MODEL_DIR, training_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Tutti i file verranno salvati in: {save_dir}")

        log_dir = os.path.join(save_dir, "tensorboard_logs")
        writer = SummaryWriter(log_dir=log_dir)
        episodes_data = []

        validation_states = []
        num_val_states = 50
        val_world = World(self.screen)
        for _ in range(num_val_states):
            val_world.updateRL()
            validation_states.append(val_world.get_current_state())
        validation_states = np.stack(validation_states)

        try:
            for episode in range(episodes):
                world = World(self.screen)
                self.agent.reset_episode()
                state = np.reshape(world.get_current_state(), [1, self.agent.state_size])
                episode_reward = 0
                episode_steps = 0
                episode_losses = []

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

                avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
                lives = world.player.sprite.life
                berries = world.player.sprite.n_bacche
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                penalty_total = world.total_penalty
                positive_total = world.total_positive

                print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {episode_steps}, "
                      f"Lives = {lives}, Berries = {berries}, Avg Loss = {avg_loss:.4f}, "
                      f"Avg Reward = {avg_reward:.4f}, Total Penalty = {penalty_total}, Total Positive = {positive_total}")

                episodes_data.append({
                    "Episode": episode + 1,
                    "Cumulative_Reward": episode_reward,
                    "Average_Reward": avg_reward,
                    "Steps": episode_steps,
                    "Remaining_Lives": lives,
                    "Berries_Eaten": berries,
                    "Average_Loss": avg_loss,
                    "Total_Penalty": penalty_total,
                    "Total_Positive": positive_total
                })

                writer.add_scalar("Reward/Cumulative", episode_reward, episode)
                writer.add_scalar("Reward/Average", avg_reward, episode)
                writer.add_scalar("Env/Avg_Loss", avg_loss, episode)
                writer.add_scalar("Env/Remaining_Lives", lives, episode)
                writer.add_scalar("Env/Berries_Eaten", berries, episode)
                writer.add_scalar("Env/Steps", episode_steps, episode)
                writer.add_scalar("Reward/Total_Penalty", penalty_total, episode)
                writer.add_scalar("Reward/Total_Positive", positive_total, episode)

                with torch.no_grad():
                    q_values = self.agent.model(torch.FloatTensor(validation_states).to(device))
                    avg_q = q_values.mean().item()
                writer.add_scalar("Validation/Avg_Q", avg_q, episode)

        except KeyboardInterrupt:
            print("\nTraining interrotto manualmente. Salvataggio dei dati...")
        finally:
            writer.close()

            results_df = pd.DataFrame(episodes_data)
            results_file = os.path.join(save_dir, f"training_results({training_name}).xlsx")
            results_df.to_excel(results_file, index=False)
            print(f"Risultati salvati in {results_file}")

            if not results_df.empty:
                def plot_line(y, label, color):
                    plt.figure(figsize=(10, 6))
                    plt.plot(results_df["Episode"], results_df[y], marker='o', linestyle='-', color=color)
                    plt.title(f'{label} per Episode')
                    plt.xlabel('Episode')
                    plt.ylabel(label)
                    plt.grid(True)
                    plt.savefig(os.path.join(save_dir, f"training_{y}.png"))
                    plt.close()

                plot_line("Cumulative_Reward", "Cumulative Reward", "blue")
                plot_line("Average_Reward", "Average Reward", "purple")
                plot_line("Remaining_Lives", "Remaining Lives", "orange")
                plot_line("Berries_Eaten", "Berries Eaten", "green")
                plot_line("Average_Loss", "Average Loss", "red")
                plot_line("Total_Penalty", "Total Penalty", "darkred")
                plot_line("Total_Positive", "Total Positive", "darkred")
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
        #model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_8.1_con_noisy\Training_Pacman_8.1_con_noisy.pth"
        main_obj = Main(screen)
        training_name = "Training_Pacman_1_con_4fantasmi_noisy"
        main_obj.simulate_training(episodes=100, training_name=training_name)
    elif mode == "testing":
        model_path = r"C:\Users\claud\Desktop\IA\ModelliSalvati\Training_Pacman_8_con_noisy\Training_Pacman_8_con_noisy.pth"
        main_obj = Main(screen, model_path=model_path)
        main_obj.simulate_testing(episodes=10)
    elif mode == "game":
        main_obj = Main(screen)
        main_obj.main()
