import pygame
import sys
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from settings import WIDTH, HEIGHT, NAV_HEIGHT, MAX_STEPS
from world import World
from reinforcement import  SarsaAgent

BASE_MODEL_DIR = r"C:\Users\mucci\Desktop\IA\ModelliSarsa"
BASE_TESTING_DIR = r"C:\Users\mucci\Desktop\IA\TestSarsa"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Main:
    def __init__(self, screen, model_path=None):
        pygame.init()
        self.screen = screen
        self.FPS = pygame.time.Clock()


        self.agent = SarsaAgent(action_size=4)
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"SARSA model loaded from {model_path}")
        else:
            print("No pre-trained model found, starting fresh.")

    def simulate_training(self, episodes, training_name):
        save_dir = os.path.join(BASE_MODEL_DIR, training_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"All files will be saved in: {save_dir}")

        writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))
        episodes_data = []

        # Prepare validation states
        num_val_states = 50
        validation_states = []
        val_world = World(self.screen, agent=self.agent)
        val_world.agent.epsilon = 0.0
        for _ in range(num_val_states):
            validation_states.append(val_world.reset())
        validation_states = np.stack(validation_states)

        try:
            for ep in range(episodes):
                world = World(self.screen, agent=self.agent)
                state = world.reset()

                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                done = False
                action = self.agent.act(state)  # prima azione scelta
                while not done and episode_steps < MAX_STEPS:
                    next_state, reward, done = world.step(action)
                    next_action = self.agent.act(next_state)  # azione successiva per SARSA

                    self.agent.remember(state, action, reward, next_state, next_action, done)

                    state = next_state
                    action = next_action  # shift per SARSA

                    episode_reward += reward
                    episode_losses.append(self.agent.last_loss)
                    episode_steps += 1

                    # render if desired
                    world.render()
                    pygame.display.update()
                    self.FPS.tick(30)

                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
                lives = world.player.sprite.life
                berries = world.player.sprite.n_bacche
                penalty_per_step = world.total_penalty / episode_steps if episode_steps > 0 else 0.0
                positive_per_step = world.total_positive / episode_steps

                level_reached = world.game_level

                episodes_data.append({
                    "Episode": ep + 1,
                    "TotalReward": episode_reward,
                    "AverageReward": avg_reward,
                    "Steps": episode_steps,
                    "RemainingLives": lives,
                    "Berries": berries,
                    "AverageLoss": avg_loss,
                    "TotalPenalty": penalty_per_step,
                    "TotalPositive": positive_per_step,
                    "Level": level_reached
                })
                window = 1000
                start_idx = max(0, ep - window + 1)
                recent = episodes_data[start_idx:ep + 1]

                def moving_avg(key):
                    return np.mean([ep[key] for ep in recent])

                if ep == 0 or (ep + 1) % 1000 == 0:
                    print(f"""[Ep {ep + 1}] Moving Averages over last {len(recent)} episodes:
                                TotalReward     : {moving_avg('TotalReward'):.4f}
                                AverageReward   : {moving_avg('AverageReward'):.4f}
                                AverageLoss     : {moving_avg('AverageLoss'):.4f}
                                Steps           : {moving_avg('Steps'):.2f}
                                RemainingLives  : {moving_avg('RemainingLives'):.2f}
                                Berries         : {moving_avg('Berries'):.2f}
                                TotalPenalty    : {moving_avg('TotalPenalty'):.2f}
                                TotalPositive   : {moving_avg('TotalPositive'):.2f}
                                Level           : {moving_avg('Level'):.2f}""")

                writer.add_scalar("Reward/Cumulative", moving_avg("TotalReward"), ep)
                writer.add_scalar("Reward/Average", moving_avg("AverageReward"), ep)
                writer.add_scalar("Env/Avg_Loss", moving_avg("AverageLoss"), ep)
                writer.add_scalar("Env/Remaining_Lives", moving_avg("RemainingLives"), ep)
                writer.add_scalar("Env/Berries_Eaten", moving_avg("Berries"), ep)
                writer.add_scalar("Env/Steps", moving_avg("Steps"), ep)
                writer.add_scalar("Reward/Total_Penalty", moving_avg("TotalPenalty"), ep)
                writer.add_scalar("Reward/Total_Positive", moving_avg("TotalPositive"), ep)
                writer.add_scalar("Env/Level", moving_avg("Level"), ep)

                # Calcolo della media dei valori Q massimi sugli stati di validazione
                total_q = 0
                count = 0
                for val_state in validation_states:
                    state_discrete = self.agent._discretize_state(val_state)
                    q_vals = self.agent.q_table.get(state_discrete, np.zeros(self.agent.action_size))
                    total_q += np.max(q_vals)
                    count += 1

                avg_q = total_q / count if count > 0 else 0
                writer.add_scalar("Validation/Avg_Q", avg_q, ep)

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        finally:
            writer.close()
            model_path = os.path.join(save_dir, "sarsa_model.pkl")
            self.agent.save(model_path)
            print(f"Model saved to {model_path}")

    def simulate_testing(self, episodes, test_name):
        self.agent.epsilon = 0.0  # disattiva esplorazione
        test_dir = os.path.join(BASE_TESTING_DIR, test_name)
        os.makedirs(test_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(test_dir, "tensorboard_logs"))

        print(f"[TEST] Risultati saranno salvati in: {test_dir}")

        test_data = []
        all_rewards = []
        all_steps = []
        all_lives = []
        all_berries = []
        all_levels = []
        win_count = 0

        for ep in range(episodes):
            world = World(self.screen, agent=self.agent)
            state = world.reset()

            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < MAX_STEPS:
                action = self.agent.act(state)
                next_state, reward, done = world.step(action)
                state = next_state

                total_reward += reward
                steps += 1

                world.render()
                pygame.display.update()
                self.FPS.tick(30)

            lives = world.player.sprite.life
            berries = world.player.sprite.n_bacche
            level = world.game_level
            win = int(world.episode_won)

            all_rewards.append(total_reward)
            all_steps.append(steps)
            all_lives.append(lives)
            all_berries.append(berries)
            all_levels.append(level)
            win_count += win

            test_data.append({
                "Episode": ep + 1,
                "TotalReward": total_reward,
                "Steps": steps,
                "RemainingLives": lives,
                "Berries": berries,
                "Level": level,
                "Win": win
            })

        avg_reward = np.mean(all_rewards)
        avg_steps = np.mean(all_steps)
        print(" STEP: ", avg_steps)
        avg_lives = np.mean(all_lives)
        avg_berries = np.mean(all_berries)
        avg_level = np.mean(all_levels)
        win_rate = win_count / episodes
        print(" win_count: ", win_count)

        print(f"\n--- RISULTATI FINALI TEST ({episodes} episodi) ---")
        print(f"Avg Reward     : {avg_reward:.2f}")
        print(f"Avg Steps      : {avg_steps:.2f}")
        print(f"Avg Lives      : {avg_lives:.2f}")
        print(f"Avg Berries    : {avg_berries:.2f}")
        print(f"Avg Level      : {avg_level:.2f}")
        print(f"Win Rate       : {win_rate:.2%}")

        writer.add_scalar("Test/Avg_Reward_All", avg_reward, episodes)
        writer.add_scalar("Test/Avg_Steps_All", avg_steps, episodes)
        writer.add_scalar("Test/Avg_Lives_All", avg_lives, episodes)
        writer.add_scalar("Test/Avg_Berries_All", avg_berries, episodes)
        writer.add_scalar("Test/Avg_Level_All", avg_level, episodes)
        writer.add_scalar("Test/Win_Rate_All", win_rate, episodes)
        writer.close()

    def main(self):
        world = World(self.screen, agent=None)
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            world.update()
            pygame.display.update()
            self.FPS.tick(30)

if __name__ == "__main__":
    mode = "game"  # or "testing" or "game"
    screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
    pygame.display.set_caption("PacMan")

    if mode == "training":
        model_path = os.path.join(BASE_MODEL_DIR, "SARSA_PacMan2", "sarsa_model.pkl")
        main_obj = Main(screen, model_path=model_path)
        main_obj.simulate_training(episodes=300000, training_name="SARSA_PacMan8")
    elif mode == "testing":
        model_path = os.path.join(BASE_MODEL_DIR, "SARSA_PacMan2", "sarsa_model.pkl")
        main_obj = Main(screen, model_path=model_path)
        main_obj.simulate_testing(episodes=1000,test_name= "Test_SARSA_PacMan1" )
    elif mode == "game":
        main_obj = Main(screen)
        main_obj.main()