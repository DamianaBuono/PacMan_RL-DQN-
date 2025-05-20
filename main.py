import pygame
import sys
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from settings import WIDTH, HEIGHT, NAV_HEIGHT, MAX_STEPS
from worldDQN import World
from reinforcementDQN import DQNAgent  # use the DQN agent

BASE_MODEL_DIR = r"C:\Users\claud\Desktop\IA\ModelliDQN"
BASE_TESTING_DIR = r"C:\Users\claud\Desktop\IA\TestDQN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Main:
    def __init__(self, screen, model_path=None):
        pygame.init()
        self.screen = screen
        self.FPS = pygame.time.Clock()

        # Create DQN agent
        self.agent = DQNAgent(state_size=14, action_size=4)
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"DQN model loaded from {model_path}")
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
                while not done and episode_steps < MAX_STEPS:
                    action = self.agent.act(state)
                    next_state, reward, done = world.step(action)

                    # store and train
                    self.agent.remember(state, action, reward, next_state, done)
                    l = self.agent.replay()
                    if l is not None:
                        episode_losses.append(l)

                    state = next_state
                    episode_reward += reward
                    #episode_losses.append(self.agent.last_loss)
                    episode_steps += 1

                    # render if desired
                    world.render()
                    pygame.display.update()
                    self.FPS.tick(30)

                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
                lives = world.player.sprite.life
                berries = world.player.sprite.n_bacche
                penalty_total = world.total_penalty
                positive_total = world.total_positive
                level_reached = world.game_level

                episodes_data.append({
                    "Episode": ep + 1,
                    "TotalReward": episode_reward,
                    "AverageReward": avg_reward,
                    "Steps": episode_steps,
                    "RemainingLives": lives,
                    "Berries": berries,
                    "AverageLoss": avg_loss,
                    "TotalPenalty": penalty_total,
                    "TotalPositive": positive_total,
                    "Level": level_reached
                })
                # world.restart_level()
                window = 1000
                start_idx = max(0, ep - window + 1)
                recent = episodes_data[start_idx:ep + 1]

                def moving_avg(key):
                    return np.mean([ep[key] for ep in recent])

                if ep == 0 or (ep + 1) % 100 == 0:
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

                # validation Q-value
                self.agent.model.eval()  # modalità eval per inference
                with torch.no_grad():
                    # validation_states è un array NumPy di shape (N, state_size)
                    val_tensor = torch.FloatTensor(validation_states).to(device)
                    qs = self.agent.model(val_tensor)  # shape: (N, action_size)
                    max_qs, _ = qs.max(dim=1, keepdim=False)  # shape: (N,)
                    avg_q = max_qs.mean().item()  # scalare

                writer.add_scalar("Validation/Avg_Q", avg_q, ep)
                self.agent.model.train()  # torna in modalità train

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        finally:
            writer.close()
            model_path = os.path.join(save_dir, "dqn_model.pth")
            self.agent.save(model_path)
            print(f"Model saved to {model_path}")

    def simulate_testing(self, episodes):
        self.agent.epsilon = 0.0
        test_log_dir = os.path.join(BASE_TESTING_DIR, "test_logs")
        os.makedirs(test_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=test_log_dir)

        all_rewards, all_steps = [], []
        for ep in range(episodes):
            world = World(self.screen, agent=self.agent)
            state = world.reset()
            done = False
            total_reward, steps = 0, 0
            while not done and steps < MAX_STEPS:
                action = self.agent.act(state)
                state, reward, done = world.step(action)
                total_reward += reward
                steps += 1
            all_rewards.append(total_reward)
            all_steps.append(steps)
            print(f"[TEST] Ep {ep+1}: Reward={total_reward}, Steps={steps}")
            writer.add_scalar("Test/Reward", total_reward, ep)
            writer.add_scalar("Test/Steps", steps, ep)

        print(f"Average Reward: {np.mean(all_rewards):.2f}, Average Steps: {np.mean(all_steps):.2f}")
        writer.add_scalar("Test/Avg_Reward_All", np.mean(all_rewards), episodes)
        writer.add_scalar("Test/Avg_Steps_All", np.mean(all_steps), episodes)
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
    mode = "training"  # or "testing" or "game"
    screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
    pygame.display.set_caption("PacMan")

    if mode == "training":
        model_path = os.path.join(BASE_MODEL_DIR, "DQN_PacMan", "dqn_model.pth")
        main_obj = Main(screen, model_path=model_path)
    elif mode == "testing":
        model_path = os.path.join(BASE_TESTING_DIR, "DQN_PacMan", "dqn_model.pth")
        main_obj = Main(screen, model_path=model_path)
        main_obj.simulate_testing(episodes=10)
    else:
        main_obj = Main(screen)
        main_obj.main()