import pygame, sys

from reinforcementDQL import DQNAgent
from settings import WIDTH, HEIGHT, NAV_HEIGHT
from worldDQL import World
import numpy as np


pygame.init()

# Imposta la finestra del gioco
screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
pygame.display.set_caption("PacMan")

class Main:
	def __init__(self, screen):
		self.screen = screen
		self.FPS = pygame.time.Clock()
		self.agent = DQNAgent(state_size=6, action_size=4)

	def main(self):
		# Avvia la modalità manuale
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

		for episode in range(episodes):
			print(f"Training Episode {episode + 1}/{episodes}")

			world = World(self.screen)
			state = world.get_current_state()
			state = np.reshape(state, [1, self.agent.state_size])

			while not world.game_over:
				action = self.agent.act(state)
				world.apply_action(action)
				next_state = world.get_current_state()
				next_state = np.reshape(next_state, [1, self.agent.state_size])

				reward = world.get_reward()
				done = world.game_over
				self.agent.remember(state, action, reward, next_state, done)
				self.agent.replay()

				state = next_state

				world.updateRL()
				pygame.display.update()  # Aggiunto per evitare schermata nera
				self.FPS.tick(30)

			reward_all_episode += world.total_reward
			print(f"Episode {episode} completed. Total Reward: {world.total_reward}. Cumulative: {reward_all_episode}")

	#salvataggio q-table pkl
		#reinforcement.save_q_table()
		#salvataggio q-table JSON
		#reinforcementDQL.save_q_table_json()


# Blocco principale
if __name__ == "__main__":
	play = Main(screen)

	# Modalità: scegli tra "training" o "game"
	mode = "training"

	if mode == "training":
		play.simulate_training(episodes=10)
	elif mode == "game":
		play.main()
