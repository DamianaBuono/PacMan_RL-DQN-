import pygame, sys

import reinforcement
from settings import WIDTH, HEIGHT, NAV_HEIGHT
from world import World


pygame.init()

# Imposta la finestra del gioco
screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))
pygame.display.set_caption("PacMan")

class Main:
	def __init__(self, screen):
		self.screen = screen
		self.FPS = pygame.time.Clock()

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

		#Modalità di allenamento per l'agente RL.
		reward_all_episode=0
		for episode in range(episodes):
			print(f"episodio:",episode)
			# Crea un nuovo mondo per ogni episodio
			world = World(self.screen)

			# Esegui il ciclo del gioco fino a quando non è terminato
			while not world.game_over:
				world.updateRL()
				pygame.display.update()
				self.FPS.tick(30)

			reward_all_episode+=world.total_reward
			# Stampa il risultato dell'episodio
			print(f"Episode {episode} completed. Score: {world.player.sprite.pac_score}. Total_Reward {world.total_reward}. Reward_all_episode: {reward_all_episode}")


		reinforcement.save_q_table()


# Blocco principale
if __name__ == "__main__":
	play = Main(screen)

	# Modalità: scegli tra "training" o "game"
	mode = "training"

	if mode == "training":
		play.simulate_training(episodes=10)
	elif mode == "game":
		play.main()
