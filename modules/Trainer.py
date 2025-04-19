from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import time
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from modules.ChessEnv import ChessEnv
from modules.Agent import ChessAgent, DQNAgent
from modules.Move import Move
from modules.ChessBoard import PieceColor


class Trainer:
	"""
	Classe pour entraîner et évaluer des agents d'échecs.
	Prend en charge l'entraînement en autojeu et l'apprentissage par renforcement.
	"""
	
	def __init__(self, 
				 checkpointDir: str = 'checkpoints',
				 logDir: str = 'logs',
				 headless: bool = True):
		"""
		Initialise le Trainer.
		
		Args:
			checkpointDir: Répertoire pour les points de contrôle du modèle
			logDir: Répertoire pour les logs d'entraînement
			headless: Mode sans affichage graphique
		"""
		self.checkpointDir = checkpointDir
		self.logDir = logDir
		self.headless = headless
		
		# Créer les répertoires si nécessaires
		os.makedirs(checkpointDir, exist_ok=True)
		os.makedirs(logDir, exist_ok=True)
		
		# Statistiques d'entraînement
		self.stats = {
			'episodes': [],
			'rewards': [],
			'policy_losses': [],
			'value_losses': [],
			'total_losses': [],
			'episode_lengths': [],
			'win_rates': [],
			'draw_rates': [],
			'checkmates': []
		}
	
	def trainAlphaZero(self, 
					   agent: ChessAgent,
					   numEpisodes: int = 1000,
					   maxStepsPerEpisode: int = 200,
					   evaluationFrequency: int = 50,
					   evaluationEpisodes: int = 10,
					   checkpointFrequency: int = 100,
					   renderFrequency: int = 0,
					   selfPlayBatchSize: int = 32) -> Dict:
		"""
		Entraîne un agent selon l'algorithme AlphaZero.
		
		Args:
			agent: Agent à entraîner
			numEpisodes: Nombre total d'épisodes d'entraînement
			maxStepsPerEpisode: Nombre maximum d'étapes par épisode
			evaluationFrequency: Fréquence d'évaluation (en épisodes)
			evaluationEpisodes: Nombre d'épisodes pour chaque évaluation
			checkpointFrequency: Fréquence de sauvegarde du modèle
			renderFrequency: Fréquence de rendu visuel (0 pour désactiver)
			selfPlayBatchSize: Taille du lot pour l'autojeu
			
		Returns:
			Dict: Statistiques d'entraînement
		"""
		# Configuration de l'environnement
		renderMode = None if self.headless else 'human'
		env = ChessEnv(render_mode=renderMode)
		
		# Environnement pour l'évaluation
		evalEnv = ChessEnv(render_mode=None)
		
		# Température initiale et son taux de décroissance
		temperature = 1.0
		temperatureDecay = 0.92
		minTemperature = 0.1
		
		# Barre de progression
		pbar = tqdm(range(1, numEpisodes + 1), desc="Entraînement")
		
		# Données d'autojeu
		selfPlayData = []
		
		for episode in pbar:
			# Démarrer un nouvel épisode
			observation = env.reset()
			done = False
			truncated = False
			totalReward = 0
			step = 0
			episodeData = []
			
			# Décroissance de la température
			if episode % 50 == 0 and temperature > minTemperature:
				temperature *= temperatureDecay
				temperature = max(temperature, minTemperature)
			
			# Boucle d'épisode
			while not done and not truncated and step < maxStepsPerEpisode:
				# Obtenir l'observation du point de vue du joueur actuel
				selfPlayObs = env.getSelfPlayObservation()
				boardState = selfPlayObs['board_state']
				
				# Obtenir les actions légales
				legalActions = env.getLegalMovesAsActions()
				
				if not legalActions:
					break  # Fin de partie
				
				# Sélectionner une action
				action = agent.selectAction(selfPlayObs, temperature, legalActions)
				
				# Stocker l'état actuel
				currentState = np.copy(boardState)
				
				# Exécuter l'action
				nextObs, reward, done, truncated, info = env.step(action)
				totalReward += reward
				step += 1
				
				# Stocker les données pour l'apprentissage
				# Utiliser le résultat final comme valeur cible (1 pour victoire, -1 pour défaite, 0 pour nul)
				# La politique MCTS sera calculée pendant l'autojeu
				if done:
					value = reward  # Utiliser la récompense finale comme valeur
				else:
					value = 0  # Valeur neutre pour les états non terminaux
				
				# Convertir l'action en politique one-hot simplifiée
				policy = np.zeros(8*8*8*8)
				policy[action] = 1.0
				
				# Stocker les données de l'étape
				episodeData.append((
					ChessAgent._prepareInput(currentState),
					policy,
					value
				))
				
				# Rendu si nécessaire
				if renderFrequency > 0 and episode % renderFrequency == 0:
					env.render()
					time.sleep(0.1)  # Pause pour voir le rendu
			
			# Fin de l'épisode
			episodeLength = step
			
			# Inversion des valeurs en fonction de la parité des étapes pour l'alternance des joueurs
			for i in range(len(episodeData)):
				state, policy, value = episodeData[i]
				# Si indice impair, inverser la valeur (point de vue joueur opposé)
				if i % 2 == 1:
					episodeData[i] = (state, policy, -value)
			
			# Ajouter les données de l'épisode au lot d'autojeu
			selfPlayData.extend(episodeData)
			
			# Entraîner le modèle lorsque suffisamment de données sont collectées
			if len(selfPlayData) >= selfPlayBatchSize:
				# Échantillonner aléatoirement un lot
				batchIndices = np.random.choice(len(selfPlayData), selfPlayBatchSize, replace=False)
				batch = [selfPlayData[i] for i in batchIndices]
				
				# Entraîner l'agent
				trainStats = agent.train(batch)
				
				# Mettre à jour les statistiques
				if episode not in self.stats['episodes']:
					self.stats['episodes'].append(episode)
					self.stats['rewards'].append(totalReward)
					self.stats['episode_lengths'].append(episodeLength)
					self.stats['policy_losses'].append(trainStats['policy_loss'])
					self.stats['value_losses'].append(trainStats['value_loss'])
					self.stats['total_losses'].append(trainStats['total_loss'])
				
				# Limiter la taille des données d'autojeu pour éviter la suradaptation
				if len(selfPlayData) > selfPlayBatchSize * 10:
					# Conserver les données les plus récentes
					selfPlayData = selfPlayData[-selfPlayBatchSize * 10:]
			
			# Mise à jour de la barre de progression
			pbar.set_postfix({
				'reward': totalReward,
				'steps': episodeLength,
				'temp': f"{temperature:.2f}"
			})
			
			# Évaluation périodique
			if evaluationFrequency > 0 and episode % evaluationFrequency == 0:
				evalStats = self.evaluateAgent(agent, evalEnv, evaluationEpisodes)
				self.stats['win_rates'].append(evalStats['win_rate'])
				self.stats['draw_rates'].append(evalStats['draw_rate'])
				self.stats['checkmates'].append(evalStats['checkmate_rate'])
				
				print(f"\nÉvaluation à l'épisode {episode}:")
				print(f"Taux de victoire: {evalStats['win_rate']:.2%}")
				print(f"Taux de nul: {evalStats['draw_rate']:.2%}")
				print(f"Taux d'échec et mat: {evalStats['checkmate_rate']:.2%}")
			
			# Sauvegarde périodique du modèle
			if checkpointFrequency > 0 and episode % checkpointFrequency == 0:
				checkpointPath = os.path.join(self.checkpointDir, f"model_episode_{episode}.pt")
				agent.saveModel(checkpointPath)
				
				# Sauvegarder également les statistiques
				self.saveStats()
		
		# Sauvegarde finale
		finalCheckpointPath = os.path.join(self.checkpointDir, "model_final.pt")
		agent.saveModel(finalCheckpointPath)
		self.saveStats()
		
		# Tracer les graphiques
		self.plotTrainingStats()
		
		return self.stats
	
	def trainDQN(self, 
				agent: DQNAgent,
				numEpisodes: int = 1000,
				maxStepsPerEpisode: int = 200,
				targetUpdateFrequency: int = 10,
				evaluationFrequency: int = 50,
				evaluationEpisodes: int = 10,
				checkpointFrequency: int = 100,
				renderFrequency: int = 0) -> Dict:
		"""
		Entraîne un agent selon l'algorithme DQN.
		
		Args:
			agent: Agent à entraîner
			numEpisodes: Nombre total d'épisodes d'entraînement
			maxStepsPerEpisode: Nombre maximum d'étapes par épisode
			targetUpdateFrequency: Fréquence de mise à jour du réseau cible
			evaluationFrequency: Fréquence d'évaluation (en épisodes)
			evaluationEpisodes: Nombre d'épisodes pour chaque évaluation
			checkpointFrequency: Fréquence de sauvegarde du modèle
			renderFrequency: Fréquence de rendu visuel (0 pour désactiver)
			
		Returns:
			Dict: Statistiques d'entraînement
		"""
		# Configuration de l'environnement
		renderMode = None if self.headless else 'human'
		env = ChessEnv(render_mode=renderMode)
		
		# Environnement pour l'évaluation
		evalEnv = ChessEnv(render_mode=None)
		
		# Barre de progression
		pbar = tqdm(range(1, numEpisodes + 1), desc="Entraînement DQN")
		
		for episode in pbar:
			# Démarrer un nouvel épisode
			observation = env.reset()
			state = observation['board_state']
			done = False
			truncated = False
			totalReward = 0
			step = 0
			
			# Boucle d'épisode
			while not done and not truncated and step < maxStepsPerEpisode:
				# Obtenir les actions légales
				legalActions = env.getLegalMovesAsActions()
				
				if not legalActions:
					break  # Fin de partie
				
				# Sélectionner une action
				action = agent.selectAction(observation, legalActions)
				
				# Exécuter l'action
				nextObs, reward, done, truncated, info = env.step(action)
				nextState = nextObs['board_state']
				totalReward += reward
				step += 1
				
				# Stocker l'expérience
				agent.remember(
					DQNAgent.prepareInput(state),
					action,
					reward,
					DQNAgent.prepareInput(nextState),
					done
				)
				
				# Apprentissage
				learnStats = agent.learn()
				
				# Mise à jour de l'état actuel
				state = nextState
				
				# Rendu si nécessaire
				if renderFrequency > 0 and episode % renderFrequency == 0:
					env.render()
					time.sleep(0.1)  # Pause pour voir le rendu
			
			# Fin de l'épisode
			episodeLength = step
			
			# Mise à jour du réseau cible
			if episode % targetUpdateFrequency == 0:
				agent.updateTargetNetwork()
			
			# Mise à jour des statistiques
			self.stats['episodes'].append(episode)
			self.stats['rewards'].append(totalReward)
			self.stats['episode_lengths'].append(episodeLength)
			
			if learnStats:
				self.stats['total_losses'].append(learnStats['loss'])
			
			# Mise à jour de la barre de progression
			pbar.set_postfix({
				'reward': totalReward,
				'steps': episodeLength,
				'epsilon': f"{agent.epsilon:.2f}"
			})
			
			# Évaluation périodique
			if evaluationFrequency > 0 and episode % evaluationFrequency == 0:
				evalStats = self.evaluateAgent(agent, evalEnv, evaluationEpisodes)
				self.stats['win_rates'].append(evalStats['win_rate'])
				self.stats['draw_rates'].append(evalStats['draw_rate'])
				self.stats['checkmates'].append(evalStats['checkmate_rate'])
				
				print(f"\nÉvaluation à l'épisode {episode}:")
				print(f"Taux de victoire: {evalStats['win_rate']:.2%}")
				print(f"Taux de nul: {evalStats['draw_rate']:.2%}")
				print(f"Taux d'échec et mat: {evalStats['checkmate_rate']:.2%}")
			
			# Sauvegarde périodique du modèle
			if checkpointFrequency > 0 and episode % checkpointFrequency == 0:
				checkpointPath = os.path.join(self.checkpointDir, f"dqn_model_episode_{episode}.pt")
				agent.saveModel(checkpointPath)
				
				# Sauvegarder également les statistiques
				self.saveStats()
		
		# Sauvegarde finale
		finalCheckpointPath = os.path.join(self.checkpointDir, "dqn_model_final.pt")
		agent.saveModel(finalCheckpointPath)
		self.saveStats()
		
		# Tracer les graphiques
		self.plotTrainingStats()
		
		return self.stats
	
	def evaluateAgent(self, agent: Union[ChessAgent, DQNAgent], env: ChessEnv, 
					  numEpisodes: int = 10) -> Dict:
		"""
		Évalue les performances d'un agent.
		
		Args:
			agent: Agent à évaluer
			env: Environnement d'évaluation
			numEpisodes: Nombre d'épisodes d'évaluation
			
		Returns:
			Dict: Statistiques d'évaluation
		"""
		wins = 0
		draws = 0
		checkmates = 0
		
		for _ in range(numEpisodes):
			observation = env.reset()
			done = False
			truncated = False
			
			# Désactiver l'exploration pour DQN pendant l'évaluation
			original_epsilon = None
			if isinstance(agent, DQNAgent):
				original_epsilon = agent.epsilon
				agent.epsilon = 0.0
			
			while not done and not truncated:
				# Obtenir les actions légales
				legalActions = env.getLegalMovesAsActions()
				
				if not legalActions:
					break
				
				# Sélectionner une action
				if isinstance(agent, ChessAgent):
					# Utiliser une température basse pour favoriser les meilleurs coups
					action = agent.selectAction(observation, 0.1, legalActions)
				else:
					action = agent.selectAction(observation, legalActions)
				
				# Exécuter l'action
				observation, reward, done, truncated, info = env.step(action)
				
				# Vérifier les conditions de fin
				if done:
					if reward > 0:
						wins += 1
					elif reward == 0:
						draws += 1
					
					if info.get('outcome') == 'checkmate':
						checkmates += 1
			
			# Restaurer l'epsilon pour DQN
			if original_epsilon is not None:
				agent.epsilon = original_epsilon
		
		# Calculer les taux
		win_rate = wins / numEpisodes
		draw_rate = draws / numEpisodes
		checkmate_rate = checkmates / numEpisodes
		
		return {
			'win_rate': win_rate,
			'draw_rate': draw_rate,
			'checkmate_rate': checkmate_rate,
			'episodes': numEpisodes
		}
	
	def saveStats(self) -> None:
		"""Sauvegarde les statistiques d'entraînement."""
		statsPath = os.path.join(self.logDir, "training_stats.json")
		
		# Convertir les données numpy en types Python standard
		saveStats = {}
		for key, value in self.stats.items():
			if isinstance(value, list) and len(value) > 0:
				if isinstance(value[0], (np.integer, np.floating)):
					saveStats[key] = [float(v) for v in value]
				else:
					saveStats[key] = value
			else:
				saveStats[key] = value
		
		with open(statsPath, 'w') as f:
			json.dump(saveStats, f, indent=4)
	
	def plotTrainingStats(self) -> None:
		"""Trace et sauvegarde des graphiques des statistiques d'entraînement."""
		if len(self.stats['episodes']) == 0:
			return
		
		# Créer un dossier pour les graphiques
		plotsDir = os.path.join(self.logDir, "plots")
		os.makedirs(plotsDir, exist_ok=True)
		
		# Graphique des récompenses
		plt.figure(figsize=(10, 6))
		plt.plot(self.stats['episodes'], self.stats['rewards'])
		plt.title('Récompenses par épisode')
		plt.xlabel('Épisode')
		plt.ylabel('Récompense totale')
		plt.grid(True)
		plt.savefig(os.path.join(plotsDir, "rewards.png"))
		plt.close()
		
		# Graphique de la longueur des épisodes
		plt.figure(figsize=(10, 6))
		plt.plot(self.stats['episodes'], self.stats['episode_lengths'])
		plt.title('Longueur des épisodes')
		plt.xlabel('Épisode')
		plt.ylabel('Nombre de pas')
		plt.grid(True)
		plt.savefig(os.path.join(plotsDir, "episode_lengths.png"))
		plt.close()
		
		# Graphique des pertes (si disponibles)
		if self.stats['total_losses']:
			plt.figure(figsize=(10, 6))
			plt.plot(self.stats['episodes'][:len(self.stats['total_losses'])], 
					 self.stats['total_losses'], label='Perte totale')
			
			if self.stats['policy_losses']:
				plt.plot(self.stats['episodes'][:len(self.stats['policy_losses'])], 
						 self.stats['policy_losses'], label='Perte de politique')
			
			if self.stats['value_losses']:
				plt.plot(self.stats['episodes'][:len(self.stats['value_losses'])], 
						 self.stats['value_losses'], label='Perte de valeur')
			
			plt.title('Pertes d\'entraînement')
			plt.xlabel('Épisode')
			plt.ylabel('Perte')
			plt.legend()
			plt.grid(True)
			plt.savefig(os.path.join(plotsDir, "losses.png"))
			plt.close()
		
		# Graphique des taux de victoire, nul et échec et mat (si disponibles)
		if self.stats['win_rates']:
			evalEpisodes = [self.stats['episodes'][i] for i in range(0, len(self.stats['episodes']), 
																	len(self.stats['episodes'])//len(self.stats['win_rates']))][:len(self.stats['win_rates'])]
			
			plt.figure(figsize=(10, 6))
			plt.plot(evalEpisodes, self.stats['win_rates'], label='Taux de victoire')
			plt.plot(evalEpisodes, self.stats['draw_rates'], label='Taux de nul')
			plt.plot(evalEpisodes, self.stats['checkmates'], label='Taux d\'échec et mat')
			plt.title('Performance de l\'agent')
			plt.xlabel('Épisode')
			plt.ylabel('Taux')
			plt.ylim(0, 1)
			plt.legend()
			plt.grid(True)
			plt.savefig(os.path.join(plotsDir, "performance.png"))
			plt.close()
	
	def loadStats(self) -> None:
		"""Charge les statistiques d'entraînement."""
		statsPath = os.path.join(self.logDir, "training_stats.json")
		
		if os.path.exists(statsPath):
			with open(statsPath, 'r') as f:
				self.stats = json.load(f)
	
	def playAgainstHuman(self, agent: Union[ChessAgent, DQNAgent], 
						 playerColor: PieceColor = PieceColor.WHITE) -> None:
		"""
		Permet à un humain de jouer contre l'agent.
		
		Args:
			agent: Agent contre lequel jouer
			playerColor: Couleur du joueur humain
		"""
		# Utiliser le mode de rendu humain
		env = ChessEnv(render_mode='human')
		observation = env.reset()
		done = False
		
		print("\n=== JEU D'ÉCHECS CONTRE L'IA ===")
		print(f"Vous jouez les {'blancs' if playerColor == PieceColor.WHITE else 'noirs'}")
		print("Entrez vos coups en notation algébrique (ex: e2e4)")
		print("Tapez 'exit' pour quitter\n")
		
		# Désactiver l'exploration pour DQN
		original_epsilon = None
		if isinstance(agent, DQNAgent):
			original_epsilon = agent.epsilon
			agent.epsilon = 0.0
		
		while not done:
			# Afficher le plateau
			env.render()
			
			currentPlayer = env.board.currentPlayer
			
			# Tour du joueur humain
			if currentPlayer == playerColor:
				valid_move = False
				while not valid_move:
					move_str = input("\nVotre coup: ")
					
					if move_str.lower() == 'exit':
						done = True
						break
					
					try:
						# Convertir la notation algébrique en Move
						move = Move.fromAlgebraic(move_str)
						
						# Vérifier si le coup est légal
						if env.board.isLegalMove(move):
							observation, reward, done, truncated, info = env.step(move)
							valid_move = True
							print(f"Coup joué: {move_str}")
						else:
							print("Coup illégal. Réessayez.")
					except ValueError:
						print("Format invalide. Utilisez la notation algébrique (ex: e2e4).")
			
			# Tour de l'IA
			else:
				print("\nL'IA réfléchit...")
				
				# Obtenir les actions légales
				legalActions = env.getLegalMovesAsActions()
				
				# Sélectionner une action
				if isinstance(agent, ChessAgent):
					action = agent.selectAction(observation, 0.1, legalActions)
				else:
					action = agent.selectAction(observation, legalActions)
				
				# Convertir l'indice d'action en Move
				move = env._actionToMove(action)
				
				# Exécuter l'action
				observation, reward, done, truncated, info = env.step(move)
				
				print(f"L'IA joue: {move.toAlgebraic()}")
			
			# Vérifier les conditions de fin
			if done:
				outcome = info.get('outcome', 'unknown')
				if outcome == 'checkmate':
					winner = 'blancs' if currentPlayer == PieceColor.BLACK else 'noirs'
					print(f"\nÉchec et mat! Les {winner} gagnent.")
				elif outcome == 'draw':
					print("\nPartie nulle!")
				elif outcome == 'max_steps_reached':
					print("\nNombre maximum de coups atteint. Partie déclarée nulle.")
				
				env.render()
		
		# Fermer l'environnement
		env.close()
		
		# Restaurer l'epsilon pour DQN
		if original_epsilon is not None:
			agent.epsilon = original_epsilon