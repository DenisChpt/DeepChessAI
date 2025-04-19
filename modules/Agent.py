from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
import os

from modules.ChessBoard import ChessBoard, PieceColor
from modules.Move import Move
from modules.NeuralNet import ChessNet, QNetwork


class ReplayBuffer:
	"""
	Tampon de rejeu pour stocker les expériences et les échantillonner pour l'apprentissage.
	"""
	
	def __init__(self, capacity: int = 100000):
		"""
		Initialise le tampon de rejeu.
		
		Args:
			capacity: Capacité maximale du tampon
		"""
		self.buffer = deque(maxlen=capacity)
	
	def push(self, state: np.ndarray, action: int, reward: float, 
			 next_state: np.ndarray, done: bool) -> None:
		"""
		Ajoute une expérience au tampon.
		
		Args:
			state: État courant
			action: Action choisie
			reward: Récompense obtenue
			next_state: État suivant
			done: Indicateur de fin d'épisode
		"""
		self.buffer.append((state, action, reward, next_state, done))
	
	def sample(self, batchSize: int) -> Tuple:
		"""
		Échantillonne un lot d'expériences du tampon.
		
		Args:
			batchSize: Taille du lot à échantillonner
			
		Returns:
			Tuple: Lot d'expériences (états, actions, récompenses, états suivants, fins)
		"""
		indices = np.random.choice(len(self.buffer), batchSize, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
		
		return (
			np.array(states),
			np.array(actions),
			np.array(rewards, dtype=np.float32),
			np.array(next_states),
			np.array(dones, dtype=np.uint8)
		)
	
	def __len__(self) -> int:
		"""Retourne la taille actuelle du tampon."""
		return len(self.buffer)


class ChessAgent:
	"""
	Agent d'apprentissage par renforcement pour les échecs.
	Implémente une approche similaire à AlphaZero avec MCTS.
	"""
	
	def __init__(self, model: Optional[ChessNet] = None, device: str = 'auto'):
		"""
		Initialise l'agent d'échecs.
		
		Args:
			model: Modèle de réseau neuronal (si None, un nouveau modèle sera créé)
			device: Périphérique d'exécution ('cpu', 'cuda', ou 'auto')
		"""
		# Déterminer le périphérique
		if device == 'auto':
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)
		
		print(f"Utilisation du périphérique: {self.device}")
		
		# Initialiser ou charger le modèle
		if model is None:
			self.model = ChessNet().to(self.device)
		else:
			self.model = model.to(self.device)
		
		# Mémoire pour le MCTS
		self.mctsCache = {}
		
		# Paramètres de recherche
		self.numMctsSimulations = 100  # Nombre de simulations MCTS par coup
		self.cPuct = 1.0               # Coefficient d'exploration
		self.dirichletEpsilon = 0.25   # Paramètre de bruit Dirichlet
		self.dirichletAlpha = 0.3      # Paramètre de concentration Dirichlet
	
	def selectAction(self, state: Dict, temperature: float = 1.0, 
					 legalActions: List[int] = None) -> int:
		"""
		Sélectionne une action en utilisant MCTS et le réseau neuronal.
		
		Args:
			state: État du jeu
			temperature: Paramètre de température pour l'échantillonnage
			legalActions: Liste des actions légales
			
		Returns:
			int: Indice de l'action choisie
		"""
		boardState = state['board_state']
		
		# Préparer l'entrée pour le réseau
		inputState = ChessNet.prepareInput(boardState)
		
		# Si aucune action légale n'est fournie, utiliser toutes les actions
		if legalActions is None:
			legalActions = list(range(8*8*8*8))  # Espace d'action complet
		
		# Si la température est proche de zéro, choisir l'action la plus probable
		if temperature < 0.01:
			# Prédire la politique et la valeur
			policy, _ = self.model.predict(inputState, self.device)
			
			# Masquer les actions illégales
			maskedPolicy = np.zeros_like(policy)
			maskedPolicy[legalActions] = policy[legalActions]
			
			# Choisir l'action la plus probable
			action = np.argmax(maskedPolicy)
			
		else:
			# Exécuter MCTS pour améliorer la politique
			policy = self._runMCTS(boardState, legalActions)
			
			# Échantillonner selon la politique ajustée par la température
			policyTemp = policy ** (1 / temperature)
			policyTemp /= np.sum(policyTemp)
			action = np.random.choice(len(policy), p=policyTemp)
		
		return action
	
	def _runMCTS(self, boardState: np.ndarray, legalActions: List[int]) -> np.ndarray:
		"""
		Exécute l'algorithme MCTS (Monte Carlo Tree Search) pour améliorer la politique.
		
		Args:
			boardState: État du plateau
			legalActions: Liste des actions légales
			
		Returns:
			np.ndarray: Politique améliorée
		"""
		# Initialiser l'arbre MCTS
		rootNode = self._initMctsNode(boardState, legalActions)
		
		# Exécuter les simulations
		for _ in range(self.numMctsSimulations):
			node = rootNode
			gameState = np.copy(boardState)
			search_path = [node]
			
			# Phase de sélection: descendre dans l'arbre jusqu'à une feuille
			while node.isExpanded and not node.isTerminal:
				action, node = self._selectChild(node)
				
				# Mettre à jour l'état du jeu
				# Note: cette fonction est une simplification, il faudrait implémenter
				# une vraie fonction de transition d'état pour un MCTS complet
				gameState = self._applyAction(gameState, action)
				
				search_path.append(node)
			
			# Phase d'expansion et évaluation
			value = 0
			if not node.isTerminal:
				# Préparer l'entrée pour le réseau
				inputState = ChessNet.prepareInput(gameState)
				
				# Prédire la politique et la valeur
				policy, value = self.model.predict(inputState, self.device)
				
				# Masquer les actions illégales
				if node.legalActions:
					maskedPolicy = np.zeros_like(policy)
					maskedPolicy[node.legalActions] = policy[node.legalActions]
					maskedPolicy /= np.sum(maskedPolicy) + 1e-8
					policy = maskedPolicy
				
				# Expansion
				node.expand(policy)
			
			# Phase de rétropropagation
			for node in reversed(search_path):
				node.updateStats(value)
				value = -value  # Inverser la valeur pour le joueur adverse
		
		# Calculer la politique basée sur les visites
		counts = np.array([child.visitCount if child else 0 
						  for action, child in rootNode.children.items()])
		policy = counts / np.sum(counts)
		
		return policy
	
	def _initMctsNode(self, state: np.ndarray, legalActions: List[int]) -> 'MctsNode':
		"""
		Initialise un nœud MCTS pour l'état donné.
		
		Args:
			state: État du jeu
			legalActions: Liste des actions légales
			
		Returns:
			MctsNode: Nœud racine de l'arbre MCTS
		"""
		# Créer un identifiant unique pour l'état
		stateKey = self._stateToKey(state)
		
		# Récupérer le nœud du cache s'il existe
		if stateKey in self.mctsCache:
			return self.mctsCache[stateKey]
		
		# Sinon, créer un nouveau nœud
		node = MctsNode(legalActions)
		self.mctsCache[stateKey] = node
		
		return node
	
	def _selectChild(self, node: 'MctsNode') -> Tuple[int, 'MctsNode']:
		"""
		Sélectionne un enfant du nœud selon la politique UCB.
		
		Args:
			node: Nœud parent
			
		Returns:
			Tuple[int, MctsNode]: Action choisie et nœud enfant
		"""
		# Calculer la valeur UCB pour chaque enfant
		ucbValues = {}
		for action, child in node.children.items():
			if child is None:
				ucbValues[action] = float('inf')  # Favoriser les actions non explorées
			else:
				# Formule UCB classique: Q + c*P*sqrt(N_parent)/N_child
				exploitation = child.totalValue / (child.visitCount + 1e-8)
				exploration = self.cPuct * node.policy[action] * np.sqrt(node.visitCount) / (1 + child.visitCount)
				ucbValues[action] = exploitation + exploration
		
		# Sélectionner l'action avec la valeur UCB maximale
		bestAction = max(ucbValues, key=ucbValues.get)
		bestChild = node.children.get(bestAction)
		
		# Si l'enfant n'existe pas encore, le créer
		if bestChild is None:
			bestChild = MctsNode(node.legalActions)
			node.children[bestAction] = bestChild
		
		return bestAction, bestChild
	
	def _stateToKey(self, state: np.ndarray) -> str:
		"""
		Convertit un état en une clé unique pour le cache MCTS.
		
		Args:
			state: État du jeu
			
		Returns:
			str: Clé unique pour l'état
		"""
		# Compresser l'état pour économiser de la mémoire
		return state.tobytes()
	
	def _applyAction(self, state: np.ndarray, action: int) -> np.ndarray:
		"""
		Applique une action à un état pour produire un nouvel état.
		Simplification pour le MCTS.
		
		Args:
			state: État courant
			action: Action à appliquer
			
		Returns:
			np.ndarray: Nouvel état
		"""
		# Cette fonction devrait transformer l'état en fonction de l'action
		# Dans une implémentation complète, il faudrait intégrer la logique du jeu d'échecs
		# Pour l'instant, nous utilisons une simple transformation aléatoire
		
		# FIXME: Implémenter une vraie fonction de transition d'état
		newState = np.copy(state)
		
		# Simuler l'application de l'action
		# (pour une implémentation réelle, il faudrait modifier correctement l'état)
		
		return newState
	
	def train(self, experiences: List[Tuple]) -> Dict[str, float]:
		"""
		Entraîne le réseau neuronal à partir d'un lot d'expériences.
		
		Args:
			experiences: Liste d'expériences (état, action, récompense, état suivant, terminé)
			
		Returns:
			Dict[str, float]: Statistiques d'entraînement
		"""
		# Déballer les expériences
		states, policies, values = zip(*experiences)
		
		# Convertir en tenseurs
		states = torch.FloatTensor(np.array(states)).to(self.device)
		target_policies = torch.FloatTensor(np.array(policies)).to(self.device)
		target_values = torch.FloatTensor(np.array(values)).reshape(-1, 1).to(self.device)
		
		# Mettre le modèle en mode entraînement
		self.model.train()
		
		# Forward pass
		policy_logits, value_preds = self.model(states)
		
		# Calculer les pertes
		policy_loss = -torch.mean(torch.sum(target_policies * policy_logits, dim=1))
		value_loss = F.mse_loss(value_preds, target_values)
		total_loss = policy_loss + value_loss
		
		# Optimiser
		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		
		return {
			'policy_loss': policy_loss.item(),
			'value_loss': value_loss.item(),
			'total_loss': total_loss.item()
		}
	
	def saveModel(self, filePath: str) -> None:
		"""
		Sauvegarde le modèle sur le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		self.model.saveModel(filePath)
	
	def loadModel(self, filePath: str) -> None:
		"""
		Charge le modèle depuis le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		self.model.loadModel(filePath, self.device)


class MctsNode:
	"""Nœud de l'arbre de recherche Monte Carlo."""
	
	def __init__(self, legalActions: List[int]):
		"""
		Initialise un nœud MCTS.
		
		Args:
			legalActions: Liste des actions légales
		"""
		self.visitCount: int = 0
		self.totalValue: float = 0.0
		self.children: Dict[int, Optional[MctsNode]] = {action: None for action in legalActions}
		self.legalActions: List[int] = legalActions
		self.isExpanded: bool = False
		self.isTerminal: bool = False
		self.policy: Dict[int, float] = {}
	
	def expand(self, policy: np.ndarray) -> None:
		"""
		Développe le nœud avec une politique.
		
		Args:
			policy: Politique prédite par le réseau
		"""
		self.isExpanded = True
		
		# Normaliser la politique sur les actions légales
		for action in self.legalActions:
			self.policy[action] = policy[action]
	
	def updateStats(self, value: float) -> None:
		"""
		Met à jour les statistiques du nœud.
		
		Args:
			value: Valeur à ajouter
		"""
		self.visitCount += 1
		self.totalValue += value


class DQNAgent:
	"""
	Agent d'apprentissage par renforcement profond (DQN) pour les échecs.
	Alternative à l'approche MCTS.
	"""
	
	def __init__(self, 
				 stateSize: int = 19*8*8, 
				 actionSize: int = 8*8*8*8,
				 learningRate: float = 0.001,
				 gamma: float = 0.99,
				 epsilon: float = 1.0,
				 epsilonMin: float = 0.01,
				 epsilonDecay: float = 0.995,
				 batchSize: int = 64,
				 device: str = 'auto'):
		"""
		Initialise l'agent DQN.
		
		Args:
			stateSize: Taille de l'espace d'états
			actionSize: Taille de l'espace d'actions
			learningRate: Taux d'apprentissage
			gamma: Facteur d'actualisation
			epsilon: Taux d'exploration initial
			epsilonMin: Taux d'exploration minimal
			epsilonDecay: Facteur de décroissance de l'exploration
			batchSize: Taille du lot d'entraînement
			device: Périphérique d'exécution ('cpu', 'cuda', ou 'auto')
		"""
		# Déterminer le périphérique
		if device == 'auto':
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)
		
		print(f"Utilisation du périphérique: {self.device}")
		
		# Paramètres d'apprentissage
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilonMin = epsilonMin
		self.epsilonDecay = epsilonDecay
		self.learningRate = learningRate
		self.batchSize = batchSize
		
		# Tailles des espaces
		self.stateSize = stateSize
		self.actionSize = actionSize
		
		# Créer les réseaux Q (principal et cible)
		self.qNetwork = QNetwork().to(self.device)
		self.targetNetwork = QNetwork().to(self.device)
		self.updateTargetNetwork()  # Synchroniser les poids initiaux
		
		# Optimiseur
		self.optimizer = torch.optim.Adam(self.qNetwork.parameters(), lr=learningRate)
		
		# Tampon de rejeu
		self.memory = ReplayBuffer()
		
		# Compteur d'étapes pour la mise à jour du réseau cible
		self.updateCounter = 0
		self.targetUpdateFrequency = 1000  # Fréquence de mise à jour du réseau cible
	
	def selectAction(self, state: Dict, legalActions: List[int] = None) -> int:
		"""
		Sélectionne une action selon la politique epsilon-greedy.
		
		Args:
			state: État du jeu
			legalActions: Liste des actions légales
			
		Returns:
			int: Indice de l'action choisie
		"""
		# Si aucune action légale n'est fournie, utiliser toutes les actions
		if legalActions is None or len(legalActions) == 0:
			return random.randint(0, self.actionSize - 1)
		
		# Exploration aléatoire avec probabilité epsilon
		if random.random() < self.epsilon:
			return random.choice(legalActions)
		
		# Exploitation: choisir la meilleure action selon le réseau Q
		boardState = state['board_state']
		inputState = QNetwork.prepareInput(boardState)
		
		# Prédire les valeurs Q
		q_values = self.qNetwork.predict(inputState, self.device)
		
		# Masquer les actions illégales avec une valeur très basse
		maskedQValues = np.ones(self.actionSize) * float('-inf')
		maskedQValues[legalActions] = q_values[legalActions]
		
		# Choisir l'action avec la valeur Q maximale
		return np.argmax(maskedQValues)
	
	def remember(self, state: np.ndarray, action: int, reward: float, 
				 next_state: np.ndarray, done: bool) -> None:
		"""
		Stocke une expérience dans le tampon de rejeu.
		
		Args:
			state: État courant
			action: Action choisie
			reward: Récompense obtenue
			next_state: État suivant
			done: Indicateur de fin d'épisode
		"""
		self.memory.push(state, action, reward, next_state, done)
	
	def learn(self) -> Optional[Dict[str, float]]:
		"""
		Apprend à partir d'un lot d'expériences du tampon de rejeu.
		
		Returns:
			Optional[Dict[str, float]]: Statistiques d'apprentissage
		"""
		# Ne rien faire si le tampon n'est pas assez rempli
		if len(self.memory) < self.batchSize:
			return None
		
		# Échantillonner un lot d'expériences
		states, actions, rewards, next_states, dones = self.memory.sample(self.batchSize)
		
		# Convertir en tenseurs
		states = torch.FloatTensor(np.array([QNetwork.prepareInput(s) for s in states])).to(self.device)
		actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
		rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
		next_states = torch.FloatTensor(np.array([QNetwork.prepareInput(s) for s in next_states])).to(self.device)
		dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
		
		# Calculer les valeurs Q actuelles
		current_q_values = self.qNetwork(states).gather(1, actions)
		
		# Calculer les valeurs Q cibles
		with torch.no_grad():
			next_q_values = self.targetNetwork(next_states).max(1, keepdim=True)[0]
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
		
		# Calculer la perte
		loss = F.mse_loss(current_q_values, target_q_values)
		
		# Optimisation
		self.optimizer.zero_grad()
		loss.backward()
		# Clip du gradient pour éviter l'explosion
		torch.nn.utils.clip_grad_norm_(self.qNetwork.parameters(), 1.0)
		self.optimizer.step()
		
		# Mettre à jour le réseau cible périodiquement
		self.updateCounter += 1
		if self.updateCounter % self.targetUpdateFrequency == 0:
			self.updateTargetNetwork()
		
		# Décroissance de epsilon
		self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)
		
		return {
			'loss': loss.item(),
			'epsilon': self.epsilon
		}
	
	def updateTargetNetwork(self) -> None:
		"""Met à jour le réseau cible avec les poids du réseau principal."""
		self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
	
	def saveModel(self, filePath: str) -> None:
		"""
		Sauvegarde le modèle sur le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		self.qNetwork.saveModel(filePath)
	
	def loadModel(self, filePath: str) -> None:
		"""
		Charge le modèle depuis le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		self.qNetwork.loadModel(filePath, self.device)
		self.updateTargetNetwork()