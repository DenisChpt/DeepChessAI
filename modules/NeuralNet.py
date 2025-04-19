from __future__ import annotations
from typing import Tuple, Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChessConvBlock(nn.Module):
	"""Bloc de convolution avec normalisation et activation."""
	
	def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 3):
		"""
		Initialise un bloc de convolution.
		
		Args:
			inChannels: Nombre de canaux d'entrée
			outChannels: Nombre de canaux de sortie
			kernelSize: Taille du noyau de convolution
		"""
		super(ChessConvBlock, self).__init__()
		self.conv = nn.Conv2d(
			inChannels, 
			outChannels, 
			kernel_size=kernelSize, 
			padding=kernelSize//2
		)
		self.bn = nn.BatchNorm2d(outChannels)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'entrée à travers le bloc.
		
		Args:
			x: Tenseur d'entrée [batch_size, inChannels, height, width]
			
		Returns:
			torch.Tensor: Sortie du bloc [batch_size, outChannels, height, width]
		"""
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		return x


class ResidualBlock(nn.Module):
	"""Bloc résiduel pour le réseau de deep learning."""
	
	def __init__(self, numChannels: int, kernelSize: int = 3):
		"""
		Initialise un bloc résiduel.
		
		Args:
			numChannels: Nombre de canaux
			kernelSize: Taille du noyau de convolution
		"""
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			numChannels, 
			numChannels, 
			kernel_size=kernelSize, 
			padding=kernelSize//2
		)
		self.bn1 = nn.BatchNorm2d(numChannels)
		self.conv2 = nn.Conv2d(
			numChannels, 
			numChannels, 
			kernel_size=kernelSize, 
			padding=kernelSize//2
		)
		self.bn2 = nn.BatchNorm2d(numChannels)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'entrée à travers le bloc.
		
		Args:
			x: Tenseur d'entrée [batch_size, numChannels, height, width]
			
		Returns:
			torch.Tensor: Sortie du bloc [batch_size, numChannels, height, width]
		"""
		residual = x
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))
		x += residual
		x = F.relu(x)
		return x


class ChessNet(nn.Module):
	"""
	Réseau neuronal pour les échecs combinant réseaux de politique et de valeur.
	Inspiré de l'architecture AlphaZero.
	"""
	
	def __init__(self, 
				 inputChannels: int = 19, 
				 numFilters: int = 128, 
				 numBlocks: int = 10,
				 policySize: int = 8*8*8*8  # taille de l'espace d'action
				 ):
		"""
		Initialise le réseau neuronal.
		
		Args:
			inputChannels: Nombre de canaux d'entrée (représentation du plateau)
			numFilters: Nombre de filtres dans les couches de convolution
			numBlocks: Nombre de blocs résiduels
			policySize: Taille du vecteur de politique (espace d'action)
		"""
		super(ChessNet, self).__init__()
		
		# Couche d'entrée
		self.inputConv = ChessConvBlock(inputChannels, numFilters)
		
		# Blocs résiduels
		self.residualBlocks = nn.ModuleList(
			[ResidualBlock(numFilters) for _ in range(numBlocks)]
		)
		
		# Tête de politique (prédire le coup suivant)
		self.policyConv = ChessConvBlock(numFilters, 32)
		self.policyHead = nn.Linear(32 * 8 * 8, policySize)
		
		# Tête de valeur (évaluer la position)
		self.valueConv = ChessConvBlock(numFilters, 32)
		self.valueFC1 = nn.Linear(32 * 8 * 8, 128)
		self.valueHead = nn.Linear(128, 1)
	
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Propage l'entrée à travers le réseau.
		
		Args:
			x: Tenseur d'entrée [batch_size, inputChannels, 8, 8]
			
		Returns:
			Tuple[torch.Tensor, torch.Tensor]: 
				- Politique [batch_size, policySize]
				- Valeur [batch_size, 1]
		"""
		# Couche d'entrée
		x = self.inputConv(x)
		
		# Blocs résiduels
		for block in self.residualBlocks:
			x = block(x)
		
		# Tête de politique
		policy = self.policyConv(x)
		policy = policy.view(policy.size(0), -1)  # Aplatir
		policy = self.policyHead(policy)
		policy = F.log_softmax(policy, dim=1)  # Log softmax pour la stabilité numérique
		
		# Tête de valeur
		value = self.valueConv(x)
		value = value.view(value.size(0), -1)  # Aplatir
		value = F.relu(self.valueFC1(value))
		value = torch.tanh(self.valueHead(value))  # tanh pour sortie entre -1 et 1
		
		return policy, value
	
	def predict(self, state: np.ndarray, device: torch.device = None) -> Tuple[np.ndarray, float]:
		"""
		Prédit la politique et la valeur pour un état donné.
		
		Args:
			state: État du jeu [19, 8, 8]
			device: Périphérique d'exécution (CPU ou GPU)
			
		Returns:
			Tuple[np.ndarray, float]: 
				- Politique [policySize]
				- Valeur [-1, 1]
		"""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Convertir l'état en tenseur
		x = torch.FloatTensor(state).unsqueeze(0).to(device)  # Ajouter dimension batch
		
		# Désactiver le calcul du gradient pour l'inférence
		with torch.no_grad():
			self.eval()  # Mode évaluation
			policy_logits, value = self.forward(x)
			
			# Convertir en numpy
			policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
			value = value.item()
		
		return policy, value
	
	def saveModel(self, filePath: str) -> None:
		"""
		Sauvegarde le modèle sur le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		torch.save({
			'model_state_dict': self.state_dict(),
		}, filePath)
		print(f"Modèle sauvegardé dans {filePath}")
	
	def loadModel(self, filePath: str, device: torch.device = None) -> None:
		"""
		Charge le modèle depuis le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
			device: Périphérique d'exécution (CPU ou GPU)
		"""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		checkpoint = torch.load(filePath, map_location=device)
		self.load_state_dict(checkpoint['model_state_dict'])
		self.to(device)
		print(f"Modèle chargé depuis {filePath}")
	
	@staticmethod
	def prepareInput(boardState: np.ndarray) -> np.ndarray:
		"""
		Prépare l'état du plateau pour l'entrée du réseau.
		
		Args:
			boardState: État du plateau [8, 8, 19]
			
		Returns:
			np.ndarray: État préparé [19, 8, 8] (format PyTorch)
		"""
		# Réorganiser les dimensions pour PyTorch: [H, W, C] -> [C, H, W]
		return np.transpose(boardState, (2, 0, 1))


class QNetwork(nn.Module):
	"""
	Réseau Q pour l'apprentissage par renforcement profond (DQN).
	Alternative à l'approche AlphaZero.
	"""
	
	def __init__(self, 
				 inputChannels: int = 19, 
				 numFilters: int = 128, 
				 numBlocks: int = 6,
				 actionSize: int = 8*8*8*8  # taille de l'espace d'action
				 ):
		"""
		Initialise le réseau Q.
		
		Args:
			inputChannels: Nombre de canaux d'entrée (représentation du plateau)
			numFilters: Nombre de filtres dans les couches de convolution
			numBlocks: Nombre de blocs résiduels
			actionSize: Taille de l'espace d'action
		"""
		super(QNetwork, self).__init__()
		
		# Couche d'entrée
		self.inputConv = ChessConvBlock(inputChannels, numFilters)
		
		# Blocs résiduels
		self.residualBlocks = nn.ModuleList(
			[ResidualBlock(numFilters) for _ in range(numBlocks)]
		)
		
		# Couches entièrement connectées
		self.fc1 = nn.Linear(numFilters * 8 * 8, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, actionSize)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Propage l'entrée à travers le réseau.
		
		Args:
			x: Tenseur d'entrée [batch_size, inputChannels, 8, 8]
			
		Returns:
			torch.Tensor: Q-values pour chaque action [batch_size, actionSize]
		"""
		# Couche d'entrée
		x = self.inputConv(x)
		
		# Blocs résiduels
		for block in self.residualBlocks:
			x = block(x)
		
		# Aplatir
		x = x.view(x.size(0), -1)
		
		# Couches entièrement connectées
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x
	
	def predict(self, state: np.ndarray, device: torch.device = None) -> np.ndarray:
		"""
		Prédit les Q-values pour un état donné.
		
		Args:
			state: État du jeu [19, 8, 8]
			device: Périphérique d'exécution (CPU ou GPU)
			
		Returns:
			np.ndarray: Q-values [actionSize]
		"""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Convertir l'état en tenseur
		x = torch.FloatTensor(state).unsqueeze(0).to(device)  # Ajouter dimension batch
		
		# Désactiver le calcul du gradient pour l'inférence
		with torch.no_grad():
			self.eval()  # Mode évaluation
			q_values = self.forward(x)
			
			# Convertir en numpy
			q_values = q_values.cpu().numpy()[0]
		
		return q_values
	
	def saveModel(self, filePath: str) -> None:
		"""
		Sauvegarde le modèle sur le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
		"""
		torch.save({
			'model_state_dict': self.state_dict(),
		}, filePath)
		print(f"Modèle sauvegardé dans {filePath}")
	
	def loadModel(self, filePath: str, device: torch.device = None) -> None:
		"""
		Charge le modèle depuis le disque.
		
		Args:
			filePath: Chemin du fichier de sauvegarde
			device: Périphérique d'exécution (CPU ou GPU)
		"""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		checkpoint = torch.load(filePath, map_location=device)
		self.load_state_dict(checkpoint['model_state_dict'])
		self.to(device)
		print(f"Modèle chargé depuis {filePath}")
	
	@staticmethod
	def prepareInput(boardState: np.ndarray) -> np.ndarray:
		"""
		Prépare l'état du plateau pour l'entrée du réseau.
		
		Args:
			boardState: État du plateau [?, ?, ?]
			
		Returns:
			np.ndarray: État préparé [19, 8, 8] (format PyTorch)
		"""
		# Vérifions la forme de l'entrée
		if boardState.shape == (19, 8, 8):
			# Déjà dans le bon format, pas besoin de transposer
			return boardState
		elif boardState.shape == (8, 8, 19):
			# Format [H, W, C] -> [C, H, W]
			return np.transpose(boardState, (2, 0, 1))
		elif boardState.shape == (8, 19, 8):
			# Déjà transposé, mais incorrectement - corrigeons
			return np.transpose(boardState, (1, 0, 2))
		else:
			# Format inconnu, essayons de le diagnostiquer
			print(f"Forme inattendue du boardState: {boardState.shape}")
			# Tentons de mettre les canaux en premier
			if 19 in boardState.shape:
				# Trouver l'indice des canaux
				channel_idx = boardState.shape.index(19)
				# Créer une liste de permutation pour mettre les canaux en premier
				perm = list(range(len(boardState.shape)))
				perm.remove(channel_idx)
				perm.insert(0, channel_idx)
				return np.transpose(boardState, perm)
			else:
				# Impossible de déterminer, retournons tel quel
				return boardState