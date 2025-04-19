from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np
import random
import time
import gym
from gym import spaces

from modules.ChessBoard import ChessBoard, PieceColor
from modules.PieceType import PieceType
from modules.Move import Move


class ChessEnv(gym.Env):
	"""
	Environnement d'échecs compatible avec l'interface Gym pour l'apprentissage par renforcement.
	
	Cette classe encapsule le jeu d'échecs en tant qu'environnement avec:
	- Un état (plateau)
	- Des actions (coups)
	- Des récompenses
	- Des états terminaux
	"""
	
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def __init__(self, render_mode: Optional[str] = None):
		"""
		Initialise l'environnement d'échecs.
		
		Args:
			render_mode: Mode de rendu ('human', 'rgb_array' ou None)
		"""
		super(ChessEnv, self).__init__()
		
		# Créer un nouveau plateau
		self.board: ChessBoard = ChessBoard()
		
		# Mode de rendu
		self.render_mode: Optional[str] = render_mode
		self.visualizer = None  # Sera initialisé à la demande
		
		# Compteurs pour les statistiques
		self.episodeSteps: int = 0
		self.totalGames: int = 0
		self.movesWithoutProgress: int = 0  # Pour les récompenses négatives sur les boucles
		
		# Limite de pas pour éviter les parties infinies
		self.maxEpisodeSteps: int = 1000
		
		# Définition de l'espace d'action
		# 8×8 positions de départ, 8×8 positions d'arrivée
		# (plus quelques actions supplémentaires pour les promotions)
		self.action_space = spaces.Discrete(8*8*8*8)
		
		# Définition de l'espace d'observation
		# Plateau 8×8 avec 19 canaux:
		# - 12 canaux pour les pièces (6 types × 2 couleurs)
		# - 1 canal pour le joueur actuel
		# - 4 canaux pour les droits de roque
		# - 1 canal pour la cible en passant
		# - 1 canal pour le compteur de demi-coups (règle des 50 coups)
		self.observation_space = spaces.Box(
			low=0, high=1, shape=(8, 8, 19), dtype=np.float32
		)
	
	def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict:
		"""
		Réinitialise l'environnement au début d'un nouvel épisode.
		
		Args:
			seed: Graine aléatoire pour la reproductibilité (optionnel)
			options: Options supplémentaires (optionnel)
			
		Returns:
			Dict: Observation de l'état initial
		"""
		# Réinitialiser le générateur aléatoire si la graine est fournie
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		
		# Réinitialiser le plateau
		self.board = ChessBoard()
		
		# Réinitialiser les compteurs
		self.episodeSteps = 0
		self.movesWithoutProgress = 0
		self.totalGames += 1
		
		# Obtenir l'observation initiale
		observation = self._getObservation()
		
		return observation
	
	def step(self, action: Union[int, Move]) -> Tuple[Dict, float, bool, bool, Dict]:
		"""
		Exécute une action dans l'environnement.
		
		Args:
			action: L'action à exécuter (indice ou objet Move)
			
		Returns:
			Tuple[Dict, float, bool, bool, Dict]: 
				- Observation
				- Récompense
				- Terminé (partie terminée)
				- Tronqué (limite d'étapes atteinte)
				- Informations supplémentaires
		"""
		# Convertir l'action en objet Move si nécessaire
		move = action if isinstance(action, Move) else self._actionToMove(action)
		
		# Obtenir le joueur actuel avant d'effectuer le coup
		current_player = self.board.currentPlayer
		
		# Vérifier si le mouvement est légal
		if not self.board.isLegalMove(move):
			# Mouvement illégal, pénalité et fin de partie
			return (
				self._getObservation(),
				-10.0,  # Forte pénalité pour un coup illégal
				True,   # Épisode terminé
				False,  # Non tronqué
				{'info': 'Coup illégal', 'valid_move': False}
			)
		
		# Sauvegarder l'état avant le coup pour calculer la récompense
		material_before = self._countMaterial()
		in_check_before = self.board.isInCheck(current_player)
		
		# Exécuter le coup
		self.board.makeMove(move)
		self.episodeSteps += 1
		
		# Évaluer le nouvel état
		material_after = self._countMaterial()
		material_delta = material_after[current_player] - material_before[current_player]
		
		# Vérifier si la partie est terminée
		done = False
		reward = 0.0
		info = {'valid_move': True}
		
		# Vérifier l'échec et mat
		opponent = PieceColor.BLACK if current_player == PieceColor.WHITE else PieceColor.WHITE
		if self.board.isCheckmate(opponent):
			done = True
			reward = 1.0  # Victoire
			info['outcome'] = 'checkmate'
		elif self.board.isCheckmate(current_player):
			done = True
			reward = -1.0  # Défaite
			info['outcome'] = 'checkmate'
		# Vérifier le pat ou l'égalité
		elif self.board.isDraw():
			done = True
			reward = 0.0  # Match nul
			info['outcome'] = 'draw'
		else:
			# La partie continue, calculer la récompense par rapport au matériel capturé
			reward = 0.01 * material_delta  # Petit bonus/malus selon le matériel gagné/perdu
			
			# Petit bonus pour mettre l'adversaire en échec
			if self.board.isInCheck(opponent):
				reward += 0.05
			
			# Petit bonus pour sortir de l'échec
			if in_check_before and not self.board.isInCheck(current_player):
				reward += 0.05
		
		# Vérifier si on dépasse le nombre maximal d'étapes
		truncated = self.episodeSteps >= self.maxEpisodeSteps
		if truncated:
			done = True
			info['outcome'] = 'max_steps_reached'
		
		# Pénalité pour éviter les parties trop longues sans capture
		if material_delta == 0:
			self.movesWithoutProgress += 1
			# Pénalité progressive qui augmente avec le nombre de coups sans progrès
			reward -= 0.001 * self.movesWithoutProgress
		else:
			self.movesWithoutProgress = 0
		
		# Ajouter des informations supplémentaires
		info['steps'] = self.episodeSteps
		info['current_player'] = self.board.currentPlayer.name
		info['is_check'] = self.board.isInCheck(self.board.currentPlayer)
		info['total_games'] = self.totalGames
		info['move'] = str(move)
		
		return (
			self._getObservation(),
			reward,
			done,
			truncated,
			info
		)
	
	def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
		"""
		Rend l'environnement courant.
		
		Args:
			mode: Mode de rendu ('human', 'rgb_array')
			
		Returns:
			Optional[np.ndarray]: Image du plateau en cas de mode 'rgb_array'
		"""
		mode = mode or self.render_mode
		
		if mode is None:
			return None
		
		if mode == 'human' or mode == 'rgb_array':
			# Initialiser le visualiseur si nécessaire
			if self.visualizer is None:
				try:
					from Visualizer import ChessVisualizer
					self.visualizer = ChessVisualizer(mode == 'human')
				except ImportError:
					print("Visualiseur non disponible. Veuillez installer pygame.")
					return None
			
			# Rendre le plateau
			return self.visualizer.render(self.board, mode)
		
		else:
			raise ValueError(f"Mode de rendu non pris en charge: {mode}")
	
	def close(self) -> None:
		"""Ferme l'environnement et libère les ressources."""
		if self.visualizer is not None:
			self.visualizer.close()
			self.visualizer = None
	
	def _getObservation(self) -> Dict:
		"""
		Récupère l'observation actuelle de l'environnement.
		
		Returns:
			Dict: Dictionnaire contenant l'état du jeu
		"""
		# Obtenir l'état complet du jeu
		gameState = self.board.getGameState()
		
		return {
			'board_state': gameState['board_state'],
			'legal_moves': [move.toAlgebraic() for move in gameState['legal_moves']],
			'current_player': gameState['current_player'].name,
			'is_check': gameState['is_check'],
			'is_terminal': gameState['is_checkmate'] or gameState['is_draw']
		}
	
	def _actionToMove(self, action: int) -> Move:
		"""
		Convertit un indice d'action en objet Move.
		
		Args:
			action: Indice de l'action dans l'espace d'action
			
		Returns:
			Move: Objet Move correspondant
		"""
		# Extraire les coordonnées à partir de l'indice
		startRow = action // (8*8*8)
		remainder = action % (8*8*8)
		startCol = remainder // (8*8)
		remainder = remainder % (8*8)
		endRow = remainder // 8
		endCol = remainder % 8
		
		# Vérifier si c'est une promotion
		promotion = None
		if (startRow == 6 and endRow == 7) or (startRow == 1 and endRow == 0):
			# Pour simplifier, on promeut toujours en Dame
			# Pour un comportement plus avancé, il faudrait élargir l'espace d'action
			promotion = PieceType.QUEEN
		
		return Move(startRow, startCol, endRow, endCol, promotion)
	
	def _moveToAction(self, move: Move) -> int:
		"""
		Convertit un objet Move en indice d'action.
		
		Args:
			move: Objet Move
			
		Returns:
			int: Indice de l'action
		"""
		# Calculer l'indice à partir des coordonnées
		action = move.startRow * (8*8*8) + move.startCol * (8*8) + move.endRow * 8 + move.endCol
		return action
	
	def _countMaterial(self) -> Dict[PieceColor, float]:
		"""
		Compte le matériel sur le plateau pour chaque joueur.
		
		Returns:
			Dict[PieceColor, float]: Valeur du matériel pour chaque joueur
		"""
		# Valeurs standard des pièces
		pieceValues = {
			PieceType.PAWN: 1.0,
			PieceType.KNIGHT: 3.0,
			PieceType.BISHOP: 3.0,
			PieceType.ROOK: 5.0,
			PieceType.QUEEN: 9.0,
			PieceType.KING: 0.0  # Le roi n'a pas de valeur matérielle
		}
		
		material = {
			PieceColor.WHITE: 0.0,
			PieceColor.BLACK: 0.0
		}
		
		# Parcourir toutes les cases du plateau
		for row in range(8):
			for col in range(8):
				piece = self.board.getPiece(row, col)
				if piece is not None:
					material[piece.color] += pieceValues[piece.type]
		
		return material
	
	def getSelfPlayObservation(self) -> Dict:
		"""
		Obtient l'observation pour l'entraînement en auto-jeu.
		L'observation est toujours du point de vue du joueur actuel.
		
		Returns:
			Dict: Observation normalisée
		"""
		obs = self._getObservation()
		
		# Si c'est le tour des noirs, inverser la perspective
		if self.board.currentPlayer == PieceColor.BLACK:
			# Rotation du plateau de 180 degrés
			board_state = obs['board_state']
			# Inverser les 12 premiers canaux (pièces)
			piece_channels = board_state[:, :, :12]
			# Réorganiser les canaux pour échanger blancs et noirs
			white_pieces = piece_channels[:, :, :6]
			black_pieces = piece_channels[:, :, 6:12]
			# Rotation de 180 degrés
			white_pieces = np.rot90(white_pieces, k=2, axes=(0, 1))
			black_pieces = np.rot90(black_pieces, k=2, axes=(0, 1))
			# Réorganiser pour la perspective du joueur noir
			piece_channels = np.concatenate([black_pieces, white_pieces], axis=2)
			
			# Mettre à jour les autres canaux
			other_channels = board_state[:, :, 12:]
			other_channels = np.rot90(other_channels, k=2, axes=(0, 1))
			
			# Reconstruire l'état du plateau
			board_state = np.concatenate([piece_channels, other_channels], axis=2)
			obs['board_state'] = board_state
		
		return obs
	
	def getLegalMoves(self) -> List[Move]:
		"""
		Récupère la liste des coups légaux pour le joueur actuel.
		
		Returns:
			List[Move]: Liste des coups légaux
		"""
		return self.board.getAllLegalMoves(self.board.currentPlayer)
	
	def getLegalMovesAsActions(self) -> List[int]:
		"""
		Récupère la liste des coups légaux sous forme d'indices d'action.
		
		Returns:
			List[int]: Liste des indices d'action légaux
		"""
		legal_moves = self.getLegalMoves()
		return [self._moveToAction(move) for move in legal_moves]