from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Union
import pygame
import numpy as np
import os
import sys
from pygame.locals import *

from modules.ChessBoard import ChessBoard, Piece, PieceType, PieceColor
from modules.Move import Move


class ChessVisualizer:
	"""
	Visualiseur de jeu d'échecs utilisant Pygame.
	Permet l'affichage graphique du plateau et l'interaction humaine.
	"""
	
	def __init__(self, interactive: bool = True, windowSize: int = 600):
		"""
		Initialise le visualiseur d'échecs.
		
		Args:
			interactive: Si True, affiche la fenêtre et permet l'interaction
			windowSize: Taille de la fenêtre en pixels
		"""
		self.interactive = interactive
		self.windowSize = windowSize
		self.squareSize = windowSize // 8
		
		# Couleurs
		self.darkSquareColor = (181, 136, 99)  # Marron
		self.lightSquareColor = (240, 217, 181)  # Beige
		self.highlightColor = (124, 252, 0, 128)  # Vert semi-transparent
		self.moveIndicatorColor = (50, 50, 50)
		
		# États de l'interface
		self.selectedSquare = None
		self.possibleMoves = []
		self.lastMove = None
		
		# Initialiser pygame si mode interactif
		if interactive:
			pygame.init()
			pygame.display.set_caption("Jeu d'Échecs - RL")
			self.screen = pygame.display.set_mode((windowSize, windowSize))
			self.clock = pygame.time.Clock()
		else:
			# Initialisation minimale pour le rendu non interactif
			pygame.init()
			self.screen = pygame.Surface((windowSize, windowSize))
		
		# Charger les images
		self.pieceImages = self._loadPieceImages()
		
		# Police pour les légendes
		self.font = pygame.font.SysFont('Arial', 14)
	
	def _loadPieceImages(self) -> Dict:
		"""
		Charge les images des pièces.
		
		Returns:
			Dict: Dictionnaire d'images par pièce et couleur
		"""
		images = {}
		
		# Chemin vers le dossier des pièces
		piecesDir = os.path.join(os.path.dirname(__file__), 'assets', 'pieces')
		
		# Créer le dossier s'il n'existe pas
		os.makedirs(piecesDir, exist_ok=True)
		
		# Vérifier si les pièces existent, sinon utiliser des formes simples
		pieceFiles = {
			(PieceColor.WHITE, PieceType.PAWN): "wp.png",
			(PieceColor.WHITE, PieceType.KNIGHT): "wn.png",
			(PieceColor.WHITE, PieceType.BISHOP): "wb.png",
			(PieceColor.WHITE, PieceType.ROOK): "wr.png",
			(PieceColor.WHITE, PieceType.QUEEN): "wq.png",
			(PieceColor.WHITE, PieceType.KING): "wk.png",
			(PieceColor.BLACK, PieceType.PAWN): "bp.png",
			(PieceColor.BLACK, PieceType.KNIGHT): "bn.png",
			(PieceColor.BLACK, PieceType.BISHOP): "bb.png",
			(PieceColor.BLACK, PieceType.ROOK): "br.png",
			(PieceColor.BLACK, PieceType.QUEEN): "bq.png",
			(PieceColor.BLACK, PieceType.KING): "bk.png",
		}
		
		imagesPresent = all(os.path.exists(os.path.join(piecesDir, file)) for file in pieceFiles.values())
		
		if imagesPresent:
			# Charger les images
			for (color, type), filename in pieceFiles.items():
				filepath = os.path.join(piecesDir, filename)
				try:
					image = pygame.image.load(filepath)
					images[(color, type)] = pygame.transform.scale(
						image, (self.squareSize, self.squareSize)
					)
				except pygame.error:
					# Si l'image ne peut pas être chargée, on utilisera des formes
					imagesPresent = False
					break
		
		if not imagesPresent:
			print("Images de pièces non trouvées. Utilisation de formes simples.")
			# Créer des formes simples pour les pièces
			for color in [PieceColor.WHITE, PieceColor.BLACK]:
				colorValue = (255, 255, 255) if color == PieceColor.WHITE else (0, 0, 0)
				borderValue = (0, 0, 0) if color == PieceColor.WHITE else (255, 255, 255)
				
				for pieceType in PieceType:
					# Créer une surface pour chaque pièce
					surface = pygame.Surface((self.squareSize, self.squareSize), pygame.SRCALPHA)
					
					# Dessiner la forme de base (cercle)
					pygame.draw.circle(
						surface,
						colorValue,
						(self.squareSize // 2, self.squareSize // 2),
						self.squareSize // 3
					)
					
					# Ajouter un contour
					pygame.draw.circle(
						surface,
						borderValue,
						(self.squareSize // 2, self.squareSize // 2),
						self.squareSize // 3,
						2
					)
					
					# Ajouter un symbole pour identifier la pièce
					symbols = {
						PieceType.PAWN: "P",
						PieceType.KNIGHT: "N",
						PieceType.BISHOP: "B",
						PieceType.ROOK: "R",
						PieceType.QUEEN: "Q",
						PieceType.KING: "K"
					}
					
					symbol = self.font.render(symbols[pieceType], True, borderValue)
					symbol_rect = symbol.get_rect(center=(self.squareSize // 2, self.squareSize // 2))
					surface.blit(symbol, symbol_rect)
					
					# Ajouter l'image au dictionnaire
					images[(color, pieceType)] = surface
		
		return images
	
	def render(self, board: ChessBoard, mode: str = 'human') -> Optional[np.ndarray]:
		"""
		Rend le plateau d'échecs.
		
		Args:
			board: Plateau d'échecs à afficher
			mode: Mode de rendu ('human' ou 'rgb_array')
			
		Returns:
			Optional[np.ndarray]: Image du plateau en cas de mode 'rgb_array'
		"""
		# Dessiner le plateau
		self._drawBoard(board)
		
		# Dessiner les pièces
		self._drawPieces(board)
		
		# Dessiner les cases sélectionnées et les mouvements possibles
		self._drawHighlights(board)
		
		# Dessiner les coordonnées
		self._drawCoordinates()
		
		# Afficher ou retourner l'image selon le mode
		if mode == 'human' and self.interactive:
			pygame.display.flip()
			self.clock.tick(30)
			return None
		elif mode == 'rgb_array':
			# Convertir la surface pygame en tableau numpy
			return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
		
		return None
	
	def _drawBoard(self, board: ChessBoard) -> None:
		"""
		Dessine le plateau vide.
		
		Args:
			board: Plateau d'échecs
		"""
		self.screen.fill(self.darkSquareColor)
		
		# Dessiner les cases claires
		for row in range(8):
			for col in range(8):
				if (row + col) % 2 == 0:  # Cases claires
					pygame.draw.rect(
						self.screen,
						self.lightSquareColor,
						pygame.Rect(
							col * self.squareSize,
							(7 - row) * self.squareSize,  # Inverser l'axe y pour coordonnées d'échecs
							self.squareSize,
							self.squareSize
						)
					)
		
		# Dessiner le dernier coup joué
		if self.lastMove:
			start_col, start_row = self.lastMove.startCol, self.lastMove.startRow
			end_col, end_row = self.lastMove.endCol, self.lastMove.endRow
			
			# Dessiner un cercle sur la case de départ
			pygame.draw.circle(
				self.screen,
				self.moveIndicatorColor,
				(
					start_col * self.squareSize + self.squareSize // 2,
					(7 - start_row) * self.squareSize + self.squareSize // 2
				),
				self.squareSize // 8,
				2
			)
			
			# Dessiner un cercle sur la case d'arrivée
			pygame.draw.circle(
				self.screen,
				self.moveIndicatorColor,
				(
					end_col * self.squareSize + self.squareSize // 2,
					(7 - end_row) * self.squareSize + self.squareSize // 2
				),
				self.squareSize // 8,
				2
			)
	
	def _drawPieces(self, board: ChessBoard) -> None:
		"""
		Dessine les pièces sur le plateau.
		
		Args:
			board: Plateau d'échecs
		"""
		for row in range(8):
			for col in range(8):
				piece = board.getPiece(row, col)
				if piece:
					# Obtenir l'image de la pièce
					image = self.pieceImages.get((piece.color, piece.type))
					
					if image:
						# Dessiner la pièce
						self.screen.blit(
							image,
							pygame.Rect(
								col * self.squareSize,
								(7 - row) * self.squareSize,  # Inverser l'axe y
								self.squareSize,
								self.squareSize
							)
						)
	
	def _drawHighlights(self, board: ChessBoard) -> None:
		"""
		Dessine les surbrillances pour la sélection et les coups possibles.
		
		Args:
			board: Plateau d'échecs
		"""
		# Dessiner la case sélectionnée
		if self.selectedSquare:
			row, col = self.selectedSquare
			highlightSurface = pygame.Surface((self.squareSize, self.squareSize), pygame.SRCALPHA)
			highlightSurface.fill((255, 255, 0, 100))  # Jaune semi-transparent
			
			self.screen.blit(
				highlightSurface,
				pygame.Rect(
					col * self.squareSize,
					(7 - row) * self.squareSize,
					self.squareSize,
					self.squareSize
				)
			)
		
		# Dessiner les mouvements possibles
		for move in self.possibleMoves:
			row, col = move.endRow, move.endCol
			
			# Créer une surface semi-transparente
			highlightSurface = pygame.Surface((self.squareSize, self.squareSize), pygame.SRCALPHA)
			highlightSurface.fill((124, 252, 0, 100))  # Vert semi-transparent
			
			self.screen.blit(
				highlightSurface,
				pygame.Rect(
					col * self.squareSize,
					(7 - row) * self.squareSize,
					self.squareSize,
					self.squareSize
				)
			)
	
	def _drawCoordinates(self) -> None:
		"""Dessine les coordonnées autour du plateau."""
		# Couleur du texte
		textColor = (50, 50, 50)
		
		# Dessiner les lettres (a-h)
		for col in range(8):
			letter = chr(ord('a') + col)
			text = self.font.render(letter, True, textColor)
			self.screen.blit(
				text,
				(
					col * self.squareSize + self.squareSize - 12,
					self.windowSize - 15
				)
			)
		
		# Dessiner les chiffres (1-8)
		for row in range(8):
			number = str(row + 1)
			text = self.font.render(number, True, textColor)
			self.screen.blit(
				text,
				(
					5,
					(7 - row) * self.squareSize + 5
				)
			)
	
	def processEvents(self, board: ChessBoard) -> Optional[Move]:
		"""
		Traite les événements pygame pour l'interaction humaine.
		
		Args:
			board: Plateau d'échecs actuel
			
		Returns:
			Optional[Move]: Coup choisi par l'utilisateur ou None
		"""
		if not self.interactive:
			return None
		
		for event in pygame.event.get():
			if event.type == QUIT:
				self.close()
				pygame.quit()
				sys.exit()
			
			elif event.type == MOUSEBUTTONDOWN:
				# Obtenir la position du clic
				mousePos = pygame.mouse.get_pos()
				col = mousePos[0] // self.squareSize
				row = 7 - (mousePos[1] // self.squareSize)  # Inverser l'axe y
				
				# Si une case est déjà sélectionnée, tenter un mouvement
				if self.selectedSquare:
					selectedRow, selectedCol = self.selectedSquare
					
					# Vérifier si la case cliquée est un coup possible
					move = None
					for possibleMove in self.possibleMoves:
						if possibleMove.endRow == row and possibleMove.endCol == col:
							move = possibleMove
							break
					
					# Si c'est un coup possible, le jouer
					if move:
						self.selectedSquare = None
						self.possibleMoves = []
						self.lastMove = move
						return move
					
					# Sinon, sélectionner une nouvelle case si elle contient une pièce du joueur actuel
					piece = board.getPiece(row, col)
					if piece and piece.color == board.currentPlayer:
						self.selectedSquare = (row, col)
						self.possibleMoves = self._getLegalMovesForPiece(board, row, col)
					else:
						self.selectedSquare = None
						self.possibleMoves = []
				
				# Sinon, sélectionner une case si elle contient une pièce du joueur actuel
				else:
					piece = board.getPiece(row, col)
					if piece and piece.color == board.currentPlayer:
						self.selectedSquare = (row, col)
						self.possibleMoves = self._getLegalMovesForPiece(board, row, col)
		
		return None
	
	def _getLegalMovesForPiece(self, board: ChessBoard, row: int, col: int) -> List[Move]:
		"""
		Obtient tous les coups légaux pour une pièce donnée.
		
		Args:
			board: Plateau d'échecs
			row: Ligne de la pièce
			col: Colonne de la pièce
			
		Returns:
			List[Move]: Liste des coups légaux pour cette pièce
		"""
		legalMoves = []
		
		allLegalMoves = board.getAllLegalMoves(board.currentPlayer)
		for move in allLegalMoves:
			if move.startRow == row and move.startCol == col:
				legalMoves.append(move)
		
		return legalMoves
	
	def setLastMove(self, move: Move) -> None:
		"""
		Définit le dernier coup joué pour l'affichage.
		
		Args:
			move: Le coup à afficher
		"""
		self.lastMove = move
	
	def close(self) -> None:
		"""Ferme le visualiseur et libère les ressources."""
		if self.interactive:
			pygame.quit()