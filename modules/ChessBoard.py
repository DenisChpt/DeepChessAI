from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import numpy as np

from modules.Move import Move
from modules.PieceType import PieceType



class PieceColor(Enum):
	"""Enumération des couleurs des pièces d'échecs."""
	WHITE = 0
	BLACK = 1


class Piece:
	"""Classe représentant une pièce d'échecs."""
	
	def __init__(self, pieceType: PieceType, pieceColor: PieceColor):
		self.type: PieceType = pieceType
		self.color: PieceColor = pieceColor
		self.hasMoved: bool = False
	
	def __str__(self) -> str:
		"""Représentation string de la pièce."""
		symbols = {
			PieceType.PAWN: 'P',
			PieceType.KNIGHT: 'N',
			PieceType.BISHOP: 'B',
			PieceType.ROOK: 'R',
			PieceType.QUEEN: 'Q',
			PieceType.KING: 'K'
		}
		symbol = symbols[self.type]
		return symbol if self.color == PieceColor.WHITE else symbol.lower()


class ChessBoard:
	"""
	Représentation d'un plateau d'échecs avec la logique de validation des coups.
	Utilise les coordonnées (0,0) pour a1 (coin inférieur gauche, blanc)
	jusqu'à (7,7) pour h8 (coin supérieur droit, noir).
	"""
	
	def __init__(self):
		# Initialiser un plateau vide 8x8
		self.board: List[List[Optional[Piece]]] = [[None for _ in range(8)] for _ in range(8)]
		self.currentPlayer: PieceColor = PieceColor.WHITE
		self.moveHistory: List[Move] = []
		self.kingPositions: Dict[PieceColor, Tuple[int, int]] = {}
		self.capturedPieces: List[Piece] = []
		self.halfMoveClock: int = 0  # Pour la règle des 50 coups
		self.fullMoveNumber: int = 1  # Incrémenté après chaque coup des noirs
		
		# État pour en passant
		self.enPassantTarget: Optional[Tuple[int, int]] = None
		
		self.setupBoard()
	
	def setupBoard(self) -> None:
		"""Initialise le plateau avec les pièces dans leur position de départ."""
		# Placement des pions
		for col in range(8):
			self.board[1][col] = Piece(PieceType.PAWN, PieceColor.WHITE)
			self.board[6][col] = Piece(PieceType.PAWN, PieceColor.BLACK)
		
		# Placement des autres pièces
		# Rangée blanche
		self.board[0][0] = Piece(PieceType.ROOK, PieceColor.WHITE)
		self.board[0][1] = Piece(PieceType.KNIGHT, PieceColor.WHITE)
		self.board[0][2] = Piece(PieceType.BISHOP, PieceColor.WHITE)
		self.board[0][3] = Piece(PieceType.QUEEN, PieceColor.WHITE)
		self.board[0][4] = Piece(PieceType.KING, PieceColor.WHITE)
		self.board[0][5] = Piece(PieceType.BISHOP, PieceColor.WHITE)
		self.board[0][6] = Piece(PieceType.KNIGHT, PieceColor.WHITE)
		self.board[0][7] = Piece(PieceType.ROOK, PieceColor.WHITE)
		
		# Rangée noire
		self.board[7][0] = Piece(PieceType.ROOK, PieceColor.BLACK)
		self.board[7][1] = Piece(PieceType.KNIGHT, PieceColor.BLACK)
		self.board[7][2] = Piece(PieceType.BISHOP, PieceColor.BLACK)
		self.board[7][3] = Piece(PieceType.QUEEN, PieceColor.BLACK)
		self.board[7][4] = Piece(PieceType.KING, PieceColor.BLACK)
		self.board[7][5] = Piece(PieceType.BISHOP, PieceColor.BLACK)
		self.board[7][6] = Piece(PieceType.KNIGHT, PieceColor.BLACK)
		self.board[7][7] = Piece(PieceType.ROOK, PieceColor.BLACK)
		
		# Enregistrement des positions des rois
		self.kingPositions[PieceColor.WHITE] = (0, 4)
		self.kingPositions[PieceColor.BLACK] = (7, 4)
	
	def getPiece(self, row: int, col: int) -> Optional[Piece]:
		"""Retourne la pièce à la position donnée ou None si la case est vide."""
		if 0 <= row < 8 and 0 <= col < 8:
			return self.board[row][col]
		return None
	
	def isEmpty(self, row: int, col: int) -> bool:
		"""Vérifie si une case est vide."""
		return self.getPiece(row, col) is None
	
	def isOpponent(self, row: int, col: int, color: PieceColor) -> bool:
		"""Vérifie si la case contient une pièce adverse."""
		piece = self.getPiece(row, col)
		return piece is not None and piece.color != color
	
	def makeMove(self, move: Move) -> bool:
		"""
		Exécute un coup sur le plateau.
		
		Args:
			move: Le coup à jouer
			
		Returns:
			bool: True si le coup est valide et a été joué, False sinon
		"""
		if not self.isLegalMove(move):
			return False
		
		# Récupérer la pièce à déplacer
		piece = self.getPiece(move.startRow, move.startCol)
		if piece is None:
			return False
		
		# Capturer une pièce si nécessaire
		capturedPiece = self.getPiece(move.endRow, move.endCol)
		if capturedPiece is not None:
			self.capturedPieces.append(capturedPiece)
		
		# Cas spécial: en passant
		if piece.type == PieceType.PAWN and move.endCol != move.startCol and capturedPiece is None:
			# C'est une capture en passant
			captureRow = move.startRow
			captureCol = move.endCol
			capturedPiece = self.getPiece(captureRow, captureCol)
			if capturedPiece is not None:
				self.board[captureRow][captureCol] = None
				self.capturedPieces.append(capturedPiece)
		
		# Mettre à jour l'horloge des demi-coups
		if piece.type == PieceType.PAWN or capturedPiece is not None:
			self.halfMoveClock = 0
		else:
			self.halfMoveClock += 1
		
		# Mettre à jour le numéro de coup complet
		if piece.color == PieceColor.BLACK:
			self.fullMoveNumber += 1
		
		# Cas spécial: roque
		if piece.type == PieceType.KING and abs(move.endCol - move.startCol) == 2:
			# Petit roque (vers la droite)
			if move.endCol > move.startCol:
				rookCol = 7
				newRookCol = move.endCol - 1
			# Grand roque (vers la gauche)
			else:
				rookCol = 0
				newRookCol = move.endCol + 1
			
			# Déplacer la tour
			rook = self.getPiece(move.startRow, rookCol)
			if rook is not None:
				self.board[move.startRow][rookCol] = None
				self.board[move.startRow][newRookCol] = rook
				rook.hasMoved = True
		
		# Mettre à jour la position du roi si nécessaire
		if piece.type == PieceType.KING:
			self.kingPositions[piece.color] = (move.endRow, move.endCol)
		
		# Effectuer le déplacement
		self.board[move.endRow][move.endCol] = piece
		self.board[move.startRow][move.startCol] = None
		
		# Marquer la pièce comme ayant bougé
		piece.hasMoved = True
		
		# Cas spécial: promotion de pion
		if move.promotion is not None and piece.type == PieceType.PAWN:
			if (piece.color == PieceColor.WHITE and move.endRow == 7) or \
			   (piece.color == PieceColor.BLACK and move.endRow == 0):
				self.board[move.endRow][move.endCol] = Piece(move.promotion, piece.color)
		
		# Mettre à jour l'état d'en passant
		self.enPassantTarget = None
		if piece.type == PieceType.PAWN and abs(move.startRow - move.endRow) == 2:
			# Calculer la case cible pour en passant (case derrière le pion)
			passantRow = (move.startRow + move.endRow) // 2
			self.enPassantTarget = (passantRow, move.startCol)
		
		# Ajouter le coup à l'historique
		self.moveHistory.append(move)
		
		# Changer de joueur
		self.currentPlayer = PieceColor.BLACK if self.currentPlayer == PieceColor.WHITE else PieceColor.WHITE
		
		return True
	
	def isLegalMove(self, move: Move) -> bool:
		"""
		Vérifie si un coup est légal.
		
		Args:
			move: Le coup à vérifier
			
		Returns:
			bool: True si le coup est légal, False sinon
		"""
		# Vérifier les limites du plateau
		if not (0 <= move.startRow < 8 and 0 <= move.startCol < 8 and 
				0 <= move.endRow < 8 and 0 <= move.endCol < 8):
			return False
		
		# Vérifier s'il y a une pièce à déplacer
		piece = self.getPiece(move.startRow, move.startCol)
		if piece is None:
			return False
		
		# Vérifier si c'est le tour du joueur
		if piece.color != self.currentPlayer:
			return False
		
		# Vérifier si la destination contient une pièce de même couleur
		destPiece = self.getPiece(move.endRow, move.endCol)
		if destPiece is not None and destPiece.color == piece.color:
			return False
		
		# Vérifier les règles spécifiques à chaque type de pièce
		if piece.type == PieceType.PAWN:
			return self._isLegalPawnMove(move, piece)
		elif piece.type == PieceType.KNIGHT:
			return self._isLegalKnightMove(move)
		elif piece.type == PieceType.BISHOP:
			return self._isLegalBishopMove(move)
		elif piece.type == PieceType.ROOK:
			return self._isLegalRookMove(move)
		elif piece.type == PieceType.QUEEN:
			return self._isLegalQueenMove(move)
		elif piece.type == PieceType.KING:
			return self._isLegalKingMove(move, piece)
		
		return False
	
	def _isLegalPawnMove(self, move: Move, piece: Piece) -> bool:
		"""Vérifie la légalité d'un coup de pion."""
		# Direction du mouvement selon la couleur
		direction = 1 if piece.color == PieceColor.WHITE else -1
		
		# Avancer d'une case
		if move.startCol == move.endCol and move.endRow == move.startRow + direction and self.isEmpty(move.endRow, move.endCol):
			return True
		
		# Avancer de deux cases depuis la position initiale
		if (move.startCol == move.endCol and 
			((piece.color == PieceColor.WHITE and move.startRow == 1 and move.endRow == 3) or
			 (piece.color == PieceColor.BLACK and move.startRow == 6 and move.endRow == 4)) and
			self.isEmpty(move.startRow + direction, move.startCol) and
			self.isEmpty(move.endRow, move.endCol)):
			return True
		
		# Capture en diagonale
		if (abs(move.startCol - move.endCol) == 1 and 
			move.endRow == move.startRow + direction and
			self.isOpponent(move.endRow, move.endCol, piece.color)):
			return True
		
		# Capture en passant
		if (abs(move.startCol - move.endCol) == 1 and 
			move.endRow == move.startRow + direction and
			self.enPassantTarget == (move.endRow, move.endCol)):
			return True
		
		return False
	
	def _isLegalKnightMove(self, move: Move) -> bool:
		"""Vérifie la légalité d'un coup de cavalier."""
		rowDiff = abs(move.endRow - move.startRow)
		colDiff = abs(move.endCol - move.startCol)
		
		# Le cavalier se déplace en L (2 dans une direction, 1 dans l'autre)
		return (rowDiff == 2 and colDiff == 1) or (rowDiff == 1 and colDiff == 2)
	
	def _isLegalBishopMove(self, move: Move) -> bool:
		"""Vérifie la légalité d'un coup de fou."""
		rowDiff = move.endRow - move.startRow
		colDiff = move.endCol - move.startCol
		
		# Le fou se déplace en diagonale
		if abs(rowDiff) != abs(colDiff) or rowDiff == 0:
			return False
		
		# Vérifier si le chemin est dégagé
		rowStep = 1 if rowDiff > 0 else -1
		colStep = 1 if colDiff > 0 else -1
		
		row, col = move.startRow + rowStep, move.startCol + colStep
		while row != move.endRow and col != move.endCol:
			if not self.isEmpty(row, col):
				return False
			row += rowStep
			col += colStep
		
		return True
	
	def _isLegalRookMove(self, move: Move) -> bool:
		"""Vérifie la légalité d'un coup de tour."""
		rowDiff = move.endRow - move.startRow
		colDiff = move.endCol - move.startCol
		
		# La tour se déplace horizontalement ou verticalement
		if rowDiff != 0 and colDiff != 0:
			return False
		
		# Vérifier si le chemin est dégagé
		if rowDiff != 0:
			# Mouvement vertical
			rowStep = 1 if rowDiff > 0 else -1
			row = move.startRow + rowStep
			while row != move.endRow:
				if not self.isEmpty(row, move.startCol):
					return False
				row += rowStep
		else:
			# Mouvement horizontal
			colStep = 1 if colDiff > 0 else -1
			col = move.startCol + colStep
			while col != move.endCol:
				if not self.isEmpty(move.startRow, col):
					return False
				col += colStep
		
		return True
	
	def _isLegalQueenMove(self, move: Move) -> bool:
		"""Vérifie la légalité d'un coup de dame."""
		# La dame se déplace comme un fou ou une tour
		return self._isLegalBishopMove(move) or self._isLegalRookMove(move)
	
	def _isLegalKingMove(self, move: Move, piece: Piece) -> bool:
		"""Vérifie la légalité d'un coup de roi."""
		rowDiff = abs(move.endRow - move.startRow)
		colDiff = abs(move.endCol - move.startCol)
		
		# Mouvement normal du roi (une case dans n'importe quelle direction)
		if rowDiff <= 1 and colDiff <= 1:
			# Vérifier si la nouvelle position serait en échec
			return not self._wouldBeInCheck(move, piece.color)
		
		# Petit roque
		if (not piece.hasMoved and rowDiff == 0 and colDiff == 2 and move.endCol > move.startCol):
			# Vérifier si la tour est présente et n'a pas bougé
			rook = self.getPiece(move.startRow, 7)
			if rook is None or rook.type != PieceType.ROOK or rook.hasMoved:
				return False
			
			# Vérifier si le chemin est dégagé
			if not self.isEmpty(move.startRow, move.startCol + 1) or not self.isEmpty(move.startRow, move.startCol + 2):
				return False
			
			# Vérifier si le roi est en échec, passerait par un échec ou finirait en échec
			if (self.isInCheck(piece.color) or 
				self._wouldBeInCheck(Move(move.startRow, move.startCol, move.startRow, move.startCol + 1), piece.color) or
				self._wouldBeInCheck(move, piece.color)):
				return False
			
			return True
		
		# Grand roque
		if (not piece.hasMoved and rowDiff == 0 and colDiff == 2 and move.endCol < move.startCol):
			# Vérifier si la tour est présente et n'a pas bougé
			rook = self.getPiece(move.startRow, 0)
			if rook is None or rook.type != PieceType.ROOK or rook.hasMoved:
				return False
			
			# Vérifier si le chemin est dégagé
			if (not self.isEmpty(move.startRow, move.startCol - 1) or 
				not self.isEmpty(move.startRow, move.startCol - 2) or 
				not self.isEmpty(move.startRow, move.startCol - 3)):
				return False
			
			# Vérifier si le roi est en échec, passerait par un échec ou finirait en échec
			if (self.isInCheck(piece.color) or 
				self._wouldBeInCheck(Move(move.startRow, move.startCol, move.startRow, move.startCol - 1), piece.color) or
				self._wouldBeInCheck(move, piece.color)):
				return False
			
			return True
		
		return False
	
	def _wouldBeInCheck(self, move: Move, color: PieceColor) -> bool:
		"""
		Vérifie si un coup mettrait le roi en échec.
		
		Args:
			move: Le coup à tester
			color: La couleur du joueur qui joue
			
		Returns:
			bool: True si le coup mettrait le roi en échec, False sinon
		"""
		# Créer une copie temporaire du plateau
		tempBoard = self.copy()
		
		# Effectuer le coup sur la copie
		piece = tempBoard.getPiece(move.startRow, move.startCol)
		tempBoard.board[move.endRow][move.endCol] = piece
		tempBoard.board[move.startRow][move.startCol] = None
		
		# Mettre à jour la position du roi si nécessaire
		if piece and piece.type == PieceType.KING:
			tempBoard.kingPositions[color] = (move.endRow, move.endCol)
		
		# Vérifier si le roi est en échec
		return tempBoard.isInCheck(color)
	
	def isInCheck(self, color: PieceColor) -> bool:
		"""
		Vérifie si le roi de la couleur donnée est en échec.
		
		Args:
			color: La couleur du roi à vérifier
			
		Returns:
			bool: True si le roi est en échec, False sinon
		"""
		kingPos = self.kingPositions.get(color)
		if kingPos is None:
			return False
		
		kingRow, kingCol = kingPos
		
		# Vérifier les attaques des pions
		pawnDir = -1 if color == PieceColor.WHITE else 1
		for colOffset in [-1, 1]:
			row, col = kingRow + pawnDir, kingCol + colOffset
			if 0 <= row < 8 and 0 <= col < 8:
				piece = self.getPiece(row, col)
				if piece and piece.type == PieceType.PAWN and piece.color != color:
					return True
		
		# Vérifier les attaques des cavaliers
		knightOffsets = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
		for rowOffset, colOffset in knightOffsets:
			row, col = kingRow + rowOffset, kingCol + colOffset
			if 0 <= row < 8 and 0 <= col < 8:
				piece = self.getPiece(row, col)
				if piece and piece.type == PieceType.KNIGHT and piece.color != color:
					return True
		
		# Vérifier les attaques en ligne droite (tour et dame)
		straightDirections = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		for rowDir, colDir in straightDirections:
			row, col = kingRow + rowDir, kingCol + colDir
			while 0 <= row < 8 and 0 <= col < 8:
				piece = self.getPiece(row, col)
				if piece is not None:
					if piece.color != color and (piece.type == PieceType.ROOK or piece.type == PieceType.QUEEN):
						return True
					break
				row += rowDir
				col += colDir
		
		# Vérifier les attaques en diagonale (fou et dame)
		diagonalDirections = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
		for rowDir, colDir in diagonalDirections:
			row, col = kingRow + rowDir, kingCol + colDir
			while 0 <= row < 8 and 0 <= col < 8:
				piece = self.getPiece(row, col)
				if piece is not None:
					if piece.color != color and (piece.type == PieceType.BISHOP or piece.type == PieceType.QUEEN):
						return True
					break
				row += rowDir
				col += colDir
		
		# Vérifier les attaques du roi
		kingOffsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
		for rowOffset, colOffset in kingOffsets:
			row, col = kingRow + rowOffset, kingCol + colOffset
			if 0 <= row < 8 and 0 <= col < 8:
				piece = self.getPiece(row, col)
				if piece and piece.type == PieceType.KING and piece.color != color:
					return True
		
		return False
	
	def getAllLegalMoves(self, color: PieceColor) -> List[Move]:
		"""
		Retourne tous les coups légaux pour la couleur donnée.
		
		Args:
			color: La couleur du joueur
			
		Returns:
			List[Move]: Liste des coups légaux
		"""
		legalMoves = []
		
		for row in range(8):
			for col in range(8):
				piece = self.getPiece(row, col)
				if piece is not None and piece.color == color:
					# Générer tous les coups possibles pour cette pièce
					possibleMoves = self._getPossibleMovesForPiece(row, col, piece)
					
					# Filtrer les coups qui mettent le roi en échec
					for move in possibleMoves:
						if not self._wouldBeInCheck(move, color):
							legalMoves.append(move)
		
		return legalMoves
	
	def _getPossibleMovesForPiece(self, row: int, col: int, piece: Piece) -> List[Move]:
		"""
		Génère tous les coups possibles pour une pièce donnée.
		
		Args:
			row: La ligne de la pièce
			col: La colonne de la pièce
			piece: La pièce à déplacer
			
		Returns:
			List[Move]: Liste des coups possibles
		"""
		moves = []
		
		if piece.type == PieceType.PAWN:
			direction = 1 if piece.color == PieceColor.WHITE else -1
			
			# Avancer d'une case
			if 0 <= row + direction < 8 and self.isEmpty(row + direction, col):
				# Vérifier la promotion
				if (piece.color == PieceColor.WHITE and row + direction == 7) or \
				   (piece.color == PieceColor.BLACK and row + direction == 0):
					for promotionType in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
						moves.append(Move(row, col, row + direction, col, promotionType))
				else:
					moves.append(Move(row, col, row + direction, col))
			
			# Avancer de deux cases depuis la position initiale
			if ((piece.color == PieceColor.WHITE and row == 1) or 
				(piece.color == PieceColor.BLACK and row == 6)) and \
			   self.isEmpty(row + direction, col) and \
			   self.isEmpty(row + 2 * direction, col):
				moves.append(Move(row, col, row + 2 * direction, col))
			
			# Captures en diagonale
			for colOffset in [-1, 1]:
				if 0 <= row + direction < 8 and 0 <= col + colOffset < 8:
					if self.isOpponent(row + direction, col + colOffset, piece.color):
						# Vérifier la promotion
						if (piece.color == PieceColor.WHITE and row + direction == 7) or \
						   (piece.color == PieceColor.BLACK and row + direction == 0):
							for promotionType in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
								moves.append(Move(row, col, row + direction, col + colOffset, promotionType))
						else:
							moves.append(Move(row, col, row + direction, col + colOffset))
			
			# Capture en passant
			if self.enPassantTarget:
				enPassantRow, enPassantCol = self.enPassantTarget
				if (enPassantRow == row + direction and 
					abs(enPassantCol - col) == 1):
					moves.append(Move(row, col, enPassantRow, enPassantCol))
		
		elif piece.type == PieceType.KNIGHT:
			knightOffsets = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
			for rowOffset, colOffset in knightOffsets:
				newRow, newCol = row + rowOffset, col + colOffset
				if 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol) or self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
		
		elif piece.type == PieceType.BISHOP:
			diagonalDirections = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
			for rowDir, colDir in diagonalDirections:
				newRow, newCol = row + rowDir, col + colDir
				while 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol):
						moves.append(Move(row, col, newRow, newCol))
						newRow += rowDir
						newCol += colDir
					elif self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
						break
					else:
						break
		
		elif piece.type == PieceType.ROOK:
			straightDirections = [(0, 1), (1, 0), (0, -1), (-1, 0)]
			for rowDir, colDir in straightDirections:
				newRow, newCol = row + rowDir, col + colDir
				while 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol):
						moves.append(Move(row, col, newRow, newCol))
						newRow += rowDir
						newCol += colDir
					elif self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
						break
					else:
						break
		
		elif piece.type == PieceType.QUEEN:
			# La dame combine les mouvements de la tour et du fou
			straightDirections = [(0, 1), (1, 0), (0, -1), (-1, 0)]
			for rowDir, colDir in straightDirections:
				newRow, newCol = row + rowDir, col + colDir
				while 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol):
						moves.append(Move(row, col, newRow, newCol))
						newRow += rowDir
						newCol += colDir
					elif self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
						break
					else:
						break
			
			diagonalDirections = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
			for rowDir, colDir in diagonalDirections:
				newRow, newCol = row + rowDir, col + colDir
				while 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol):
						moves.append(Move(row, col, newRow, newCol))
						newRow += rowDir
						newCol += colDir
					elif self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
						break
					else:
						break
		
		elif piece.type == PieceType.KING:
			# Mouvements normaux du roi
			kingOffsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
			for rowOffset, colOffset in kingOffsets:
				newRow, newCol = row + rowOffset, col + colOffset
				if 0 <= newRow < 8 and 0 <= newCol < 8:
					if self.isEmpty(newRow, newCol) or self.isOpponent(newRow, newCol, piece.color):
						moves.append(Move(row, col, newRow, newCol))
			
			# Petit roque
			if not piece.hasMoved and col == 4:
				rookCol = 7
				rook = self.getPiece(row, rookCol)
				if (rook is not None and rook.type == PieceType.ROOK and 
					rook.color == piece.color and not rook.hasMoved and
					self.isEmpty(row, col + 1) and self.isEmpty(row, col + 2)):
					moves.append(Move(row, col, row, col + 2))
			
			# Grand roque
			if not piece.hasMoved and col == 4:
				rookCol = 0
				rook = self.getPiece(row, rookCol)
				if (rook is not None and rook.type == PieceType.ROOK and 
					rook.color == piece.color and not rook.hasMoved and
					self.isEmpty(row, col - 1) and self.isEmpty(row, col - 2) and
					self.isEmpty(row, col - 3)):
					moves.append(Move(row, col, row, col - 2))
		
		return moves
	
	def isCheckmate(self, color: PieceColor) -> bool:
		"""
		Vérifie si le joueur de la couleur donnée est en échec et mat.
		
		Args:
			color: La couleur du joueur à vérifier
			
		Returns:
			bool: True si le joueur est en échec et mat, False sinon
		"""
		# Vérifier si le roi est en échec
		if not self.isInCheck(color):
			return False
		
		# Vérifier s'il existe un coup légal qui permet d'échapper à l'échec
		legalMoves = self.getAllLegalMoves(color)
		return len(legalMoves) == 0
	
	def isStalemate(self, color: PieceColor) -> bool:
		"""
		Vérifie si le joueur de la couleur donnée est en pat.
		
		Args:
			color: La couleur du joueur à vérifier
			
		Returns:
			bool: True si le joueur est en pat, False sinon
		"""
		# Vérifier si le roi n'est pas en échec
		if self.isInCheck(color):
			return False
		
		# Vérifier s'il n'existe aucun coup légal
		legalMoves = self.getAllLegalMoves(color)
		return len(legalMoves) == 0
	
	def isDraw(self) -> bool:
		"""
		Vérifie si la partie est nulle (pat, répétition ou règle des 50 coups).
		
		Returns:
			bool: True si la partie est nulle, False sinon
		"""
		# Vérifier le pat
		if self.isStalemate(self.currentPlayer):
			return True
		
		# Vérifier la règle des 50 coups
		if self.halfMoveClock >= 100:  # 50 coups complets (100 demi-coups)
			return True
		
		# Vérifier l'insuffisance matérielle
		if self._hasInsufficientMaterial():
			return True
		
		# Vérifier la répétition de position (simplifiée)
		# Note: une implémentation complète nécessiterait de stocker l'historique des positions
		
		return False
	
	def _hasInsufficientMaterial(self) -> bool:
		"""
		Vérifie s'il reste suffisamment de matériel pour gagner.
		
		Returns:
			bool: True s'il y a insuffisance matérielle, False sinon
		"""
		# Compter les pièces
		pieceCount = {
			PieceColor.WHITE: {piece_type: 0 for piece_type in PieceType},
			PieceColor.BLACK: {piece_type: 0 for piece_type in PieceType}
		}
		
		for row in range(8):
			for col in range(8):
				piece = self.getPiece(row, col)
				if piece is not None:
					pieceCount[piece.color][piece.type] += 1
		
		# Roi contre roi
		if (sum(pieceCount[PieceColor.WHITE].values()) == 1 and 
			sum(pieceCount[PieceColor.BLACK].values()) == 1):
			return True
		
		# Roi et cavalier contre roi
		if ((sum(pieceCount[PieceColor.WHITE].values()) == 2 and 
			 pieceCount[PieceColor.WHITE][PieceType.KNIGHT] == 1 and
			 sum(pieceCount[PieceColor.BLACK].values()) == 1) or
			(sum(pieceCount[PieceColor.BLACK].values()) == 2 and 
			 pieceCount[PieceColor.BLACK][PieceType.KNIGHT] == 1 and
			 sum(pieceCount[PieceColor.WHITE].values()) == 1)):
			return True
		
		# Roi et fou contre roi
		if ((sum(pieceCount[PieceColor.WHITE].values()) == 2 and 
			 pieceCount[PieceColor.WHITE][PieceType.BISHOP] == 1 and
			 sum(pieceCount[PieceColor.BLACK].values()) == 1) or
			(sum(pieceCount[PieceColor.BLACK].values()) == 2 and 
			 pieceCount[PieceColor.BLACK][PieceType.BISHOP] == 1 and
			 sum(pieceCount[PieceColor.WHITE].values()) == 1)):
			return True
		
		# Roi et fou contre roi et fou (fous de même couleur)
		if (pieceCount[PieceColor.WHITE][PieceType.BISHOP] == 1 and
			pieceCount[PieceColor.BLACK][PieceType.BISHOP] == 1 and
			sum(pieceCount[PieceColor.WHITE].values()) == 2 and
			sum(pieceCount[PieceColor.BLACK].values()) == 2):
			# Vérifier si les fous sont sur des cases de même couleur
			whiteBishopSquareColor = None
			blackBishopSquareColor = None
			
			for row in range(8):
				for col in range(8):
					piece = self.getPiece(row, col)
					if piece and piece.type == PieceType.BISHOP:
						squareColor = (row + col) % 2
						if piece.color == PieceColor.WHITE:
							whiteBishopSquareColor = squareColor
						else:
							blackBishopSquareColor = squareColor
			
			if whiteBishopSquareColor == blackBishopSquareColor:
				return True
		
		return False
	
	def copy(self) -> ChessBoard:
		"""
		Crée une copie profonde du plateau.
		
		Returns:
			ChessBoard: Une nouvelle instance du plateau avec le même état
		"""
		newBoard = ChessBoard.__new__(ChessBoard)
		newBoard.board = [[None for _ in range(8)] for _ in range(8)]
		
		# Copier les pièces
		for row in range(8):
			for col in range(8):
				piece = self.getPiece(row, col)
				if piece is not None:
					newPiece = Piece(piece.type, piece.color)
					newPiece.hasMoved = piece.hasMoved
					newBoard.board[row][col] = newPiece
		
		# Copier les autres attributs
		newBoard.currentPlayer = self.currentPlayer
		newBoard.moveHistory = self.moveHistory.copy()
		newBoard.kingPositions = self.kingPositions.copy()
		newBoard.capturedPieces = self.capturedPieces.copy()
		newBoard.halfMoveClock = self.halfMoveClock
		newBoard.fullMoveNumber = self.fullMoveNumber
		newBoard.enPassantTarget = self.enPassantTarget
		
		return newBoard
	
	def toBoardArray(self) -> np.ndarray:
		"""
		Convertit le plateau en un tableau numpy codé pour l'apprentissage.
		
		Returns:
			np.ndarray: Un tableau 8x8x12 représentant le plateau
		"""
		# 12 canaux: 6 types de pièces pour chaque couleur
		boardArray = np.zeros((8, 8, 12), dtype=np.float32)
		
		for row in range(8):
			for col in range(8):
				piece = self.getPiece(row, col)
				if piece is not None:
					# Calculer l'indice du canal
					channelIdx = piece.type.value
					if piece.color == PieceColor.BLACK:
						channelIdx += 6
					
					# Marquer la présence de la pièce
					boardArray[row, col, channelIdx] = 1.0
		
		return boardArray
	
	def getGameState(self) -> Dict:
		"""
		Retourne l'état complet du jeu pour l'apprentissage.
		
		Returns:
			Dict: Un dictionnaire contenant les informations sur l'état du jeu
		"""
		boardArray = self.toBoardArray()
		
		# Ajouter un canal pour le joueur actuel
		playerLayer = np.ones((8, 8, 1), dtype=np.float32) if self.currentPlayer == PieceColor.WHITE else np.zeros((8, 8, 1), dtype=np.float32)
		
		# Ajouter des canaux pour les droits de roque
		castlingRights = np.zeros((8, 8, 4), dtype=np.float32)
		# Canal 0: roque côté roi blanc
		# Canal 1: roque côté dame blanc
		# Canal 2: roque côté roi noir
		# Canal 3: roque côté dame noir
		
		whiteKing = self.getPiece(0, 4)
		whiteKingRook = self.getPiece(0, 7)
		whiteQueenRook = self.getPiece(0, 0)
		blackKing = self.getPiece(7, 4)
		blackKingRook = self.getPiece(7, 7)
		blackQueenRook = self.getPiece(7, 0)
		
		if (whiteKing and whiteKing.type == PieceType.KING and not whiteKing.hasMoved and
			whiteKingRook and whiteKingRook.type == PieceType.ROOK and not whiteKingRook.hasMoved):
			castlingRights[:, :, 0] = 1.0
		
		if (whiteKing and whiteKing.type == PieceType.KING and not whiteKing.hasMoved and
			whiteQueenRook and whiteQueenRook.type == PieceType.ROOK and not whiteQueenRook.hasMoved):
			castlingRights[:, :, 1] = 1.0
		
		if (blackKing and blackKing.type == PieceType.KING and not blackKing.hasMoved and
			blackKingRook and blackKingRook.type == PieceType.ROOK and not blackKingRook.hasMoved):
			castlingRights[:, :, 2] = 1.0
		
		if (blackKing and blackKing.type == PieceType.KING and not blackKing.hasMoved and
			blackQueenRook and blackQueenRook.type == PieceType.ROOK and not blackQueenRook.hasMoved):
			castlingRights[:, :, 3] = 1.0
		
		# Ajouter un canal pour la cible en passant
		enPassantLayer = np.zeros((8, 8, 1), dtype=np.float32)
		if self.enPassantTarget:
			row, col = self.enPassantTarget
			enPassantLayer[row, col, 0] = 1.0
		
		# Ajouter un canal pour le compteur de demi-coups (règle des 50 coups)
		halfMoveLayer = np.ones((8, 8, 1), dtype=np.float32) * (self.halfMoveClock / 100.0)
		
		# Combiner tous les canaux
		stateArray = np.concatenate([
			boardArray,
			playerLayer,
			castlingRights,
			enPassantLayer,
			halfMoveLayer
		], axis=2)
		
		return {
			'board_state': stateArray,
			'current_player': self.currentPlayer,
			'legal_moves': self.getAllLegalMoves(self.currentPlayer),
			'is_check': self.isInCheck(self.currentPlayer),
			'is_checkmate': self.isCheckmate(self.currentPlayer),
			'is_stalemate': self.isStalemate(self.currentPlayer),
			'is_draw': self.isDraw(),
			'half_move_clock': self.halfMoveClock,
			'full_move_number': self.fullMoveNumber
		}
	
	def __str__(self) -> str:
		"""Représentation string du plateau d'échecs."""
		result = ""
		for row in range(7, -1, -1):
			result += f"{row + 1} "
			for col in range(8):
				piece = self.getPiece(row, col)
				if piece is None:
					result += ". "
				else:
					result += f"{piece} "
			result += "\n"
		result += "  a b c d e f g h"
		return result