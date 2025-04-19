from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np

from modules.PieceType import PieceType


class Move:
	"""
	Représente un coup aux échecs.
	Coordonnées: (0,0) = a1 (coin inférieur gauche, blanc) à (7,7) = h8 (coin supérieur droit, noir)
	"""
	
	def __init__(self, startRow: int, startCol: int, endRow: int, endCol: int, 
				 promotion: Optional[PieceType] = None):
		"""
		Initialise un coup.
		
		Args:
			startRow: Ligne de départ (0-7)
			startCol: Colonne de départ (0-7)
			endRow: Ligne d'arrivée (0-7)
			endCol: Colonne d'arrivée (0-7)
			promotion: Type de pièce pour la promotion d'un pion (optionnel)
		"""
		self.startRow: int = startRow
		self.startCol: int = startCol
		self.endRow: int = endRow
		self.endCol: int = endCol
		self.promotion: Optional[PieceType] = promotion
	
	@classmethod
	def fromAlgebraic(cls, notation: str) -> Move:
		"""
		Crée un coup à partir de la notation algébrique standard (e.g., 'e2e4', 'e7e8q').
		
		Args:
			notation: Coup en notation algébrique
			
		Returns:
			Move: L'objet Move correspondant
		"""
		if len(notation) < 4:
			raise ValueError(f"Notation de coup invalide: {notation}")
		
		# Conversion des colonnes (a-h -> 0-7)
		startCol = ord(notation[0]) - ord('a')
		endCol = ord(notation[2]) - ord('a')
		
		# Conversion des lignes (1-8 -> 0-7)
		startRow = int(notation[1]) - 1
		endRow = int(notation[3]) - 1
		
		# Vérifier les limites
		if not (0 <= startCol < 8 and 0 <= startRow < 8 and 0 <= endCol < 8 and 0 <= endRow < 8):
			raise ValueError(f"Coordonnées hors limites dans: {notation}")
		
		# Promotion (optionnelle)
		promotion = None
		if len(notation) > 4:
			promotionChar = notation[4].lower()
			if promotionChar == 'q':
				promotion = PieceType.QUEEN
			elif promotionChar == 'r':
				promotion = PieceType.ROOK
			elif promotionChar == 'b':
				promotion = PieceType.BISHOP
			elif promotionChar == 'n':
				promotion = PieceType.KNIGHT
		
		return cls(startRow, startCol, endRow, endCol, promotion)
	
	def toAlgebraic(self) -> str:
		"""
		Convertit le coup en notation algébrique standard.
		
		Returns:
			str: Le coup en notation algébrique
		"""
		# Conversion des colonnes (0-7 -> a-h)
		startCol = chr(self.startCol + ord('a'))
		endCol = chr(self.endCol + ord('a'))
		
		# Conversion des lignes (0-7 -> 1-8)
		startRow = str(self.startRow + 1)
		endRow = str(self.endRow + 1)
		
		notation = startCol + startRow + endCol + endRow
		
		# Ajouter la promotion si nécessaire
		if self.promotion is not None:
			promotionMap = {
				PieceType.QUEEN: 'q',
				PieceType.ROOK: 'r',
				PieceType.BISHOP: 'b',
				PieceType.KNIGHT: 'n'
			}
			notation += promotionMap.get(self.promotion, '')
		
		return notation
	
	def toMoveIndex(self) -> int:
		"""
		Convertit le coup en un indice unique pour l'apprentissage par réseau neuronal.
		L'espace total des coups possibles aux échecs est de 64*64*4 = 16384,
		mais beaucoup de ces coups sont illégaux.
		
		Returns:
			int: Indice unique du coup dans l'espace vectoriel
		"""
		# Indice de base pour le déplacement (startRow, startCol) -> (endRow, endCol)
		baseIndex = self.startRow * 8 * 64 + self.startCol * 64 + self.endRow * 8 + self.endCol
		
		# Ajouter un offset pour la promotion
		if self.promotion is not None:
			# Utiliser des indices différents pour chaque type de promotion
			promotionOffset = {
				PieceType.QUEEN: 0,
				PieceType.ROOK: 1,
				PieceType.BISHOP: 2,
				PieceType.KNIGHT: 3
			}
			return baseIndex + promotionOffset.get(self.promotion, 0) * 4096
		
		return baseIndex
	
	@classmethod
	def fromMoveIndex(cls, index: int) -> Move:
		"""
		Crée un coup à partir d'un indice.
		
		Args:
			index: Indice unique du coup
			
		Returns:
			Move: L'objet Move correspondant
		"""
		# Déterminer s'il s'agit d'une promotion
		promotionType = None
		if index >= 4096:
			promotionIndex = index // 4096
			index %= 4096
			
			promotionTypes = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
			if 0 <= promotionIndex < len(promotionTypes):
				promotionType = promotionTypes[promotionIndex]
		
		# Extraire les coordonnées
		startRow = index // (8 * 64)
		index %= (8 * 64)
		startCol = index // 64
		index %= 64
		endRow = index // 8
		endCol = index % 8
		
		return cls(startRow, startCol, endRow, endCol, promotionType)
	
	def toOneHot(self, shape: Tuple[int, int, int, int] = (8, 8, 8, 8)) -> np.ndarray:
		"""
		Convertit le coup en représentation one-hot pour l'apprentissage.
		
		Args:
			shape: Forme du tenseur de sortie (start_row, start_col, end_row, end_col)
			
		Returns:
			np.ndarray: Représentation one-hot du coup
		"""
		# Créer un tenseur de zéros
		moveOneHot = np.zeros(shape, dtype=np.float32)
		
		# Marquer le coup avec un 1
		moveOneHot[self.startRow, self.startCol, self.endRow, self.endCol] = 1.0
		
		return moveOneHot
	
	@classmethod
	def fromOneHot(cls, oneHot: np.ndarray) -> Move:
		"""
		Crée un coup à partir d'une représentation one-hot.
		
		Args:
			oneHot: Représentation one-hot du coup
			
		Returns:
			Move: L'objet Move correspondant
		"""
		# Trouver l'indice du 1 dans le tenseur
		indices = np.where(oneHot > 0.5)
		if len(indices[0]) == 0:
			raise ValueError("Le tenseur one-hot ne contient pas de valeur 1")
		
		startRow, startCol, endRow, endCol = indices[0][0], indices[1][0], indices[2][0], indices[3][0]
		
		# Pour simplifier, on ne gère pas la promotion ici (nécessiterait une dimension supplémentaire)
		return cls(int(startRow), int(startCol), int(endRow), int(endCol))
	
	def toPolicyVector(self, boardSize: int = 8) -> np.ndarray:
		"""
		Convertit le coup en vecteur de politique pour l'apprentissage.
		
		Args:
			boardSize: Taille du plateau (typiquement 8)
			
		Returns:
			np.ndarray: Vecteur représentant la politique (probabilités des actions)
		"""
		# Calculer l'espace total des actions: départ x arrivée + promotions
		actionSpace = boardSize * boardSize * boardSize * boardSize
		policyVector = np.zeros(actionSpace, dtype=np.float32)
		
		# Indice de base pour ce coup
		baseIndex = self.startRow * boardSize**3 + self.startCol * boardSize**2 + self.endRow * boardSize + self.endCol
		
		# Mettre 1.0 à l'indice correspondant
		policyVector[baseIndex] = 1.0
		
		return policyVector
	
	@classmethod
	def fromPolicyVector(cls, policy: np.ndarray, boardSize: int = 8) -> Move:
		"""
		Crée un coup à partir d'un vecteur de politique.
		
		Args:
			policy: Vecteur de politique (probabilités des actions)
			boardSize: Taille du plateau (typiquement 8)
			
		Returns:
			Move: L'objet Move correspondant à l'action la plus probable
		"""
		# Trouver l'indice de l'action la plus probable
		actionIndex = np.argmax(policy)
		
		# Extraire les coordonnées
		startRow = actionIndex // boardSize**3
		actionIndex %= boardSize**3
		startCol = actionIndex // boardSize**2
		actionIndex %= boardSize**2
		endRow = actionIndex // boardSize
		endCol = actionIndex % boardSize
		
		return cls(int(startRow), int(startCol), int(endRow), int(endCol))
	
	def __eq__(self, other: object) -> bool:
		"""Vérifie l'égalité avec un autre coup."""
		if not isinstance(other, Move):
			return False
		return (self.startRow == other.startRow and
				self.startCol == other.startCol and
				self.endRow == other.endRow and
				self.endCol == other.endCol and
				self.promotion == other.promotion)
	
	def __str__(self) -> str:
		"""Représentation string du coup en notation algébrique."""
		return self.toAlgebraic()
	
	def __repr__(self) -> str:
		"""Représentation détaillée du coup."""
		return f"Move({self.startRow}, {self.startCol}, {self.endRow}, {self.endCol}, {self.promotion})"


def getMoveList() -> List[Move]:
	"""
	Génère une liste de tous les coups possibles aux échecs, y compris les illégaux.
	Utile pour créer un mapping pour le réseau neuronal.
	
	Returns:
		List[Move]: Liste de tous les coups possibles
	"""
	moves = []
	
	# Mouvements standards (sans promotion)
	for startRow in range(8):
		for startCol in range(8):
			for endRow in range(8):
				for endCol in range(8):
					# Exclure les coups qui commencent et terminent sur la même case
					if startRow != endRow or startCol != endCol:
						moves.append(Move(startRow, startCol, endRow, endCol))
	
	# Mouvements avec promotion
	for col in range(8):
		# Promotions pour les pions blancs (de la 7e à la 8e rangée)
		for endCol in range(8):
			for promotion in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
				moves.append(Move(6, col, 7, endCol, promotion))
		
		# Promotions pour les pions noirs (de la 2e à la 1ère rangée)
		for endCol in range(8):
			for promotion in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
				moves.append(Move(1, col, 0, endCol, promotion))
	
	return moves