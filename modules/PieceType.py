from enum import Enum

class PieceType(Enum):
	"""Enumération des types de pièces d'échecs."""
	PAWN = 0
	KNIGHT = 1
	BISHOP = 2
	ROOK = 3
	QUEEN = 4
	KING = 5