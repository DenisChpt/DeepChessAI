from __future__ import annotations
import argparse
import os
import torch
import sys
from typing import Dict, Optional, Any

from modules.ChessEnv import ChessEnv
from modules.ChessBoard import PieceColor
from modules.Agent import ChessAgent, DQNAgent
from modules.Trainer import Trainer
from modules.NeuralNet import ChessNet, QNetwork


def parseArguments() -> Dict[str, Any]:
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Dict[str, Any]: Dictionnaire des arguments parsés
	"""
	parser = argparse.ArgumentParser(description="Chess RL - Apprentissage par renforcement aux échecs")
	
	# Mode d'exécution
	parser.add_argument('--headless', action='store_true', help="Mode sans interface graphique pour un entraînement rapide")
	parser.add_argument('--visual', action='store_true', help="Mode avec interface graphique (pygame)")
	
	# Actions principales
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train', action='store_true', help="Entraîner un nouvel agent")
	group.add_argument('--continue-training', action='store_true', help="Continuer l'entraînement d'un agent existant")
	group.add_argument('--play', action='store_true', help="Jouer contre un agent entraîné")
	group.add_argument('--evaluate', action='store_true', help="Évaluer un agent entraîné")
	
	# Paramètres de l'algorithme
	parser.add_argument('--algorithm', choices=['alphazero', 'dqn'], default='alphazero', 
						help="Algorithme d'apprentissage (alphazero ou dqn)")
	
	# Paramètres d'entraînement
	parser.add_argument('--episodes', type=int, default=1000, help="Nombre d'épisodes d'entraînement")
	parser.add_argument('--batch-size', type=int, default=32, help="Taille du lot d'entraînement")
	
	# Paramètres du modèle
	parser.add_argument('--model-path', type=str, help="Chemin vers un modèle pré-entraîné")
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help="Répertoire des points de contrôle")
	parser.add_argument('--log-dir', type=str, default='logs', help="Répertoire des logs")
	
	# Paramètres pour jouer
	parser.add_argument('--play-as', choices=['white', 'black', 'random'], default='white', 
						help="Couleur avec laquelle jouer contre l'IA")
	
	# Autres paramètres
	parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', 
						help="Périphérique d'exécution (cpu, cuda, auto)")
	
	args = parser.parse_args()
	
	# Validation
	if not (args.train or args.continue_training or args.play or args.evaluate):
		parser.error("Vous devez spécifier une action: --train, --continue-training, --play ou --evaluate")
	
	if args.continue_training and not args.model_path:
		parser.error("--continue-training nécessite --model-path")
	
	if args.play and not args.model_path:
		parser.error("--play nécessite --model-path")
	
	if args.evaluate and not args.model_path:
		parser.error("--evaluate nécessite --model-path")
	
	return vars(args)


def main() -> None:
	"""Point d'entrée principal du programme."""
	# Parser les arguments
	args = parseArguments()
	
	# Déterminer le mode headless/visual
	headless = args['headless'] and not args['visual']
	
	# Créer les répertoires si nécessaires
	os.makedirs(args['checkpoint_dir'], exist_ok=True)
	os.makedirs(args['log_dir'], exist_ok=True)
	
	# Initialiser le Trainer
	trainer = Trainer(
		checkpointDir=args['checkpoint_dir'],
		logDir=args['log_dir'],
		headless=headless
	)
	
	# Initialiser l'agent
	agent = None
	
	if args['algorithm'] == 'alphazero':
		if args['model_path'] and (args['continue_training'] or args['play'] or args['evaluate']):
			# Charger un modèle existant
			model = ChessNet()
			try:
				model.loadModel(args['model_path'])
				print(f"Modèle chargé depuis {args['model_path']}")
			except Exception as e:
				print(f"Erreur lors du chargement du modèle: {e}")
				sys.exit(1)
				
			agent = ChessAgent(model=model, device=args['device'])
		else:
			# Créer un nouvel agent
			agent = ChessAgent(device=args['device'])
	
	elif args['algorithm'] == 'dqn':
		if args['model_path'] and (args['continue_training'] or args['play'] or args['evaluate']):
			# Charger un modèle existant
			agent = DQNAgent(device=args['device'])
			try:
				agent.loadModel(args['model_path'])
				print(f"Modèle DQN chargé depuis {args['model_path']}")
			except Exception as e:
				print(f"Erreur lors du chargement du modèle DQN: {e}")
				sys.exit(1)
		else:
			# Créer un nouvel agent DQN
			agent = DQNAgent(device=args['device'])
	
	# Exécuter l'action demandée
	if args['train'] or args['continue_training']:
		print(f"Démarrage de l'entraînement avec l'algorithme {args['algorithm']}...")
		
		if args['algorithm'] == 'alphazero':
			trainer.trainAlphaZero(
				agent=agent,
				numEpisodes=args['episodes'],
				selfPlayBatchSize=args['batch_size'],
				renderFrequency=0 if headless else 50
			)
		else:  # DQN
			trainer.trainDQN(
				agent=agent,
				numEpisodes=args['episodes'],
				renderFrequency=0 if headless else 50
			)
		
		print("Entraînement terminé!")
	
	elif args['play']:
		print(f"Jouer contre l'agent entraîné (algorithme: {args['algorithm']})...")
		
		# Déterminer la couleur du joueur
		if args['play_as'] == 'white':
			playerColor = PieceColor.WHITE
		elif args['play_as'] == 'black':
			playerColor = PieceColor.BLACK
		else:  # random
			import random
			playerColor = random.choice([PieceColor.WHITE, PieceColor.BLACK])
		
		# Jouer contre l'agent
		trainer.playAgainstHuman(agent, playerColor)
	
	elif args['evaluate']:
		print(f"Évaluation de l'agent (algorithme: {args['algorithm']})...")
		
		# Créer l'environnement d'évaluation
		evalEnv = ChessEnv(render_mode=None if headless else 'human')
		
		# Évaluer l'agent
		evalStats = trainer.evaluateAgent(agent, evalEnv, numEpisodes=50)
		
		print("\nRésultats de l'évaluation:")
		print(f"Taux de victoire: {evalStats['win_rate']:.2%}")
		print(f"Taux de nul: {evalStats['draw_rate']:.2%}")
		print(f"Taux d'échec et mat: {evalStats['checkmate_rate']:.2%}")


if __name__ == "__main__":
	main()