# Chess RL - Jeu d'échecs avec apprentissage par renforcement

Ce projet implémente un jeu d'échecs complet avec un agent d'intelligence artificielle capable d'apprendre à jouer via des techniques d'apprentissage par renforcement profond.

## Caractéristiques

- Moteur d'échecs entièrement fonctionnel avec toutes les règles (incluant le roque, la promotion, etc.)
- Mode visuel avec Pygame pour jouer contre l'IA
- Mode headless pour entraînement rapide
- Deux algorithmes d'apprentissage par renforcement :
  - AlphaZero (algorithme basé sur MCTS et politique/valeur)
  - DQN (Deep Q-Network)
- Possibilité de reprendre l'entraînement
- Surveillance et visualisation des performances d'entraînement

## Prérequis

- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib
- tqdm

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/username/chess-rl.git
cd chess-rl
```

2. Installer les dépendances :
```bash
pip install torch numpy pygame matplotlib tqdm
```

3. (Optionnel) Créer les dossiers pour les ressources et logs :
```bash
mkdir -p assets/pieces checkpoints logs
```

## Structure du projet

Le projet est organisé selon l'architecture suivante :

- `ChessBoard.py` : Représentation du plateau, validation des coups, détection de fin de partie
- `Move.py` : Objet coup et encodage pour le réseau
- `ChessEnv.py` : Environnement d'échecs compatible Gym
- `NeuralNet.py` : Modèle PyTorch (AlphaZero et DQN)
- `Agent.py` : Logique d'apprentissage et choix d'action
- `Trainer.py` : Boucle d'entraînement RL
- `Visualizer.py` : Rendu graphique avec pygame
- `main.py` : CLI principal

## Utilisation

### Entraîner un nouvel agent

#### AlphaZero (approche politique/valeur)

```bash
# Mode headless (entraînement rapide)
python main.py --train --algorithm alphazero --episodes 1000 --headless

# Mode visuel (pour observer l'entraînement)
python main.py --train --algorithm alphazero --episodes 1000 --visual
```

#### DQN (Deep Q-Network)

```bash
# Mode headless (entraînement rapide)
python main.py --train --algorithm dqn --episodes 1000 --headless

# Mode visuel (pour observer l'entraînement)
python main.py --train --algorithm dqn --episodes 1000 --visual
```

### Continuer l'entraînement d'un agent existant

```bash
python main.py --continue-training --algorithm alphazero --model-path checkpoints/model_episode_500.pt --episodes 500
```

### Jouer contre un agent entraîné

```bash
# Jouer avec les blancs (par défaut)
python main.py --play --algorithm alphazero --model-path checkpoints/model_final.pt --visual

# Jouer avec les noirs
python main.py --play --algorithm alphazero --model-path checkpoints/model_final.pt --visual --play-as black

# Jouer avec une couleur aléatoire
python main.py --play --algorithm alphazero --model-path checkpoints/model_final.pt --visual --play-as random
```

### Évaluer un agent entraîné

```bash
python main.py --evaluate --algorithm alphazero --model-path checkpoints/model_final.pt
```

## Comprendre le code

### Représentation du plateau

Le plateau est représenté par une grille 8x8 avec des coordonnées (0,0) pour a1 (coin inférieur gauche, blanc) jusqu'à (7,7) pour h8 (coin supérieur droit, noir).

Pour l'apprentissage, le plateau est encodé en un tenseur avec :
- 12 canaux pour les pièces (6 types × 2 couleurs)
- 1 canal pour le joueur actuel
- 4 canaux pour les droits de roque
- 1 canal pour la cible en passant
- 1 canal pour le compteur de demi-coups

### Algorithmes d'apprentissage

#### AlphaZero (simplifié)

L'implémentation s'inspire de l'algorithme AlphaZero avec :
- Un réseau neuronal à double tête (politique et valeur)
- Recherche Monte Carlo Tree Search (MCTS) pour améliorer la politique
- Auto-jeu pour générer des données d'entraînement
- Apprentissage supervisé à partir des parties générées

#### DQN (Deep Q-Network)

Une implémentation classique de DQN avec :
- Tampon de rejeu pour briser les corrélations temporelles
- Réseau cible pour la stabilité d'apprentissage
- Exploration ε-greedy avec décroissance
- Double DQN pour réduire la surestimation des valeurs Q

## Réglage de l'apprentissage

Les performances d'apprentissage peuvent être ajustées via plusieurs paramètres :

### Pour AlphaZero
- Nombre de simulations MCTS
- Température d'exploration
- Taille du lot d'auto-jeu
- Profondeur du réseau

### Pour DQN
- Taux d'apprentissage
- Facteur de décroissance ε
- Fréquence de mise à jour du réseau cible
- Taille du tampon de rejeu

## Tests unitaires

Des tests unitaires sont fournis pour valider le moteur d'échecs et l'apprentissage :

```bash
# Exécuter tous les tests
pytest

# Exécuter des tests spécifiques
pytest tests/test_chess_board.py
pytest tests/test_chess_env.py
```

## Visualisation des résultats

Les statistiques d'entraînement sont automatiquement enregistrées dans le dossier `logs/`. Vous pouvez visualiser les courbes d'apprentissage, récompenses et taux de victoire avec matplotlib :

```bash
# Ouvrir les graphiques générés pendant l'entraînement
cd logs/plots
```

## Performances

Les deux algorithmes ont des caractéristiques différentes :

- **AlphaZero** : Apprend une stratégie plus sophistiquée grâce au MCTS, mais nécessite plus de calcul et de temps d'entraînement.
- **DQN** : Converge plus rapidement, mais peut avoir du mal à planifier sur le long terme dans un jeu complexe comme les échecs.

Avec une configuration matérielle modeste (CPU multi-cœurs ou GPU d'entrée de gamme), l'entraînement peut prendre de quelques heures à plusieurs jours selon la profondeur d'apprentissage souhaitée.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request pour améliorer le projet.

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Remerciements

- Inspiration de l'architecture AlphaZero de DeepMind
- Les bibliothèques PyTorch, Pygame et autres projets open-source qui ont rendu ce projet possible