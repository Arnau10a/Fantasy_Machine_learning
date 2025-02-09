# Fantasy Machine Learning

Aquest projecte implementa un agent d'aprenentatge per reforç (Reinforcement Learning) que aprèn a gestionar un equip de fantasy football. L'agent pren decisions sobre compres i vendes de jugadors per maximitzar els punts i el valor de l'equip al llarg d'una temporada.

## Descripció

El sistema utilitza Deep Q-Learning (DQN) per aprendre estratègies òptimes de gestió d'equip, tenint en compte:
- Puntuacions dels jugadors
- Preus de mercat
- Forma dels jugadors
- Pressupost disponible
- Rendiment històric

## Característiques Principals

- Simulació completa d'una temporada de fantasy football
- Sistema de transferències realista
- Actualització dinàmica de preus basada en el rendiment
- Visualització detallada de resultats i estadístiques
- Entrenament amb Deep Q-Learning

## Requisits

```bash
pip install -r requirements.txt
```

Dependències principals:
- gymnasium
- stable-baselines3
- numpy
- matplotlib

## Estructura del Projecte

- `FantasyEnv.py`: Implementació de l'entorn de simulació utilitzant Gymnasium
- `train_fantasy.py`: Script per entrenar el model DQN
- `visualitzar_temporada.py`: Eina per visualitzar i analitzar els resultats d'una temporada

## Ús

1. **Entrenar el model**:
```bash
python train_fantasy.py
```

2. **Visualitzar una temporada**:
```bash
python visualitzar_temporada.py
```

## Visualitzacions

El sistema genera dos tipus de gràfics:
- `resultats_temporada.png`: Mostra l'evolució dels punts, beneficis i pressupost
- `resultats_jugadors.png`: Visualitza l'evolució dels preus, forma i mitjana de punts dels jugadors

## Llicència

Aquest projecte està sota la llicència MIT.