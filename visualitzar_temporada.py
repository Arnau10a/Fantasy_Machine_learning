import gymnasium as gym
from stable_baselines3 import DQN, PPO
import numpy as np
import matplotlib.pyplot as plt

# Registrem l'entorn
gym.register(
    id='Fantasy-v0',
    entry_point='FantasyEnv:FantasyEnv'
)

# Creem l'entorn
env = gym.make('Fantasy-v0')

def visualize_season(model):
    obs, info = env.reset()
    total_reward = 0
    
    # Llistes per guardar l'històric
    historic_punts = []
    historic_beneficis = []
    historic_pressupost = []
    historic_jugadors = {i: {'preus': [], 'formes': [], 'mitjanes': []} for i in range(11)}
    jornades = []
    
    for jornada in range(38):
        action, _states = model.predict(obs, deterministic=True)
        jugador_vendre = int(action) // 13
        jugador_comprar = int(action) % 13
        
        print(f"\nJornada {jornada + 1}")
        print(f"Recomanació de transferència:")
        print(f"Vendre jugador {jugador_vendre}:")
        env.render_player(jugador_vendre)
        if jugador_comprar == 12:
            print("No comprar cap jugador")
        else:
            print(f"Comprar jugador {jugador_comprar} del mercat")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Guardem les dades per als gràfics
        historic_punts.append(info['punts_jornada'])
        historic_beneficis.append(reward)
        historic_pressupost.append(info['pressupost'])
        jornades.append(jornada + 1)
        
        # Guardem les dades dels jugadors
        for i, jugador in enumerate(env.state['jugadors']):
            historic_jugadors[i]['preus'].append(jugador['preu'])
            historic_jugadors[i]['formes'].append(jugador['forma'])
            historic_jugadors[i]['mitjanes'].append(jugador['mitjana_punts'])
        
        print(f"\nResultats de la jornada:")
        print(f"Benefici/Pèrdua: {reward:,.2f}€")
        print(f"Punts jornada: {info['punts_jornada']:.2f}")
        env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nBenefici total temporada: {total_reward:,.2f}€")
    
    # Creem els gràfics en dues figures separades
    # Primera figura: gràfics generals
    plt.figure(figsize=(15, 10))
    
    # Gràfic de punts per jornada
    plt.subplot(3, 1, 1)
    plt.plot(jornades, historic_punts, 'b-', label='Punts per jornada')
    plt.fill_between(jornades, historic_punts, alpha=0.2)
    plt.title('Punts per Jornada')
    plt.xlabel('Jornada')
    plt.ylabel('Punts')
    plt.grid(True)
    plt.legend()
    
    # Gràfic de beneficis per jornada
    plt.subplot(3, 1, 2)
    plt.plot(jornades, historic_beneficis, 'g-', label='Benefici per jornada')
    plt.fill_between(jornades, historic_beneficis, alpha=0.2)
    plt.title('Benefici per Jornada')
    plt.xlabel('Jornada')
    plt.ylabel('Benefici (€)')
    plt.grid(True)
    plt.legend()
    
    # Gràfic de pressupost acumulat
    plt.subplot(3, 1, 3)
    plt.plot(jornades, historic_pressupost, 'r-', label='Pressupost disponible')
    plt.fill_between(jornades, historic_pressupost, alpha=0.2)
    plt.title('Evolució del Pressupost')
    plt.xlabel('Jornada')
    plt.ylabel('Pressupost (€)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resultats_temporada.png')
    plt.close()
    
    # Segona figura: gràfics dels jugadors
    plt.figure(figsize=(15, 15))
    
    # Gràfic de preus dels jugadors
    plt.subplot(3, 1, 1)
    for i in range(11):
        plt.plot(jornades, historic_jugadors[i]['preus'], label=f'Jugador {i}')
    plt.title('Evolució dels Preus dels Jugadors')
    plt.xlabel('Jornada')
    plt.ylabel('Preu (€)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gràfic de forma dels jugadors
    plt.subplot(3, 1, 2)
    for i in range(11):
        plt.plot(jornades, historic_jugadors[i]['formes'], label=f'Jugador {i}')
    plt.title('Evolució de la Forma dels Jugadors')
    plt.xlabel('Jornada')
    plt.ylabel('Forma (0-10)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gràfic de mitjana de punts dels jugadors
    plt.subplot(3, 1, 3)
    for i in range(11):
        plt.plot(jornades, historic_jugadors[i]['mitjanes'], label=f'Jugador {i}')
    plt.title('Evolució de la mitjana de punts dels jugadors')
    plt.xlabel('Jornada')
    plt.ylabel('Mitjana de punts')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('resultats_jugadors.png')
    plt.close()
    
    print("\nGràfics guardats a 'resultats_temporada.png' i 'resultats_jugadors.png'")

if __name__ == "__main__":
    print("Carregant model entrenat...")
    try:
        # Intenta carregar PPO primer
        model = PPO.load("fantasy_ppo")
        print("Model PPO carregat correctament")
    except:
        try:
            # Si no troba PPO, intenta carregar DQN
            model = DQN.load("fantasy_dqn")
            print("Model DQN carregat correctament")
        except:
            print("No s'ha trobat cap model entrenat. Has d'entrenar primer.")
            exit()
    
    print("\nVisualitzant una temporada completa amb el model carregat...")
    visualize_season(model) 