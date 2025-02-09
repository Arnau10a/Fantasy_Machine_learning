import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FantasyEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        
        # Normalitzem l'espai d'observació per millorar l'aprenentatge del DQN
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(44,),
            dtype=np.float32
        )
        
        # Mantenim l'espai d'accions discret per DQN
        self.action_space = gym.spaces.Discrete(11 * 13)
        
        # Constants per normalització
        self.MAX_PREU = 150_000_000
        self.MAX_PUNTS = 25
        self.MAX_FORMA = 10
        self.MAX_PRESSUPOST = 200_000_000
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Inicialitzem l'estat del joc
        self.jornada_actual = 0
        self.diners_disponibles = 100_000_000  # 100M€ inicial
        self.equip_actual = self._inicialitzar_equip()
        
        # Inicialitzem l'estat directament amb els diccionaris
        self.state = {
            'jornada': 0,
            'pressupost': self.diners_disponibles,
            'punts_totals': 0,
            'jugadors': self.equip_actual  # Ja són diccionaris, no cal convertir-los
        }
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        jugador_vendre = action // 13 # 
        jugador_comprar = action % 13
        
        # Simulem la jornada primer
        punts_jornada = self._simular_jornada()
        self._actualitzar_valors_jugadors()
        
        # Processem la transferència
        benefici_transferencia = 0
        if jugador_vendre < 11:
            if jugador_comprar == 12:
                benefici_transferencia = self._processar_venda(jugador_vendre)
            else:
                benefici_transferencia = self._processar_transferencia(jugador_vendre, jugador_comprar)
        
        # Calculem la recompensa
        reward = self._calcular_reward(punts_jornada, benefici_transferencia)
        
        # Actualitzem l'estat
        self.state['punts_totals'] += punts_jornada
        self.state['jornada'] += 1
        
        # Comprovem si hem acabat
        terminated = self.state['jornada'] >= 38
        truncated = False
        
        # Bonus final si acabem amb èxit
        if terminated and self.state['pressupost'] > 0:
            reward += (self.state['pressupost'] / self.MAX_PRESSUPOST) * 0.5
        
        observation = self._get_observation()
        info = {
            'punts_jornada': punts_jornada,
            'pressupost': self.state['pressupost']
        }
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Converteix l'estat actual en un array d'observació normalitzat"""
        obs = np.zeros(44, dtype=np.float32)
        
        for i, jugador in enumerate(self.state['jugadors']):
            base_idx = i * 4
            obs[base_idx:base_idx + 4] = [
                jugador['preu'] / self.MAX_PREU,  # Normalitzem preu
                jugador['punts_anteriors'] / self.MAX_PUNTS,  # Normalitzem punts
                jugador['mitjana_punts'] / self.MAX_PUNTS,  # Normalitzem mitjana
                jugador['forma'] / self.MAX_FORMA  # Normalitzem forma
            ]
        
        return obs

    def _inicialitzar_jugadors(self):
        jugadors = []
        for _ in range(18):
            jugador = {
                'preu': np.random.randint(1000000, 15000000),
                'mitjana_punts': np.random.uniform(0, 8),
                'forma': np.random.uniform(0, 10),
                'rival_dificultat': np.random.randint(1, 5),
                'titular': True if len(jugadors) < 11 else False
            }
            jugadors.append(jugador)
        return jugadors

    def _actualitzar_valors_jugadors(self):
        for jugador in self.state['jugadors']:
            if jugador['titular']:
                # Calculem el rendiment relatiu a la mitjana esperada
                rendiment = (jugador['mitjana_punts'] * jugador['forma']) / 15
                rendiment_esperat = 8  # Punt d'equilibri
                
                # La variació base pot ser positiva o negativa
                variacio_base = (rendiment - rendiment_esperat) * 0.02
                
                # Afegim un component aleatori per més variabilitat
                variacio_aleatoria = np.random.normal(0, 0.01)  # 1% de variabilitat
                
                # Calculem la variació total
                variacio_total = (variacio_base + variacio_aleatoria) * jugador['preu']
                
                # Limitem la variació màxima (tant positiva com negativa)
                variacio_maxima = jugador['preu'] * 0.10
                variacio_final = np.clip(variacio_total, -variacio_maxima, variacio_maxima)
                
                # Actualitzem el preu
                nou_preu = jugador['preu'] + variacio_final
                jugador['preu'] = np.clip(nou_preu, 1_000_000, self.MAX_PREU)

    def _processar_transferencia(self, index_vendre, index_comprar):
        """Processa una transferència completa (venda + compra)"""
        jugador_actual = self.state['jugadors'][index_vendre]
        jugador_mercat = self._obtenir_jugador_mercat(index_comprar)
        
        # Calculem el benefici de la venda
        benefici_venda = jugador_actual['preu']
        
        # Comprovem si podem permetre'ns el jugador nou
        if jugador_mercat['preu'] > self.state['pressupost'] + benefici_venda:
            return -1000000  # Penalització per intentar una compra impossible
        
        # Actualitzem el pressupost
        self.state['pressupost'] = self.state['pressupost'] + benefici_venda - jugador_mercat['preu']
        
        # Actualitzem l'equip
        self.state['jugadors'][index_vendre] = jugador_mercat
        
        return benefici_venda - jugador_mercat['preu']

    def _generar_jugador(self, titular=False, preu=None):
        """Genera un nou jugador amb característiques correlacionades amb el preu"""
        if preu is None:
            preu = np.random.randint(1000000, 15000000)
        
        # Ajustem el factor de qualitat per ser més conservador
        factor_qualitat = (np.log(preu) - np.log(1000000)) / (np.log(15000000) - np.log(1000000))
        
        # Reduïm els rangs de variació
        mitjana_base = 4 + (factor_qualitat * 8)  # Entre 4 i 12 punts
        forma_base = 5 + (factor_qualitat * 4)    # Entre 5 i 9 de forma
        
        # Reduïm la variabilitat aleatòria
        mitjana = np.clip(np.random.normal(mitjana_base, 1.5), 0, 15)
        forma = np.clip(np.random.normal(forma_base, 0.8), 0, 10)
        
        return {
            'preu': preu,
            'punts_anteriors': 0,
            'mitjana_punts': mitjana,
            'prediccio_seguent': mitjana + np.random.normal(0, 0.5),
            'forma': forma,
            'rival_dificultat': np.random.randint(1, 5),
            'titular': titular
        }

    def _simular_jornada(self):
        punts_totals = 0
        for jugador in self.state['jugadors']:
            if jugador['titular']:
                punts_base = np.random.normal(
                    jugador['mitjana_punts'],
                    scale=3.0
                )
                punts = max(0, min(25, punts_base * (jugador['forma'] / 10)))
                punts_totals += punts
                
        return punts_totals

    def render(self, mode='human'):
        """Mostra l'estat actual de l'equip"""
        print(f"Pressupost disponible: {self.state['pressupost']:,}€")
        print(f"Punts totals: {self.state['punts_totals']:.2f}")
        print(f"Jornada: {self.state['jornada']}")
        print("\nEquip actual:")
        for i, jugador in enumerate(self.state['jugadors']):
            if jugador['titular']:
                print(f"Jugador {i}:")
                self.render_player(i)

    def _processar_venda(self, index):
        """Processa només la venda d'un jugador"""
        jugador = self.state['jugadors'][index]
        benefici = jugador['preu']
        
        # Creem un jugador bàsic per substituir-lo
        jugador_basic = {
            'preu': 1000000,  # Preu mínim
            'punts_anteriors': 0,
            'mitjana_punts': 2,  # Rendiment baix
            'prediccio_seguent': 2,
            'forma': 5,
            'rival_dificultat': 3,
            'titular': True
        }
        
        self.state['pressupost'] += benefici - jugador_basic['preu']
        self.state['jugadors'][index] = jugador_basic
        
        return benefici - jugador_basic['preu']

    def _inicialitzar_equip(self):
        """Inicialitza l'equip amb jugadors de diferents nivells de preu"""
        equip = []
        
        # Distribuïm els preus inicials en diferents rangs
        preus = [
            np.random.randint(10000000, 15000000),  # 2 jugadors cars
            np.random.randint(10000000, 15000000),
            np.random.randint(7000000, 10000000),   # 3 jugadors mitjans-alts
            np.random.randint(7000000, 10000000),
            np.random.randint(7000000, 10000000),
            np.random.randint(4000000, 7000000),    # 4 jugadors mitjans
            np.random.randint(4000000, 7000000),
            np.random.randint(4000000, 7000000),
            np.random.randint(4000000, 7000000),
            np.random.randint(1000000, 4000000),    # 2 jugadors barats
            np.random.randint(1000000, 4000000)
        ]
        
        # Generem els jugadors amb els preus predeterminats
        for preu in preus:
            jugador = self._generar_jugador(titular=True, preu=preu)
            equip.append(jugador)
        
        return equip

    def render_player(self, index):
        """Mostra la informació d'un jugador específic"""
        if index >= len(self.state['jugadors']):
            print("Índex de jugador no vàlid")
            return
        
        jugador = self.state['jugadors'][index]
        print(f"  Preu: {jugador['preu']:,}€")
        print(f"  mitjana punts: {jugador['mitjana_punts']:.2f}")
        print(f"  Forma: {jugador['forma']:.1f}/10")
        print(f"  Dificultat rival: {jugador['rival_dificultat']}/5")

    def _obtenir_jugador_mercat(self, index):
        """Genera un jugador del mercat amb característiques correlacionades amb el preu"""
        preu = np.random.randint(1000000, 150000000)
        
        # Calculem un factor de qualitat basat en el preu (0 a 1)
        factor_qualitat = (preu - 1000000) / (150000000 - 1000000)
        
        # Generem les estadístiques base
        mitjana_base = 5 + (factor_qualitat * 10)  # Entre 5 i 15 punts
        forma_base = 4 + (factor_qualitat * 6)     # Entre 4 i 10 de forma
        
        # Afegim una mica de variabilitat aleatòria
        mitjana = np.clip(np.random.normal(mitjana_base, 2.0), 0, 15)
        forma = np.clip(np.random.normal(forma_base, 1.0), 0, 10)
        
        return {
            'preu': preu,
            'punts_anteriors': np.random.uniform(0, mitjana * 1.5),
            'mitjana_punts': mitjana,
            'prediccio_seguent': mitjana + np.random.normal(0, 1),
            'forma': forma,
            'rival_dificultat': np.random.randint(1, 5),
            'titular': True
        }

    def _calcular_reward(self, punts_jornada, benefici_transferencia):
        """Calcula la recompensa de manera més equilibrada"""
        reward = 0
        
        # Reduïm el pes del benefici econòmic
        reward += (benefici_transferencia / self.MAX_PREU) * 0.3
        
        # Augmentem el pes del rendiment esportiu
        reward += (punts_jornada / (self.MAX_PUNTS * 11)) * 0.7
        
        # Penalització per pressupost negatiu
        if self.state['pressupost'] < 0:
            reward = -1
        
        # Penalització per massa transferències
        if benefici_transferencia != 0:
            reward -= 0.1  # Petit cost per fer transferències
        
        return reward
