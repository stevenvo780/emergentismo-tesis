import numpy as np
from universo import Universo
from types_universo import PhysicsRules, systemRules
from time_procedural import calcular_entropia_condicional
import random
from threading import Thread, Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform  # type: ignore
import json
import time
import threading
import os
from typing import List
lock_guardar = Lock()


class Entrenador:
    def __init__(self):
        self.lock = Lock()
        self.mejor_maxima_recompensa: float = 0.0
        self.actual_total_recompensa = 0
        self.generaciones_sin_mejora = 0
        self.universo = Universo()
        self.universo.tiempo = 1
        self.claves_parametros = [key for key in vars(
            self.universo.physics_rules).keys()]
        self.elite = []
        self.cargar_mejor_universo()
        self.cargar_mejor_puntaje()
        self.poblaciones = {}
        for clave in self.claves_parametros:
            self.poblaciones[clave] = [self.crear_red_neuronal()
                                       for _ in range(systemRules.POPULATION_SIZE)]
        self.cargar_poblacion()
        self.pausado = False
        self.recompensa_actual_generacion = 0
        self.recompensas_por_clave = {clave: []
                                      for clave in self.claves_parametros}
        self.contador_test_poblacion = 0

    def big_bang(self):
        while True:
            if not self.pausado:
                self.universo.next()
                self.universo.tiempo += 1
                if self.contador_test_poblacion <= systemRules.POPULATION_SIZE and self.universo.tiempo % systemRules.INTERVALO_ENTRENAMIENTO == 0:
                    recompensaLast = calcular_entropia_condicional(
                        self.universo.cargasMatriz, self.universo.energiasMatriz, self.universo.matriz_distancias) * systemRules.FACTOR_ENTROPIA
                    if self.recompensa_actual_generacion < recompensaLast:
                        nuevos_valores = []
                        for clave in self.claves_parametros:
                            nuevos_valores.append(self.predecir_valores(
                                self.poblaciones[clave][self.contador_test_poblacion]))
                            self.recompensas_por_clave[clave].append(
                                recompensaLast)
                        self.contador_test_poblacion += 1
                        self.reiniciarUniverso(nuevos_valores)
                    self.recompensa_actual_generacion = recompensaLast
                elif self.contador_test_poblacion >= systemRules.POPULATION_SIZE:
                    with lock_guardar:
                        thread = threading.Thread(target=save_matrices_to_json, args=(
                            self.universo.energiasMatriz, self.universo.cargasMatriz))
                        thread.start()
                    self.entrenar()
                    if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_TERMINAR:
                        print("Terminando el algoritmo debido a la falta de mejora.")
                        break
                    self.contador_test_poblacion = 0
                    self.recompensas_por_clave = {clave: []
                                                  for clave in self.claves_parametros}
            else:
                time.sleep(1)

    def entrenar(self):
        max_recompensas: List[float] = []
        for clave in self.claves_parametros:
            self.evolve_population(clave)
            self.actual_total_recompensa += sum(
                self.recompensas_por_clave[clave])
            max_recompensas.append(max(self.recompensas_por_clave[clave]))
        max_recompensa = max(max_recompensas)
        if max_recompensa > self.mejor_maxima_recompensa:
            self.mejor_maxima_recompensa = max_recompensa
            self.generaciones_sin_mejora = 0
            systemRules.VARIACION_NEURONAL_GRANDE /= 1.05
            systemRules.VARIACION_NEURONAL_PEQUEÑA /= 0.5
        else:
            self.generaciones_sin_mejora += 1
        if self.actual_total_recompensa > systemRules.MEJOR_TOTAL_RECOMPENSA:
            systemRules.MEJOR_TOTAL_RECOMPENSA = self.actual_total_recompensa
            systemRules.VARIACION_NEURONAL_GRANDE /= 2.05
            systemRules.VARIACION_NEURONAL_PEQUEÑA /= 1.05
            self.guardar_mejor_puntaje()
            self.guardar_mejor_universo()
            self.guardar_poblacion()
        elif self.generaciones_sin_mejora % systemRules.GENERACIONES_PARA_AUMENTO_MUTACION:
            systemRules.VARIACION_NEURONAL_GRANDE *= 2.05
            systemRules.VARIACION_NEURONAL_PEQUEÑA *= 1.05

        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_REINICIO:
            self.reiniciar_poblacion()

    def mutate(self, neural_network):
        if self.generaciones_sin_mejora % systemRules.GENERACIONES_PARA_AUMENTO_MUTACION:
            mutation_rate = systemRules.VARIACION_NEURONAL_GRANDE
        else:
            mutation_rate = systemRules.VARIACION_NEURONAL_PEQUEÑA

        weights = neural_network.get_weights()
        for i in range(len(weights)):
            mutation_value = np.random.normal(0, 1, weights[i].shape)
            mutation_mask = np.random.rand(*weights[i].shape) < mutation_rate
            weights[i] += mutation_value * mutation_mask
        neural_network.set_weights(weights)

    def evolve_population(self, clave):
        nueva_poblacion = self.crear_nueva_poblacion(clave)
        self.aplicar_mutaciones(nueva_poblacion)
        self.mantener_elite(clave, nueva_poblacion, num_elite=2)

    def crossover(self, parent1, parent2):
        child1 = self.crear_red_neuronal()
        child2 = self.crear_red_neuronal()
        for i in range(len(parent1.get_weights())):
            shape = parent1.get_weights()[i].shape
            if np.prod(shape) <= 2:
                continue
            punto_cruce1 = random.randint(1, np.prod(shape) - 3)
            punto_cruce2 = random.randint(punto_cruce1 + 1, np.prod(shape) - 2)
            weights1 = np.concatenate((parent1.get_weights()[i].flatten()[:punto_cruce1],
                                       parent2.get_weights()[i].flatten()[
                punto_cruce1:punto_cruce2],
                parent1.get_weights()[i].flatten()[punto_cruce2:]))
            weights2 = np.concatenate((parent2.get_weights()[i].flatten()[:punto_cruce1],
                                       parent1.get_weights()[i].flatten()[
                punto_cruce1:punto_cruce2],
                parent2.get_weights()[i].flatten()[punto_cruce2:]))
            child1_weights = child1.get_weights()
            child1_weights[i] = weights1.reshape(shape)
            child2_weights = child2.get_weights()
            child2_weights[i] = weights2.reshape(shape)
            child1.set_weights(child1_weights)
            child2.set_weights(child2_weights)
        return child1, child2

    def predecir_valores(self, neural_network):
        input_data = np.array([len(self.claves_parametros) / 100],
                              dtype=float)
        return neural_network.predict(
            input_data.reshape(1, -1)).flatten()[0]

    def aplicar_mutaciones(self, nueva_poblacion):
        for nn in nueva_poblacion:
            if random.random() < systemRules.TASA_APRENDIZAJE:
                self.mutate(nn)

    def mantener_elite(self, clave, nueva_poblacion, num_elite):
        sorted_indices = np.argsort(self.recompensas_por_clave[clave])[::-1]
        elite_indices = sorted_indices[:num_elite]
        self.elite = [self.poblaciones[clave][i] for i in elite_indices]

        worst_index = np.argmin(self.recompensas_por_clave[clave])
        best_elite_member = self.elite[0]
        nueva_poblacion[worst_index] = best_elite_member

        self.poblaciones[clave] = nueva_poblacion[:len(
            self.poblaciones[clave])]

    def crear_red_neuronal(self):
        model = Sequential([
            Dense(systemRules.NEURONAS_DENSIDAD_ENTRADA, input_dim=1,
                  activation='relu', kernel_initializer=RandomUniform(minval=-1, maxval=1)),
            Dense(systemRules.NEURONAS_PROFUNDIDAD, activation='relu',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1)),
            Dense(1, activation='sigmoid',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1))
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def crear_nueva_poblacion(self, clave):
        indices_mayores_recompensas = np.argsort(
            self.recompensas_por_clave[clave])[-2:]
        padres = [self.poblaciones[clave][i]
                  for i in indices_mayores_recompensas]
        nueva_poblacion = []
        for _ in range(len(self.poblaciones[clave]) // 2):
            child1, child2 = self.crossover(padres[0], padres[1])
            nueva_poblacion.extend([child1, child2])

        return nueva_poblacion

    def reiniciar_poblacion(self):
        self.generaciones_sin_mejora = 0
        self.mejor_maxima_recompensa = float('-inf')
        for clave in self.claves_parametros:
            self.poblaciones[clave] = [self.crear_red_neuronal()
                                       for _ in range(systemRules.POPULATION_SIZE)]

    def reiniciarUniverso(self, mejores_nuevos_valores):
        physics_rules = PhysicsRules()
        for i, clave in enumerate(self.claves_parametros):
            valor = mejores_nuevos_valores[i]
            setattr(physics_rules, clave, valor)
        systemRules.FILAS = systemRules.GIRD_SIZE
        systemRules.COLUMNAS = systemRules.GIRD_SIZE
        self.universo = Universo(physics_rules)

    def cargar_mejor_puntaje(self):
        try:
            with open('system_rules.json', 'r') as file:
                rules = json.load(file)
                self.mejor_maxima_recompensa = rules.get(
                    "MEJOR_TOTAL_RECOMPENSA", float('-inf'))
        except (FileNotFoundError, json.JSONDecodeError):
            self.guardar_mejor_puntaje()

    def guardar_mejor_puntaje(self):
        rules = {key: getattr(systemRules, key) for key in dir(
            systemRules) if not key.startswith('__')}
        rules["MEJOR_TOTAL_RECOMPENSA"] = self.mejor_maxima_recompensa
        with open('system_rules.json', 'w') as file:
            json.dump(rules, file)

    def cargar_red_neuronal(self):
        model = self.crear_red_neuronal()
        try:
            model.load_weights('mejor_red_neuronal.h5')
        except:
            pass
        return model

    def guardar_poblacion(self):
        if not os.path.exists('poblacion_guardada'):
            os.mkdir('poblacion_guardada')

        for clave in self.claves_parametros:
            subdirectorio = f'poblacion_guardada/{clave}'
            if not os.path.exists(subdirectorio):
                os.mkdir(subdirectorio)
            for i, modelo in enumerate(self.poblaciones[clave]):
                modelo.save_weights(f'{subdirectorio}/red_neuronal_{i}.h5')

    def cargar_poblacion(self):
        if os.path.exists('poblacion_guardada'):
            for clave in self.claves_parametros:
                subdirectorio = f'poblacion_guardada/{clave}'
                if os.path.exists(subdirectorio):
                    for i in range(len(self.poblaciones[clave])):
                        try:
                            self.poblaciones[clave][i].load_weights(
                                f'{subdirectorio}/red_neuronal_{i}.h5')
                        except:
                            print(
                                f"No se pudo cargar la red neuronal {i} para {clave}")
                else:
                    print(
                        f"No hay una población guardada para cargar para {clave}.")
        else:
            print("No hay una población guardada para cargar.")

    def cargar_mejor_universo(self):
        try:
            with open('mejor_universo.json', 'r') as file:
                mejor_universo = json.load(file)
                for clave, valor in mejor_universo.items():
                    setattr(self.universo.physics_rules, clave, valor)
        except (FileNotFoundError, json.JSONDecodeError):
            self.guardar_mejor_universo()

    def guardar_mejor_universo(self):
        if self.elite:
            mejores_nuevos_valores = []
            for clave in self.claves_parametros:
                mejores_nuevos_valores.append(self.predecir_valores(
                    self.poblaciones[clave][self.contador_test_poblacion-1]))
        else:
            mejores_nuevos_valores = np.random.rand(
                systemRules.POPULATION_SIZE)

        mejor_universo = {clave: float(valor) for clave, valor in zip(
            self.claves_parametros, mejores_nuevos_valores)}

        with open('mejor_universo.json', 'w') as file:
            json.dump(mejor_universo, file)

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.big_bang)
        self.entrenamiento_thread.start()

    def pausarEntrenamiento(self):
        self.pausado = True

    def reanudarEntrenamiento(self):
        self.pausado = False


def save_matrices_to_json(energiasMatriz, cargasMatriz):
    with lock_guardar:
        with open('energiasMatriz.json', 'w') as file:
            json.dump(energiasMatriz.tolist(), file)

        with open('cargasMatriz.json', 'w') as file:
            json.dump(cargasMatriz.tolist(), file)
