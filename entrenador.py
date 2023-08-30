import numpy as np
from universo import Universo
from types_universo import PhysicsRules, ProceduralRules, neuronalRules
from time_procedural import calcular_entropia_condicional
import random
from threading import Thread, Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform, he_normal, glorot_uniform  # type: ignore
import json
import time
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
lock_guardar = Lock()


class Entrenador:
    def __init__(self):
        self.mejor_maxima_recompensa: float = 0.0
        self.actual_total_recompensa = 0
        self.generaciones_sin_mejora = 0
        self.universos = []
        temp_instance = PhysicsRules()
        self.claves_parametros = [key for key in vars(
            temp_instance).keys() if not key.startswith("__")]
        del temp_instance
        self.elite = []
        self.cargar_mejor_puntaje()
        self.poblacion = [self.crear_red_neuronal()
                          for _ in range(neuronalRules.POPULATION_SIZE)]
        self.cargar_poblacion()
        self.pausado = False
        self.recompensa_actual_generacion = 0
        self.recompensas = []
        self.run = True

    def run_universo(self, neural_network):
        try:
            procedural_rules = ProceduralRules()
            nuevos_valores = self.predecir_valores(neural_network)
            physics_rules = PhysicsRules()
            for i, clave in enumerate(self.claves_parametros):
                valor = nuevos_valores[i]
                setattr(physics_rules, clave, valor)
            universo = Universo(physics_rules, procedural_rules)
            universo.tiempo = 1

            self.universos.append(universo)

            recompensa = 0
            recompensa_last = 0
            no_hay_mejora = True
            while no_hay_mejora == True:
                universo.tiempo += 1
                universo.next()
                if universo.tiempo % neuronalRules.INTERVALO_ENTRENAMIENTO == 0:
                    recompensa += calcular_entropia_condicional(
                        universo.cargasMatriz, universo.energiasMatriz, universo.matriz_distancias
                    ) * neuronalRules.FACTOR_ENTROPIA
                    if recompensa / 2 < recompensa_last:
                        no_hay_mejora = False
                    else:
                        recompensa_last = recompensa

            return universo, recompensa
        except Exception as e:
            print(f"Excepción capturada en run_universo: {e}")
            traceback.print_exc()
            return None, -1

    def big_bang(self):
        while self.run:
            if not self.pausado:
                futures = []
                self.universos.clear()
                try:
                    with ThreadPoolExecutor() as executor:
                        for neural_network in self.poblacion:
                            future = executor.submit(
                                self.run_universo, neural_network)
                            futures.append(future)
                except RuntimeError as e:
                    print(f"Error: {e}")
                    break  # O cualquier otro manejo de error que desees
                recompensas = []
                for future in futures:
                    _, recompensa = future.result()
                    recompensas.append(recompensa)
                self.recompensas = recompensas
                self.entrenar()
                if self.generaciones_sin_mejora >= neuronalRules.GENERACIONES_PARA_TERMINAR:
                    print("Terminando el algoritmo debido a la falta de mejora.")
                    break

                if self.generaciones_sin_mejora >= neuronalRules.GENERACIONES_PARA_REINICIO:
                    self.reiniciar_poblacion()
                index_max_recompensa = np.argmax(recompensas)
                self.guardar_mejor_puntaje()
                self.guardar_poblacion()
                self.guardar_mejor_universo(index_max_recompensa)
            else:
                time.sleep(1)

    def entrenar(self):
        self.evolve_population()
        self.actual_total_recompensa = sum(self.recompensas)
        max_recompensa: float = max(self.recompensas)
        if max_recompensa > self.mejor_maxima_recompensa:
            self.mejor_maxima_recompensa = max_recompensa
            self.generaciones_sin_mejora = 0
        else:
            self.generaciones_sin_mejora += 1
        if self.actual_total_recompensa > neuronalRules.MEJOR_TOTAL_RECOMPENSA:
            neuronalRules.MEJOR_TOTAL_RECOMPENSA = self.actual_total_recompensa
            self.guardar_mejor_puntaje()
            self.guardar_poblacion()

    def mutate(self, neural_network):
        if self.generaciones_sin_mejora % neuronalRules.GENERACIONES_PARA_AUMENTO_MUTACION == 0:
            mutation_rate = neuronalRules.VARIACION_NEURONAL_GRANDE
        else:
            mutation_rate = neuronalRules.VARIACION_NEURONAL_PEQUEÑA

        std_dev = np.log(1 + self.generaciones_sin_mejora / 100.0)

        weights = neural_network.get_weights()
        for i in range(len(weights)):
            gradients = np.random.normal(0, std_dev, weights[i].shape)
            mutation_mask = np.random.rand(*weights[i].shape) < mutation_rate
            weights[i] += gradients * mutation_mask
        neural_network.set_weights(weights)

    def evolve_population(self):
        nueva_poblacion = self.crear_nueva_poblacion()
        self.aplicar_mutaciones(nueva_poblacion)
        self.mantener_elite(nueva_poblacion, num_elite=2)

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
            input_data.reshape(1, -1)).flatten()

    def aplicar_mutaciones(self, nueva_poblacion):
        for nn in nueva_poblacion:
            if random.random() < neuronalRules.TASA_APRENDIZAJE:
                self.mutate(nn)

    def mantener_elite(self, nueva_poblacion, num_elite):
        sorted_indices = np.argsort(self.recompensas)[::-1]
        elite_indices = sorted_indices[:num_elite]
        self.elite = [self.poblacion[i] for i in elite_indices]
        # remplaza al peor por un elite
        worst_index = np.argmin(self.recompensas)
        best_elite_member = self.elite[0]
        nueva_poblacion[worst_index] = best_elite_member

        self.poblacion = nueva_poblacion[:len(
            self.poblacion)]

    def crear_red_neuronal(self):
        model = Sequential([
            Dense(neuronalRules.NEURONAS_DENSIDAD_ENTRADA, input_dim=1,
                  activation='relu', kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 1e6))),  # type: ignore
            Dense(neuronalRules.NEURONAS_PROFUNDIDAD, activation='relu',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 1e6))),  # type: ignore
            Dense(len(self.claves_parametros), activation='sigmoid',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=np.random.randint(0, 1e6)))  # type: ignore
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def crear_nueva_poblacion(self):
        indices_mayores_recompensas = np.argsort(
            self.recompensas)[-2:]
        padres = [self.poblacion[i]
                  for i in indices_mayores_recompensas]
        nueva_poblacion = []
        for _ in range(len(self.poblacion) // 2):
            child1, child2 = self.crossover(padres[0], padres[1])
            nueva_poblacion.extend([child1, child2])

        return nueva_poblacion

    def reiniciar_poblacion(self):
        self.generaciones_sin_mejora = 0
        self.mejor_maxima_recompensa = float('-inf')
        self.poblacion = [self.crear_red_neuronal()
                          for _ in range(neuronalRules.POPULATION_SIZE)]

    def cargar_mejor_puntaje(self):
        try:
            with open('system_rules.json', 'r') as file:
                rules = json.load(file)
                self.mejor_maxima_recompensa = rules.get(
                    "MEJOR_TOTAL_RECOMPENSA", float('-inf'))
        except (FileNotFoundError, json.JSONDecodeError):
            self.guardar_mejor_puntaje()

    def guardar_mejor_puntaje(self):
        rules = {key: getattr(neuronalRules, key) for key in dir(
            neuronalRules) if not key.startswith('__')}
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

        subdirectorio = 'poblacion_guardada'
        if not os.path.exists(subdirectorio):
            os.mkdir(subdirectorio)

        for i, modelo in enumerate(self.poblacion):
            modelo.save_weights(f'{subdirectorio}/red_neuronal_{i}.h5')

    def cargar_poblacion(self):
        subdirectorio = 'poblacion_guardada'
        if os.path.exists(subdirectorio):
            for i in range(len(self.poblacion)):
                try:
                    self.poblacion[i].load_weights(
                        f'{subdirectorio}/red_neuronal_{i}.h5')
                except:
                    print(
                        f"No se pudo cargar la red neuronal {i} para la población.")
        else:
            print("No hay una población guardada para cargar.")

    def guardar_mejor_universo(self, index_universo):
        mejores_nuevos_valores = self.universos[index_universo].physics_rules
        mejor_universo = {key: float(value) for key, value in vars(
            mejores_nuevos_valores).items()}
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
