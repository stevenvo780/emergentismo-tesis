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
import numpy as np
import time
import threading
import json

lock_guardar = Lock()


def save_matrices_to_json(energiasMatriz, cargasMatriz, matriz_distancias, ):
    with lock_guardar:
        with open('energiasMatriz.json', 'w') as file:
            json.dump(energiasMatriz.tolist(), file)

        with open('cargasMatriz.json', 'w') as file:
            json.dump(cargasMatriz.tolist(), file)


class Entrenador:
    def __init__(self):
        self.mejor_recompensa = float('-inf')
        self.total_recompensa = 0
        self.generaciones_sin_mejora = 0
        self.universo = Universo()
        self.claves_parametros = [key for key in vars(
            self.universo.physics_rules).keys()]
        self.cargar_mejor_universo()
        self.cargar_mejor_puntaje()
        self.lock = Lock()
        self.poblacion = [self.cargar_red_neuronal() if i == 0 else self.crear_red_neuronal(
        ) for i in range(len(self.claves_parametros))]
        self.pausado = False
        self.elite = []
        self.recompensas = []

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.big_bang)
        self.entrenamiento_thread.start()

    def pausarEntrenamiento(self):
        self.pausado = True

    def reanudarEntrenamiento(self):
        self.pausado = False

    def big_bang(self):
        while True:
            if not self.pausado:
                self.universo.next()
                self.universo.tiempo += 1
                if self.universo.tiempo % systemRules.INTERVALO_ENTRENAMIENTO == 0 and self.universo.tiempo != 0:
                    with lock_guardar:
                        thread = threading.Thread(target=save_matrices_to_json, args=(
                            self.universo.energiasMatriz, self.universo.cargasMatriz, self.universo.matriz_distancias))
                        thread.start()
                    self.entrenar()
                    if self.actualizar_tasas_y_guardar():
                        break
            else:
                time.sleep(1)

    def cargar_mejor_puntaje(self):
        try:
            with open('system_rules.json', 'r') as file:
                rules = json.load(file)
                self.mejor_recompensa = rules.get(
                    "MEJOR_RECOMPENSA", float('-inf'))
        except (FileNotFoundError, json.JSONDecodeError):
            self.guardar_mejor_puntaje()

    def guardar_mejor_puntaje(self):
        rules = {key: getattr(systemRules, key) for key in dir(
            systemRules) if not key.startswith('__')}
        rules["MEJOR_RECOMPENSA"] = self.mejor_recompensa
        with open('system_rules.json', 'w') as file:
            json.dump(rules, file)

    def cargar_red_neuronal(self):
        model = self.crear_red_neuronal()
        try:
            model.load_weights('mejor_red_neuronal.h5')
        except:
            pass
        return model

    def guardar_red_neuronal(self, neural_network):
        neural_network.save_weights('mejor_red_neuronal.h5')

    def cargar_mejor_universo(self):
        try:
            with open('mejor_universo.json', 'r') as file:
                mejor_universo = json.load(file)
                for clave, valor in mejor_universo.items():
                    setattr(self.universo.physics_rules, clave, valor)
        except (FileNotFoundError, json.JSONDecodeError):
            self.guardar_mejor_universo(
                [getattr(self.universo.physics_rules, clave) for clave in self.claves_parametros])

    def guardar_mejor_universo(self, mejores_nuevos_valores):
        mejor_universo = {clave: float(valor) for clave, valor in zip(
            self.claves_parametros, mejores_nuevos_valores)}
        with open('mejor_universo.json', 'w') as file:
            json.dump(mejor_universo, file)

    def crear_red_neuronal(self):
        model = Sequential([
            Dense(systemRules.NEURONAS_DENSIDAD_ENTRADA, input_dim=len(self.claves_parametros), activation='relu',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1)),
            Dense(systemRules.NEURONAS_PROFUNDIDAD, activation='relu',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1)),
            Dense(len(self.claves_parametros), activation='sigmoid',
                  kernel_initializer=RandomUniform(minval=-1, maxval=1))
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def calcular_nuevos_valores(self, neural_network):
        input_data = np.array([getattr(self.universo.physics_rules, key)
                               for key in self.claves_parametros], dtype=float)
        nuevos_valores = neural_network.predict(
            input_data.reshape(1, -1)).flatten()
        self.transformar_valores(nuevos_valores)
        return nuevos_valores

    def transformar_valores(self, nuevos_valores):
        for i, clave in enumerate(self.claves_parametros):
            nuevos_valores[i] = max(0, nuevos_valores[i])
        return nuevos_valores

    def aplicar_nuevos_valores(self, nuevos_valores):
        for i, clave in enumerate(self.claves_parametros):
            setattr(self.universo.physics_rules, clave, nuevos_valores[i])

    def mutate(self, neural_network):
        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_AUMENTO_MUTACION:
            mutation_rate = systemRules.VARIACION_NEURONAL_GRANDE
        else:
            mutation_rate = systemRules.VARIACION_NEURONAL_PEQUEÑA

        weights = neural_network.get_weights()
        for i in range(len(weights)):
            mutation_value = np.random.normal(0, 1, weights[i].shape)
            mutation_mask = np.random.rand(*weights[i].shape) < mutation_rate
            weights[i] += mutation_value * mutation_mask
        neural_network.set_weights(weights)

    def calcularRecompensa(self):
        cargas = self.universo.cargasMatriz
        energias = self.universo.energiasMatriz
        entropia_condicional = calcular_entropia_condicional(
            cargas, energias, self.universo.matriz_distancias)
        recompensa = entropia_condicional * systemRules.FACTOR_ENTROPIA

        return recompensa

    def reiniciarUniverso(self, mejores_nuevos_valores):
        physics_rules = PhysicsRules()
        for i, clave in enumerate(self.claves_parametros):
            valor = mejores_nuevos_valores[i]
            setattr(physics_rules, clave, valor)
        systemRules.FILAS = systemRules.GIRD_SIZE
        systemRules.COLUMNAS = systemRules.GIRD_SIZE
        self.universo = Universo(physics_rules)

    def fitness_function(self, nuevos_valores):
        self.aplicar_nuevos_valores(nuevos_valores)
        return self.calcularRecompensa()

    def entrenar(self):
        mejores_nuevos_valores = [
            getattr(self.universo.physics_rules, key) for key in self.claves_parametros]
        recompensas = []
        for nn in self.poblacion:
            nuevos_valores = self.calcular_nuevos_valores(nn)
            nuevos_valores = self.transformar_valores(nuevos_valores)
            recompensa = self.fitness_function(nuevos_valores)
            recompensas.append(recompensa)
            if recompensa > self.mejor_recompensa:
                self.mejor_recompensa = recompensa
                mejores_nuevos_valores = nuevos_valores
        recompensas_np = np.array(recompensas)
        recompensas_np = np.nan_to_num(
            recompensas_np, nan=0.0, posinf=0.0, neginf=0.0)
        min_recompensa = min(recompensas_np)
        self.recompensas = recompensas_np - min_recompensa
        self.total_recompensa = sum(self.recompensas)
        self.evolve_population()

        if mejores_nuevos_valores is not None:
            if self.total_recompensa > systemRules.MEJOR_RECOMPENSA:
                systemRules.MEJOR_RECOMPENSA = self.total_recompensa
                self.guardar_mejor_universo(mejores_nuevos_valores)
                self.guardar_mejor_puntaje()
            if self.total_recompensa < systemRules.MEJOR_RECOMPENSA:
                self.aplicar_nuevos_valores(mejores_nuevos_valores)
                self.reiniciarUniverso(mejores_nuevos_valores)

    def actualizar_tasas_y_guardar(self):
        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_AUMENTO_MUTACION:
            systemRules.VARIACION_NEURONAL_GRANDE *= 1.05
        else:
            systemRules.VARIACION_NEURONAL_GRANDE /= 1.05

        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_TERMINAR:
            print("Terminando el algoritmo debido a la falta de mejora.")
            return True
        return False

    def seleccionar_padres(self):
        epsilon = 1e-9
        if self.total_recompensa == 0:
            probabilidades_seleccion = np.ones(
                len(self.recompensas)) / len(self.recompensas)
        else:
            probabilidades_seleccion = self.total_recompensa / \
                (self.total_recompensa + epsilon)
            probabilidades_seleccion /= np.sum(probabilidades_seleccion)

        indices_seleccionados = np.random.choice(
            len(self.poblacion), size=len(self.poblacion), p=probabilidades_seleccion)
        return [self.poblacion[i] for i in indices_seleccionados]

    def crear_nueva_poblacion(self, seleccionados):
        nueva_poblacion = []
        for i in range(0, len(seleccionados), 2):
            parent1 = seleccionados[i]
            parent2 = seleccionados[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            nueva_poblacion.extend([child1, child2])
        return nueva_poblacion

    def aplicar_mutaciones(self, nueva_poblacion):
        for nn in nueva_poblacion:
            if random.random() < systemRules.TASA_APRENDIZAJE:
                self.mutate(nn)

    def actualizar_generacion(self, recompensas):
        max_recompensa = max(recompensas)
        if max_recompensa > self.mejor_recompensa:
            self.mejor_recompensa = max_recompensa
            self.generaciones_sin_mejora = 0
        else:
            self.generaciones_sin_mejora += 1

    def evolve_population(self):
        seleccionados = self.seleccionar_padres()
        nueva_poblacion = self.crear_nueva_poblacion(seleccionados)
        self.aplicar_mutaciones(nueva_poblacion)
        self.actualizar_generacion(self.recompensas)
        self.mantener_elite(num_elite=2)
        sorted_indices = np.argsort(self.recompensas)
        worst_indices = sorted_indices[:len(self.elite)]
        for i, elite_member in zip(worst_indices, self.elite):
            nueva_poblacion[i] = elite_member

        self.poblacion = nueva_poblacion

        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_REINICIO:
            self.reiniciar_poblacion()

    def mantener_elite(self, num_elite):
        sorted_indices = np.argsort(self.recompensas)[::-1]
        elite_indices = sorted_indices[:num_elite]
        self.elite = [self.poblacion[i] for i in elite_indices]

    def reiniciar_poblacion(self):
        self.generaciones_sin_mejora = 0
        self.mejor_recompensa = float('-inf')
        self.poblacion = [self.crear_red_neuronal()
                          for _ in range(len(self.claves_parametros))]

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
