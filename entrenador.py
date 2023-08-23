import numpy as np
from universo import Universo
from types_universo import PhysicsRules, systemRules
import random
from threading import Thread, Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform  # type: ignore
import json
import numpy as np
import cupy as cp


def contar_estructuras_cerradas(matriz_relaciones):
    matriz_adj = (matriz_relaciones > 0).astype(cp.int32)
    estructuras_cerradas = {
        "círculos_4_nodos": cp.trace(cp.linalg.matrix_power(matriz_adj, 4)).item() // 8,
        "cuadrados": cp.trace(cp.linalg.matrix_power(matriz_adj, 4)).item() // 24,
        "hexágonos": cp.trace(cp.linalg.matrix_power(matriz_adj, 6)).item() // 720
    }
    return estructuras_cerradas


class Entrenador:
    def __init__(self):
        self.mejor_recompensa = float('-inf')
        self.ultimo_puntaje = 0
        self.generaciones_sin_mejora = 0
        self.universo = Universo()
        self.claves_parametros = [key for key in vars(
            self.universo.physics_rules).keys()]
        self.cargar_mejor_universo()
        self.cargar_mejor_puntaje()
        self.lock = Lock()
        self.poblacion = [self.cargar_red_neuronal() if i == 0 else self.crear_red_neuronal(
        ) for i in range(systemRules.NEURONAS_SALIDA_CANTIDAD)]

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.big_bang)
        self.entrenamiento_thread.start()

    def big_bang(self):
        while True:
            self.universo.next()
            self.universo.tiempo += 1
            if self.universo.tiempo % systemRules.INTERVALO_ENTRENAMIENTO == 0 and self.universo.tiempo != 0:
                self.entrenar()

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
            if clave == 'FACTOR_RELACION':
                nuevos_valores[i] = int(
                    nuevos_valores[i] * systemRules.FACTOR_RELACION_LIMIT)
            else:
                nuevos_valores[i] = max(0, nuevos_valores[i])
        return nuevos_valores

    def aplicar_nuevos_valores(self, nuevos_valores):
        for i, clave in enumerate(self.claves_parametros):
            setattr(self.universo.physics_rules, clave, nuevos_valores[i])

    def mutate(self, neural_network, increase_mutation=False):
        factor = systemRules.VARIACION_NEURONAL_GRANDE if increase_mutation else systemRules.VARIACION_NEURONAL_PEQUEÑA
        weights = neural_network.get_weights()
        for i in range(len(weights)):
            mutation_rate = factor * 2 if increase_mutation else factor
            weights[i] += np.random.normal(0, mutation_rate, weights[i].shape)
        neural_network.set_weights(weights)

    def calcularRecompensa(self):
        matriz_relaciones = self.universo.matriz_relaciones
        numeroDeRelaciones = cp.sum(matriz_relaciones > 0).item()

        estructuras_cerradas = contar_estructuras_cerradas(matriz_relaciones)
        recompensa_por_relaciones = numeroDeRelaciones * \
            systemRules.RECOMPENSA_POR_RELACION
        recompensa = recompensa_por_relaciones

        total_estructuras_cerradas = 0
        for idx, (estructura, cantidad) in enumerate(estructuras_cerradas.items()):
            # print(f'{estructura}: {cantidad}')
            total_estructuras_cerradas += cantidad
            recompensa += cantidad * systemRules.RECOMPENSA_EXTRA_CERRADA * idx

        proporcion_estructuras_cerradas = total_estructuras_cerradas / \
            (numeroDeRelaciones + 1e-5)

        if proporcion_estructuras_cerradas < systemRules.UMBRAL_PROPORCION_ESTRUCUTRAS_CERRADAS:
            penalizacion = (systemRules.UMBRAL_PROPORCION_ESTRUCUTRAS_CERRADAS - proporcion_estructuras_cerradas) * \
                10 * systemRules.PENALIZACION_RELACIONES_SINFORMA
            recompensa -= penalizacion

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

        total_recompensa = sum(recompensas)
        if total_recompensa <= 0:
            for nn in self.poblacion:
                self.mutate(nn, increase_mutation=True)
            num_to_reset = int(len(self.poblacion) *
                               systemRules.PORCENTAJE_POBLACION_MUTACION)
            for i in range(num_to_reset):
                self.poblacion[i] = self.crear_red_neuronal()

        elif total_recompensa < systemRules.MEJOR_RECOMPENSA:
            for nn in self.poblacion:
                self.mutate(nn, increase_mutation=False)

        recompensas_np = cp.asarray(recompensas).get()
        best_nn = self.poblacion[np.argmax(recompensas_np)]
        self.guardar_red_neuronal(best_nn)

        if total_recompensa != 0:
            self.evolve_population(recompensas)

        if mejores_nuevos_valores is not None:
            if total_recompensa > systemRules.MEJOR_RECOMPENSA:
                systemRules.MEJOR_RECOMPENSA = total_recompensa
                self.guardar_mejor_universo(mejores_nuevos_valores)
                self.guardar_mejor_puntaje()
            if total_recompensa < systemRules.MEJOR_RECOMPENSA:
                self.aplicar_nuevos_valores(mejores_nuevos_valores)
                self.reiniciarUniverso(mejores_nuevos_valores)
        self.ultimo_puntaje = total_recompensa
        self.poblacion[0] = best_nn

    def evolve_population(self, recompensas):
        recompensas_np = np.array(recompensas)
        recompensas_np = np.nan_to_num(
            recompensas_np, nan=0.0, posinf=0.0, neginf=0.0)
        min_recompensa = min(recompensas_np)
        adjusted_recompensas = recompensas_np - min_recompensa
        total_recompensa = sum(adjusted_recompensas)
        epsilon = 1e-9

        if total_recompensa == 0:
            probabilidades_seleccion = np.ones(
                len(adjusted_recompensas)) / len(adjusted_recompensas)
        else:
            probabilidades_seleccion = adjusted_recompensas / \
                (total_recompensa + epsilon)
            # Normalizar probabilidades
            probabilidades_seleccion /= np.sum(probabilidades_seleccion)

        indices_seleccionados = np.random.choice(
            len(self.poblacion), size=len(self.poblacion), p=probabilidades_seleccion)
        seleccionados = [self.poblacion[i] for i in indices_seleccionados]
        nueva_poblacion = []
        for i in range(0, len(seleccionados), 2):
            parent1 = seleccionados[i]
            parent2 = seleccionados[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            nueva_poblacion.extend([child1, child2])
        for nn in nueva_poblacion:
            if random.random() < systemRules.TASA_APRENDIZAJE:
                self.mutate(nn)
        self.poblacion = nueva_poblacion
        max_recompensa = max(recompensas)
        if max_recompensa > self.mejor_recompensa:
            self.mejor_recompensa = max_recompensa
            self.generaciones_sin_mejora = 0
        else:
            self.generaciones_sin_mejora += 1

        if self.generaciones_sin_mejora >= systemRules.GENERACIONES_PARA_REINICIO:
            self.reiniciar_poblacion()

    def reiniciar_poblacion(self):
        self.generaciones_sin_mejora = 0
        self.mejor_recompensa = float('-inf')
        self.poblacion = [self.crear_red_neuronal()
                          for _ in range(systemRules.NEURONAS_SALIDA_CANTIDAD)]

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
