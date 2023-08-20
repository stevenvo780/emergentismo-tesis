import numpy as np
from universo import Universo
from types_universo import PhysicsRules, systemRules, SystemRules
import random
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, wait
from keras.models import Sequential
from keras.layers import Dense
import json


class Entrenador:
    def __init__(self):
        self.universo = Universo()
        self.claves_parametros = [key for key in vars(
            self.universo.physicsRules).keys()]  # Mover esta línea antes de la llamada a cargar_mejor_universo
        self.cargar_mejor_universo()
        self.tiempoLimiteSinEstructuras = systemRules.TIEMPO_LIMITE_ESTRUCTURA
        self.tasaDeAprendizaje = systemRules.TASA_APRENDIZAJE
        self.tiempoSinEstructuras = 0
        self.lock = Lock()
        self.poblacion = [self.cargar_red_neuronal() if i == 0 else self.crear_red_neuronal()
                          for i in range(systemRules.NEURONAS_CANTIDAD)]

    def cargar_red_neuronal(self):
        model = self.crear_red_neuronal()  # Crear la estructura de la red
        try:
            model.load_weights('mejor_red_neuronal.h5')  # Cargar los pesos
        except:
            pass  # Manejar la excepción si no hay un archivo de pesos guardado
        return model

    def guardar_red_neuronal(self, neural_network):
        neural_network.save_weights('mejor_red_neuronal.h5')

    def cargar_mejor_universo(self):
        try:
            with open('mejor_universo.json', 'r') as file:
                mejor_universo = json.load(file)
                for clave, valor in mejor_universo.items():
                    setattr(self.universo.physicsRules, clave, valor)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si el archivo no existe o no es válido, guardar los valores actuales como el mejor universo
            self.guardar_mejor_universo(
                [getattr(self.universo.physicsRules, clave) for clave in self.claves_parametros])

    def guardar_mejor_universo(self, mejores_nuevos_valores):
        mejor_universo = {clave: float(valor) for clave, valor in zip(
            self.claves_parametros, mejores_nuevos_valores)}
        with open('mejor_universo.json', 'w') as file:
            json.dump(mejor_universo, file)

    def crear_red_neuronal(self):
        model = Sequential([
            Dense(12, input_dim=len(self.claves_parametros), activation='relu'),
            Dense(8, activation='relu'),
            Dense(len(self.claves_parametros), activation='sigmoid')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def calcular_nuevos_valores(self, neural_network):
        input_data = np.array(
            [getattr(self.universo.physicsRules, key) for key in self.claves_parametros], dtype=float)
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
            setattr(self.universo.physicsRules, clave, nuevos_valores[i])

    def mutate(self, neural_network):
        weights = neural_network.get_weights()
        for i in range(len(weights)):
            weights[i] += np.random.normal(0,
                                           systemRules.NEURONAL_FACTOR, weights[i].shape)
            if np.random.rand() < 0.05:
                weights[i] += np.random.normal(
                    0, systemRules.NEURONAL_FACTOR, weights[i].shape) * 0.5
        neural_network.set_weights(weights)

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.entrenamientoPerpetuo)
        self.entrenamiento_thread.start()

    def entrenamientoPerpetuo(self):
        while True:
            self.nextStepRecursivo()

    def nextStepRecursivo(self):
        self.universo.next()
        self.universo.tiempo += 1
        if self.universo.tiempo % self.tiempoLimiteSinEstructuras == 0:
            self.entrenar()

    def actualizarConfiguracion(self, tiempoLimiteSinEstructuras, tasaDeAprendizaje):
        self.tiempoLimiteSinEstructuras = tiempoLimiteSinEstructuras
        self.tasaDeAprendizaje = tasaDeAprendizaje

    def calcularRecompensa(self, nodos):
        numeroDeRelaciones = 0
        numeroDeEstructurasCerradas = 0
        step = len(nodos) // systemRules.NUM_THREADS

        def process_nodes(start_index, end_index):
            nonlocal numeroDeRelaciones, numeroDeEstructurasCerradas
            local_count = 0
            local_closed_count = 0
            for i in range(start_index, end_index):
                nodo = nodos[i]
                nodosRelacionados = [
                    rel.nodoId for rel in nodo.memoria.relaciones]
                # Contamos todas las relaciones
                local_count += len(nodosRelacionados)
                if i in nodosRelacionados:  # Verificamos si el nodo tiene una relación consigo mismo, lo que indica una estructura cerrada
                    local_closed_count += 1

            with self.lock:
                numeroDeRelaciones += local_count
                numeroDeEstructurasCerradas += local_closed_count

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                process_nodes, i * step, (i + 1) * step if i != systemRules.NUM_THREADS - 1 else len(nodos)) for i in range(systemRules.NUM_THREADS)]
            wait(futures)

        recompensa = (numeroDeRelaciones * systemRules.RECOMPENSA_POR_RELACION) + \
            (numeroDeEstructurasCerradas * systemRules.RECOMPENSA_EXTRA_CERRADA)
        return recompensa

    def reiniciarUniverso(self, mejores_nuevos_valores):
        physicsRules = PhysicsRules()
        for i, clave in enumerate(self.claves_parametros):
            valor = mejores_nuevos_valores[i]
            setattr(physicsRules, clave, valor)
        systemRules.FILAS = systemRules.GIRD_SIZE
        systemRules.COLUMNAS = systemRules.GIRD_SIZE
        self.universo = Universo(physicsRules)

    def fitness_function(self, nuevos_valores):
        self.aplicar_nuevos_valores(nuevos_valores)
        return self.calcularRecompensa(self.universo.nodos)

    def entrenar(self):
        mejores_nuevos_valores, mejor_recompensa = None, float('-inf')
        recompensas = []
        for nn in self.poblacion:
            nuevos_valores = self.calcular_nuevos_valores(nn)
            nuevos_valores = self.transformar_valores(nuevos_valores)
            recompensa = self.fitness_function(nuevos_valores)
            recompensas.append(recompensa)
            if recompensa > mejor_recompensa:
                mejor_recompensa = recompensa
                mejores_nuevos_valores = nuevos_valores

        total_recompensa = sum(recompensas)
        print('total_recompensa', total_recompensa)

        # Conservar el mejor individuo (elitismo)
        best_nn = self.poblacion[np.argmax(recompensas)]
        self.guardar_red_neuronal(best_nn)

        if total_recompensa < systemRules.PUNTAGE_MINIMO_REINICIO and mejores_nuevos_valores is None:
            mejores_nuevos_valores = self.aplicar_pesos_temporales(
                mejores_nuevos_valores)

        if total_recompensa != 0:
            self.evolve_population(recompensas)

        print(mejores_nuevos_valores)
        if mejores_nuevos_valores is not None:
            self.guardar_mejor_universo(mejores_nuevos_valores)
            self.aplicar_nuevos_valores(mejores_nuevos_valores)
            self.tiempoSinEstructuras = 0
            self.reiniciarUniverso(mejores_nuevos_valores)

        self.poblacion[0] = best_nn

    def evolve_population(self, recompensas):
        total_recompensa = sum(recompensas)
        probabilidades_seleccion = [
            rec / total_recompensa for rec in recompensas]
        seleccionados = np.random.choice(self.poblacion, size=len(
            self.poblacion), p=probabilidades_seleccion)
        nueva_poblacion = []
        for i in range(0, len(seleccionados), 2):
            parent1 = seleccionados[i]
            parent2 = seleccionados[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            nueva_poblacion.extend([child1, child2])
        for nn in nueva_poblacion:
            if random.random() < self.tasaDeAprendizaje:
                self.mutate(nn)
        self.poblacion = nueva_poblacion

    def crossover(self, parent1, parent2):
        child1 = self.crear_red_neuronal()
        child2 = self.crear_red_neuronal()
        for i in range(len(parent1.get_weights())):
            shape = parent1.get_weights()[i].shape
            if np.prod(shape) <= 2:
                continue
            punto_cruce = random.randint(1, np.prod(shape) - 2)
            weights1 = np.concatenate((parent1.get_weights()[i].flatten(
            )[:punto_cruce], parent2.get_weights()[i].flatten()[punto_cruce:]))
            weights2 = np.concatenate((parent2.get_weights()[i].flatten(
            )[:punto_cruce], parent1.get_weights()[i].flatten()[punto_cruce:]))
            child1_weights = child1.get_weights()
            child1_weights[i] = weights1.reshape(shape)
            child2_weights = child2.get_weights()
            child2_weights[i] = weights2.reshape(shape)
            child1.set_weights(child1_weights)
            child2.set_weights(child2_weights)
        return child1, child2

    def aplicar_pesos_temporales(self, mejores_nuevos_valores):
        pesos_temporales = (np.random.rand(
            len(self.claves_parametros)) - 0.5) * 0.1
        return [getattr(self.universo.physicsRules, clave) + peso * getattr(self.universo.physicsRules, clave) for clave, peso in zip(self.claves_parametros, pesos_temporales)]
