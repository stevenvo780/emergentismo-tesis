import numpy as np
from universo import Universo
from types_universo import PhysicsRules, SystemRules
import random
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, wait
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


class Entrenador:
    def __init__(self):
        self.universo = Universo()
        self.tiempoLimiteSinEstructuras = SystemRules.TIEMPO_LIMITE_ESTRUCTURA
        self.tasaDeAprendizaje = 0.05
        self.tiempoSinEstructuras = 0
        self.claves_parametros = [key for key in vars(
            self.universo.valoresSistema).keys()]
        self.poblacion = [self.crear_red_neuronal() for _ in range(10)]

    def crear_red_neuronal(self):
        model = Sequential()
        model.add(Dense(12, input_dim=len(
            self.claves_parametros), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(len(self.claves_parametros), activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def fitness_function(self, neural_network):
        valores = self.universo.valoresSistema
        input_data = np.array([getattr(valores, key)
                               for key in self.claves_parametros])
        nuevos_valores = neural_network.predict(
            input_data.reshape(1, -1)).flatten()
        return self.calcularRecompensa(self.universo.nodos)

    def calcular_nuevos_valores(self, neural_network):
        valores = self.universo.valoresSistema
        input_data = np.array([getattr(valores, key)
                               for key in self.claves_parametros])
        nuevos_valores = neural_network.predict(
            input_data.reshape(1, -1)).flatten()
        return nuevos_valores

    def aplicar_nuevos_valores(self, nuevos_valores):
        for i, clave in enumerate(self.claves_parametros):
            setattr(self.universo.valoresSistema, clave, nuevos_valores[i])

    def mutate(self, neural_network):
        for i in range(len(neural_network.get_weights())):
            neural_network.get_weights(
            )[i] += np.random.normal(0, 0.1, neural_network.get_weights()[i].shape)

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.entrenamientoPerpetuo)
        self.entrenamiento_thread.start()

    def entrenamientoPerpetuo(self):
        while True:  # Puedes agregar una condición de salida si lo deseas
            self.nextStepRecursivo()

    def nextStepRecursivo(self):
        self.universo.next()
        self.universo.tiempo += 1
        visualizar = self.universo.tiempo % self.tiempoLimiteSinEstructuras
        if visualizar == 0:
            self.entrenarPerpetuo()

    def actualizarConfiguracion(self, tiempoLimiteSinEstructuras, tasaDeAprendizaje):
        self.tiempoLimiteSinEstructuras = tiempoLimiteSinEstructuras
        self.tasaDeAprendizaje = tasaDeAprendizaje

    def calcularRecompensa(self, nodos):
        numeroDeEstructuras = 0
        nodosVisitados = set()
        step = len(nodos) // 4

        def process_nodes(start_index, end_index):
            nonlocal numeroDeEstructuras, nodosVisitados
            for i in range(start_index, end_index):
                nodo = nodos[i]
                if nodo.id in nodosVisitados:
                    continue
                nodosRelacionados = [
                    rel.nodoId for rel in nodo.memoria.relaciones]
                if len(nodosRelacionados) >= self.universo.valoresSistema.ESPERADO_EMERGENTE:
                    esEstructuraValida = all([
                        any(nodoRelacionado.id == idRelacionado and nodoRelacionado.memoria.energia > self.universo.valoresSistema.ENERGIA
                            for nodoRelacionado in nodos)
                        for idRelacionado in nodosRelacionados
                    ])
                    if esEstructuraValida or nodo.id in nodosRelacionados:  # Verificar relaciones consigo mismo
                        for idRelacionado in nodosRelacionados:
                            nodosVisitados.add(idRelacionado)
                        numeroDeEstructuras += 1

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                process_nodes, i * step, (i + 1) * step if i != 3 else len(nodos)) for i in range(4)]
            wait(futures)

        if numeroDeEstructuras == 0:
            self.tiempoSinEstructuras += 1
        else:
            self.tiempoSinEstructuras = 0

        return numeroDeEstructuras

    def reiniciarUniverso(self, mejores_nuevos_valores):
        valoresSistema = PhysicsRules()

        # Lista de claves que deben ser redondeadas a enteros, excluyendo 'FILAS' y 'COLUMNAS'
        enteros = ['LIMITE_EDAD', 'CRECIMIENTO_X', 'CRECIMIENTO_Y', 'LIMITE_RELACIONAL',
                   'DISTANCIA_MAXIMA_RELACION', 'ESPERADO_EMERGENTE', 'FACTOR_RELACION']

        # Aplicar los nuevos valores, excepto para 'FILAS' y 'COLUMNAS'
        for i, clave in enumerate(self.claves_parametros):
            if clave not in [
                'FILAS',
                'COLUMNAS',
                'CRECIMIENTO_X',
                'CRECIMIENTO_Y',
                'LIMITE_RELACIONAL',
                'ESPERADO_EMERGENTE',
            ]:  # No cambiar 'FILAS' y 'COLUMNAS'
                valor = mejores_nuevos_valores[i]
                if clave in enteros:
                    # Redondear si la clave está en la lista de enteros
                    valor = round(valor)
                setattr(valoresSistema, clave, valor)

        self.universo = Universo(valoresSistema)

    def entrenarPerpetuo(self):
        mejores_nuevos_valores = None
        mejor_recompensa = float('-inf')
        recompensas = []
        for nn in self.poblacion:
            nuevos_valores = self.calcular_nuevos_valores(nn)
            recompensa = self.fitness_function(nn)
            recompensas.append(recompensa)
            if recompensa > mejor_recompensa:
                mejor_recompensa = recompensa
                mejores_nuevos_valores = nuevos_valores
        total_recompensa = sum(recompensas)
        print('total_recompensa', total_recompensa)

        pesos_temporales = (np.random.rand(
            len(self.claves_parametros)) - 0.5) * 0.1

        # Verificar si total_recompensa es cero
        if total_recompensa != 0:
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

            # Mutación
            for nn in nueva_poblacion:
                if random.random() < self.tasaDeAprendizaje:
                    self.mutate(nn)

            self.poblacion = nueva_poblacion
        print(mejores_nuevos_valores)
        if self.tiempoSinEstructuras > self.tiempoLimiteSinEstructuras or total_recompensa < SystemRules.PUNTAGE_MINIMO_REINICIO:
            if mejores_nuevos_valores is None:
                # Sumar o restar valores basándose en pesos temporales
                mejores_nuevos_valores = [getattr(self.universo.valoresSistema, clave) + peso
                                          for clave, peso in zip(self.claves_parametros, pesos_temporales)]
            self.aplicar_nuevos_valores(mejores_nuevos_valores)
            self.tiempoSinEstructuras = 0
            self.reiniciarUniverso(mejores_nuevos_valores)

    def crossover(self, parent1, parent2):
        # Cruce más detallado para los pesos y sesgos de las redes neuronales
        child1 = self.crear_red_neuronal()
        child2 = self.crear_red_neuronal()
        for i in range(len(parent1.get_weights())):
            shape = parent1.get_weights()[i].shape
            if np.prod(shape) <= 2:
                continue  # Saltar a la siguiente iteración si el producto es <= 2
            punto_cruce = random.randint(1, np.prod(shape) - 1)
            weights1 = np.concatenate(
                (parent1.get_weights()[i].flatten()[:punto_cruce], parent2.get_weights()[i].flatten()[punto_cruce:]))
            weights2 = np.concatenate(
                (parent2.get_weights()[i].flatten()[:punto_cruce], parent1.get_weights()[i].flatten()[punto_cruce:]))
            child1.get_weights()[i] = weights1.reshape(shape)
            child2.get_weights()[i] = weights2.reshape(shape)
        return child1, child2
