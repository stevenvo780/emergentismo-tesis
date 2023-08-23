from types_universo import NodoInterface, PhysicsRules, systemRules, Relacion
from space import next_step, expandir_espacio, crear_nodo
from time_procedural import calcular_distancias_matricial
import random
from uuid import uuid4
from typing import List
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def update_node(nodo, carga_nueva, energia_nueva, matriz_relacion, nodos):
    nodo.cargas = carga_nueva.tolist()
    nodo.energia = energia_nueva.tolist()
    nodo.relaciones = [Relacion(nodoId=nodos[j].id, cargaCompartida=carga_compartida)
                       for j, carga_compartida in enumerate(matriz_relacion) if carga_compartida != 0]

class Universo:
    def __init__(self, physics_rules: 'PhysicsRules' = PhysicsRules()):
        self.nodos: List['NodoInterface'] = []
        self.physics_rules = physics_rules
        self.id = str(uuid4())
        self.determinacionesDelSistema()
        self.tiempo: int = 0
        self.energiasMatriz: cp.ndarray
        self.cargasMatriz: cp.ndarray
        self.matriz_distancias: cp.ndarray
        self.matriz_relaciones: cp.ndarray = cp.zeros_like(self.matriz_distancias)
        self.lock = Lock()

    def determinacionesDelSistema(self):
        for i in range(systemRules.FILAS):
            for j in range(systemRules.COLUMNAS):
                cargas = random.random() * 2 - 1
                energia = 1 - abs(cargas)
                if random.random() > self.physics_rules.PROBABILIDAD_VIDA_INICIAL:
                    cargas = 0
                    energia = 0
                nodo = crear_nodo(i, j, cargas, energia)
                self.nodos.append(nodo)
        self.energiasMatriz = cp.array(
            [nodo.energia for nodo in self.nodos], dtype=cp.float16)
        self.cargasMatriz = cp.array(
            [nodo.cargas for nodo in self.nodos], dtype=cp.float16)
        self.matriz_distancias = calcular_distancias_matricial(self.nodos)
        self.state = True

    def actualizar_nodos(self):
        try:
            with ThreadPoolExecutor() as executor:
                for i, nodo in enumerate(self.nodos):
                    executor.submit(update_node, nodo,
                                    self.cargasMatriz[i], self.energiasMatriz[i], self.matriz_relaciones[i], self.nodos)
        except Exception as e:
            print(e)

    def next(self):
        self.cargasMatriz, self.energiasMatriz, self.matriz_relaciones = next_step(
            self)
        if self.tiempo % 100 == 0:
            expandir_espacio(self.nodos)
            self.energiasMatriz = cp.array(
                [nodo.energia for nodo in self.nodos], dtype=cp.float16)
            self.cargasMatriz = cp.array(
                [nodo.cargas for nodo in self.nodos], dtype=cp.float16)
            self.matriz_distancias = calcular_distancias_matricial(self.nodos)
