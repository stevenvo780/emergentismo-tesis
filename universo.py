from types_universo import NodoInterface, PhysicsRules, systemRules, Relacion
from space import next_step, expandir_espacio, crear_nodo
from time_procedural import calcular_distancias_matricial, calcular_relaciones_matricial, calcular_energia
import random
from uuid import uuid4
from typing import List
import cupy as cp
from threading import Lock


class Universo:
    def __init__(self, physics_rules: 'PhysicsRules' = PhysicsRules()):
        self.nodos: List['NodoInterface'] = []
        self.physics_rules = physics_rules
        self.id = str(uuid4())
        self.tiempo: int = 0
        self.cargasMatriz: cp.ndarray
        self.energiasMatriz: cp.ndarray
        self.matriz_distancias: cp.ndarray
        self.lock = Lock()
        self.determinacionesDelSistema()

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
        self.cargasMatriz = cp.array(
            [nodo.cargas for nodo in self.nodos], dtype=cp.float16)
        self.energiasMatriz = calcular_energia(cp.ones_like(
            self.cargasMatriz), self.cargasMatriz, self.physics_rules)

        self.matriz_distancias = calcular_distancias_matricial(self.nodos)

    def obtener_relaciones(self):
        return calcular_relaciones_matricial(self.physics_rules, self.cargasMatriz, self.matriz_distancias)

    def next(self):
        self.cargasMatriz, self.energiasMatriz = next_step(self)
        if self.tiempo % systemRules.CONSTANTE_HUBBLE == 0:
            expandir_espacio(self.nodos)
            self.matriz_distancias = calcular_distancias_matricial(self.nodos)
            self.cargasMatriz = cp.array(
                [nodo.cargas for nodo in self.nodos], dtype=cp.float16)
            self.energiasMatriz = calcular_energia(cp.ones_like(
                self.cargasMatriz), self.cargasMatriz, self.physics_rules)
