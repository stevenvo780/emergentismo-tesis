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
        self.physics_rules = physics_rules
        self.id = str(uuid4())
        self.tiempo: int = 0
        self.cargasMatriz: cp.ndarray = cp.zeros(
            (systemRules.FILAS, systemRules.COLUMNAS), dtype=cp.float16)  # Inicialización agregada
        self.energiasMatriz: cp.ndarray
        self.matriz_distancias: cp.ndarray
        self.lock = Lock()
        self.determinacionesDelSistema()

    def determinacionesDelSistema(self):
        self.cargasMatriz = cp.random.uniform(-1, 1,
                                              (systemRules.FILAS, systemRules.COLUMNAS))
        mask = cp.random.rand(
            systemRules.FILAS, systemRules.COLUMNAS) > self.physics_rules.PROBABILIDAD_VIDA_INICIAL
        self.cargasMatriz[mask] = 0

        self.energiasMatriz = calcular_energia(cp.ones_like(
            self.cargasMatriz), self.cargasMatriz, self.physics_rules)
        self.matriz_distancias = calcular_distancias_matricial(
            systemRules.FILAS, systemRules.COLUMNAS)

    def obtener_relaciones(self):
        return calcular_relaciones_matricial(self.physics_rules, self.cargasMatriz, self.matriz_distancias)

    def next(self):
        self.cargasMatriz, self.energiasMatriz = next_step(self)
        if self.tiempo % systemRules.CONSTANTE_HUBBLE == 0:
            # self.cargasMatriz = expandir_espacio(self.cargasMatriz)
            self.matriz_distancias = calcular_distancias_matricial(
                systemRules.FILAS, systemRules.COLUMNAS)
            self.energiasMatriz = calcular_energia(cp.ones_like(
                self.cargasMatriz), self.cargasMatriz, self.physics_rules)
