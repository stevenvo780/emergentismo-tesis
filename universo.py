from types_universo import NodoInterface, IPhysicsRules, PhysicsRules, systemRules
from space import next_step, expandir_espacio, crear_nodo
import random
from uuid import uuid4
from typing import List

class Universo:
    def __init__(self, physicsRules: 'IPhysicsRules' = PhysicsRules()):
        self.nodos: List['NodoInterface'] = []
        self.physicsRules = physicsRules
        self.id = str(uuid4())
        self.determinacionesDelSistema()
        self.tiempo: int = 0

    def determinacionesDelSistema(self):
        for i in range(systemRules.FILAS):
            for j in range(systemRules.COLUMNAS):
                cargas = random.random() * 2 - 1
                energia = 1 - abs(cargas)
                if random.random() > self.physicsRules.PROBABILIDAD_VIDA_INICIAL:
                    cargas = 0
                    energia = 0
                nodo = crear_nodo(i, j, cargas, energia)
                self.nodos.append(nodo)

    def next(self):
        self.nodos = next_step(self.nodos, self.physicsRules)
        if self.tiempo % 100 == 0:
            expandir_espacio(self.nodos)
