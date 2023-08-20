from time_procedural import expandir_espacio, crear_nodo
from types_universo import NodoInterface, IPhysicsRules, PhysicsRules, systemRules
from space import next_step
import random
from datetime import datetime
from typing import Dict, List

class Universo:
    def __init__(self, physicsRules: 'IPhysicsRules' = PhysicsRules()):
        self.nodos: List['NodoInterface'] = []
        self.physicsRules = physicsRules
        self.id = self.generarId()
        self.determinacionesDelSistema()
        self.tiempo: int = 0

    def generarId(self) -> str:
        return (
            datetime.now().isoformat() +
            '-' +
            str(systemRules.FILAS) +
            '-' +
            str(systemRules.COLUMNAS) +
            '-' +
            str(int(random.random() * 1e9))
        )

    def deserializarId(self, id: str) -> Dict[str, object]:
        partes = id.split('-')
        return {
            'fecha': partes[0],
            'filas': int(partes[1]),
            'columnas': int(partes[2]),
            'randomString': partes[3],
        }

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
            # Asumiendo que expandirEspacio es definido en otro lugar
            expandir_espacio(self.nodos, self.physicsRules)

