from types_universo import PhysicsRules, neuronalRules, ProceduralRules
from space import next_step, expandir_espacio
from time_procedural import calcular_distancias_matricial, calcular_relaciones_matricial, calcular_energia
from uuid import uuid4
import cupy as cp
from threading import Lock


class Universo:
    def __init__(self, physics_rules: 'PhysicsRules' = PhysicsRules(), procedural_rules: 'ProceduralRules' = ProceduralRules()):
        self.physics_rules = physics_rules
        self.procedural_rules = procedural_rules
        self.id = str(uuid4())
        self.tiempo: int = 0
        self.cargasMatriz: cp.ndarray = cp.zeros(
            (self.procedural_rules.FILAS, self.procedural_rules.COLUMNAS), dtype=cp.float16)  # type: ignore # Inicialización agregada
        self.energiasMatriz: cp.ndarray
        self.matriz_distancias: cp.ndarray
        self.lock = Lock()
        self.determinacionesDelSistema()

    def determinacionesDelSistema(self):
        self.cargasMatriz = cp.random.uniform(-1, 1,
                                              (self.procedural_rules.FILAS, self.procedural_rules.COLUMNAS))
        mask = cp.random.rand(
            self.procedural_rules.FILAS, self.procedural_rules.COLUMNAS) > self.physics_rules.PROBABILIDAD_VIDA_INICIAL
        self.cargasMatriz[mask] = 0
        self.energiasMatriz = calcular_energia(cp.ones_like(
            self.cargasMatriz), self.cargasMatriz, self.physics_rules)
        self.matriz_distancias = calcular_distancias_matricial(
            self.procedural_rules.FILAS, self.procedural_rules.COLUMNAS)

    def obtener_relaciones(self):
        return calcular_relaciones_matricial(self.physics_rules, self.cargasMatriz, self.matriz_distancias)

    def next(self):
        self.cargasMatriz, self.energiasMatriz = next_step(self)
        if self.tiempo % neuronalRules.CONSTANTE_HUBBLE == 0:
            self.cargasMatriz = expandir_espacio(self, self.cargasMatriz)
            self.matriz_distancias = calcular_distancias_matricial(
                self.procedural_rules.FILAS, self.procedural_rules.COLUMNAS)
            self.energiasMatriz = calcular_energia(cp.ones_like(
                self.cargasMatriz), self.cargasMatriz, self.physics_rules)
