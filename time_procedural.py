from typing import List
from types_universo import PhysicsRules, systemRules, NodoInterface
import cupy as cp

def calcular_cargas(cargas: cp.ndarray, matriz_distancias: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
    vecinos_vivos = cp.sum(matriz_distancias == 1, axis=1)
    # Usar una probabilidad que dependa del número de vecinos vivos para el nacimiento
    nacimiento = (cargas == 0) & (cp.random.rand(len(cargas)) <
                                  physics_rules.PROBABILIDAD_VIDA_INICIAL * vecinos_vivos)
    # Usar una función logística para la supervivencia
    supervivencia = (cargas != 0) & (cp.random.rand(len(cargas)) < 1 / (1 +
                                                                        cp.exp(-physics_rules.FACTOR_ESTABILIDAD * (vecinos_vivos - physics_rules.PROBABILIDAD_SUPERVIVENCIA))))
    muerte = ~nacimiento & ~supervivencia
    cargas_nuevas = cargas.copy()
    cargas_nuevas[nacimiento] = cp.random.uniform(
        -1, 1, cp.sum(nacimiento).item())
    cargas_nuevas[muerte] = 0

    # Añade una interacción estocástica no lineal para mayor complejidad
    fluctuacion = cp.sin(cargas_nuevas) * cp.random.uniform(-physics_rules.FLUCTUACION_MAXIMA,
                                                            physics_rules.FLUCTUACION_MAXIMA, len(cargas_nuevas), dtype=cp.float32)
    cargas_nuevas += fluctuacion
    return cp.clip(cargas_nuevas, -systemRules.LIMITE_INTERCAMBIO, systemRules.LIMITE_INTERCAMBIO)


def calcular_energia(energias: cp.ndarray, cargas: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
    # Usar una función escalón para la energía
    return cp.where(cp.abs(energias + cargas) > physics_rules.UMBRAL_CARGA, 1, 0)


def calcular_relaciones_matricial(physics_rules: PhysicsRules, cargas: cp.ndarray, matriz_distancias: cp.ndarray) -> cp.ndarray:
    mask = matriz_distancias == 1
    diferencia_cargas = cp.where(mask, cp.abs(cargas[:, None] - cargas), 0)
    relacion_mask = mask & (diferencia_cargas > physics_rules.ENERGIA)
    return cp.where(relacion_mask, cp.exp(diferencia_cargas), 0)


def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([nodo.id.split('-')[1:]
                      for nodo in nodos], dtype=cp.int16)
    diff = coords[:, None, :] - coords
    return cp.sqrt(cp.sum(diff ** 2, axis=-1))
