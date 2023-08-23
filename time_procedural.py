from typing import List
from types_universo import NodoInterface, PhysicsRules, systemRules
import cupy as cp

def intercambiar_cargas_matricial(cargas: cp.ndarray, matriz_relaciones):
    diferencias_cargas = (cargas[:, None] - cargas) / 2
    mask_intercambio = (diferencias_cargas != 0) & (cargas[:, None] * cargas < 0) & (matriz_relaciones > 0)
    diferencias_cargas *= mask_intercambio
    return cargas - cp.sum(diferencias_cargas, axis=1)

def calcular_energia_matricial(energias: cp.ndarray, matriz_relaciones: cp.ndarray) -> cp.ndarray:
    return cp.clip(1 - cp.abs(energias) + cp.abs(cp.sum(matriz_relaciones, axis=1)), None, 1)

def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([nodo.id.split('-')[1:] for nodo in nodos], dtype=cp.int16)
    diff = coords[:, None, :] - coords
    return cp.sqrt(cp.sum(diff ** 2, axis=-1))

def relacionar_nodos_matricial(physics_rules: PhysicsRules, energias: cp.ndarray, cargas: cp.ndarray, matriz_distancias: cp.ndarray) -> cp.ndarray:
    mask = matriz_distancias == 1
    diferencia_energias = cp.where(mask, cp.abs(energias[:, None] - energias), 0)
    diferencia_cargas = cp.where(mask, cp.abs(cargas[:, None] - cargas), 0)
    relacion_mask = mask & (diferencia_energias > physics_rules.ENERGIA - systemRules.TOLERANCIA_ENERGIA)
    return cp.where(relacion_mask, diferencia_cargas, 0)

def simulacion_estocastica(cargas: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
    transicion_indices = cp.random.rand(len(cargas)) < physics_rules.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] *= -1
    fluctuacion = cp.random.uniform(-physics_rules.FLUCTUACION_MAXIMA, physics_rules.FLUCTUACION_MAXIMA, len(cargas), dtype=cp.float32) # type: ignore
    cargas += fluctuacion * cp.random.choice([1, -1], len(cargas))
    tunel_indices = (cargas > 0.5) & (cp.random.rand(len(cargas)) < physics_rules.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0
    return cp.clip(cargas, -systemRules.LIMITE_INTERCAMBIO, systemRules.LIMITE_INTERCAMBIO)

def calcular_cargas(cargas: cp.ndarray, matriz_relaciones: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
    return simulacion_estocastica(intercambiar_cargas_matricial(cargas, matriz_relaciones), physics_rules)

def liberar_memoria_gpu():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.MemoryPool().free_all_blocks()
