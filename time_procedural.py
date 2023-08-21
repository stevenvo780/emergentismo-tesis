from typing import List
from types_universo import NodoInterface, IPhysicsRules, systemRules
import numpy as np
import cupy as cp

def intercambiar_cargas_matricial(valores_sistema: IPhysicsRules, nodos: List[NodoInterface], matriz_relaciones: cp.ndarray) -> cp.ndarray:
    cargas = cp.array([nodo.cargas for nodo in nodos], dtype=cp.float32)
    matriz_cargas = matriz_relaciones + cargas[:, None]
    liberar_memoria_gpu()
    return matriz_cargas

def calcular_energia_matricial(nodos: List[NodoInterface], matriz_relaciones: cp.ndarray) -> cp.ndarray:
    matriz_adyacencia = matriz_relaciones
    carga_en_cadena_abs = cp.abs(cp.sum(matriz_adyacencia, axis=1))
    energias = 1 - cp.abs(cp.array([nodo.cargas for nodo in nodos])) + carga_en_cadena_abs
    energias = cp.clip(energias, None, 1)
    liberar_memoria_gpu()
    return energias

def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([list(map(int, nodo.id.split('-')[1:])) for nodo in nodos], dtype=cp.float32)
    diff = coords[:, None, :] - coords
    matriz_distancias = cp.linalg.norm(diff, axis=-1)
    liberar_memoria_gpu()
    return matriz_distancias


def relacionar_nodos_matricial(valores_sistema: IPhysicsRules, nodos: List[NodoInterface], matriz_distancias: cp.ndarray) -> cp.ndarray:
    energias = cp.array([nodo.energia for nodo in nodos])
    cargas = cp.array([nodo.cargas for nodo in nodos])

    energias_mask = energias > valores_sistema.ENERGIA
    cargas_mask = (cargas[:, None] * cargas < 0)
    diferencia_cargas = cp.abs(cargas[:, None] - cargas)
    non_zero_mask = matriz_distancias != 0

    probabilidad_relacion = cp.where(
        non_zero_mask & (matriz_distancias < systemRules.DISTANCIA_MAXIMA_RELACION),
        diferencia_cargas / (2 * matriz_distancias),
        0
    ) * valores_sistema.FACTOR_RELACION

    relacion_mask = cp.random.rand(*probabilidad_relacion.shape) < probabilidad_relacion
    relacion_mask &= cargas_mask & energias_mask[:, None]

    carga_compartida = (cargas[:, None] + cargas) / 2
    matriz_relaciones = cp.where(relacion_mask, carga_compartida, 0)
    liberar_memoria_gpu()
    return matriz_relaciones


def calcular_cargas(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> cp.ndarray:
    cargas = cp.array([float(nodo.cargas) for nodo in nodos], dtype=cp.float32)

    transicion_indices = cp.random.rand(len(nodos)) < valores_sistema.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] *= -1

    fluctuacion = cp.random.uniform(-valores_sistema.FLUCTUACION_MAXIMA,
                                    valores_sistema.FLUCTUACION_MAXIMA, len(nodos))
    cargas += fluctuacion

    aleatorio_factor = cp.random.choice([1, -1], len(nodos))
    cargas += fluctuacion * aleatorio_factor

    tunel_indices = (cargas > 0.5) & (cp.random.rand(len(nodos)) < valores_sistema.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0

    cargas = cp.clip(cargas, -1, 1)
    liberar_memoria_gpu()
    return cargas

def liberar_memoria_gpu():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.MemoryPool().free_all_blocks()