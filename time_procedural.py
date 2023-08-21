from typing import List
from types_universo import NodoInterface, IPhysicsRules, SystemRules, systemRules

import cupy as cp


def intercambiar_cargas_matricial(valores_sistema, nodos, matriz_relaciones):
    cargas = cp.array([nodo.cargas for nodo in nodos], dtype=cp.float16)
    diferencias_cargas = (cargas[:, None] - cargas) / 2
    diferencias_cargas = cp.clip(diferencias_cargas, - systemRules.LIMITE_INTERCAMBIO, systemRules.LIMITE_INTERCAMBIO)
    mask_intercambio = (diferencias_cargas != 0) & \
                       (cargas[:, None] * cargas < 0) & \
                       (matriz_relaciones > 0)
    diferencias_cargas *= mask_intercambio
    cargas_nuevas = cargas - cp.sum(diferencias_cargas, axis=1)

    matriz_cargas = matriz_relaciones * diferencias_cargas

    return cargas_nuevas, matriz_cargas


def calcular_energia_matricial(nodos: List[NodoInterface], matriz_relaciones: cp.ndarray) -> cp.ndarray:
    matriz_adyacencia = matriz_relaciones
    carga_en_cadena_abs = cp.abs(cp.sum(matriz_adyacencia, axis=1))
    liberar_memoria_gpu()  # Limpieza de memoria
    energias = 1 - \
        cp.abs(cp.array([nodo.cargas for nodo in nodos])) + carga_en_cadena_abs
    energias = cp.clip(energias, None, 1)
    liberar_memoria_gpu()
    return energias


def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([list(map(int, nodo.id.split('-')[1:]))
                      for nodo in nodos], dtype=cp.float16)
    diff = coords[:, None, :] - coords
    matriz_distancias = cp.linalg.norm(diff, axis=-1)
    liberar_memoria_gpu()  # Limpieza de memoria
    return matriz_distancias


def relacionar_nodos_matricial(valores_sistema: IPhysicsRules, nodos: List[NodoInterface], matriz_distancias: cp.ndarray) -> cp.ndarray:
    energias = cp.array([nodo.energia for nodo in nodos])
    cargas = cp.array([nodo.cargas for nodo in nodos])

    mask = matriz_distancias == 1

    diferencia_energias = cp.where(mask, cp.abs(energias[:, None] - energias), 0)
    diferencia_cargas = cp.where(mask, cp.abs(cargas[:, None] - cargas), 0)

    relacion_mask = mask & (diferencia_energias > valores_sistema.ENERGIA)
    carga_compartida = cp.where(relacion_mask, diferencia_cargas, 0)

    liberar_memoria_gpu()

    return carga_compartida



def calcular_cargas(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> cp.ndarray:
    cargas = cp.array([float(nodo.cargas) for nodo in nodos], dtype=cp.float16)

    transicion_indices = cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] *= -1

    fluctuacion = cp.random.uniform(-valores_sistema.FLUCTUACION_MAXIMA,
                                    valores_sistema.FLUCTUACION_MAXIMA, len(nodos))
    cargas += fluctuacion

    aleatorio_factor = cp.random.choice([1, -1], len(nodos))
    cargas += fluctuacion * aleatorio_factor

    tunel_indices = (cargas > 0.5) & (cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0

    cargas = cp.clip(cargas, -1, 1)
    liberar_memoria_gpu()  # Limpieza de memoria
    return cargas


def liberar_memoria_gpu():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.MemoryPool().free_all_blocks()
