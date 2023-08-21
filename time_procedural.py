from typing import List
from types_universo import NodoInterface, IPhysicsRules, systemRules
import numpy as np
import cupy as cp


def intercambiar_cargas_matricial(valores_sistema: IPhysicsRules, nodos: List[NodoInterface], matriz_relaciones: cp.ndarray) -> cp.ndarray:
    matriz_cargas = matriz_relaciones  # Ya tenemos la matriz de relaciones calculada
    cargas = cp.array([nodo.cargas for nodo in nodos], dtype=cp.float32)
    matriz_cargas += (cargas[:, None] + cargas) / 2
    liberar_memoria_gpu()
    return matriz_cargas


def calcular_energia_matricial(nodos: List[NodoInterface], matriz_relaciones: cp.ndarray) -> cp.ndarray:
    # Utilizamos la matriz de relaciones ya calculada
    matriz_adyacencia = matriz_relaciones

    # Calcular la carga en cadena para cada nodo
    carga_en_cadena = cp.sum(matriz_adyacencia, axis=1)

    # Calcular la energía para cada nodo
    energias = 1 - cp.abs(cp.array([nodo.cargas for nodo in nodos])) + cp.abs(carga_en_cadena)
    energias = cp.clip(energias, None, 1)
    liberar_memoria_gpu()
    return energias


def liberar_memoria_gpu():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.MemoryPool().free_all_blocks()


def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([list(map(int, nodo.id.split('-')[1:]))
                      for nodo in nodos], dtype=cp.float32)
    i_diff = coords[:, 0][:, None] - coords[:, 0]
    j_diff = coords[:, 1][:, None] - coords[:, 1]
    matriz_distancias = cp.hypot(i_diff, j_diff)
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
        non_zero_mask & (matriz_distancias <
                         systemRules.DISTANCIA_MAXIMA_RELACION),
        diferencia_cargas / 2 / matriz_distancias,
        0
    ) * valores_sistema.FACTOR_RELACION

    relacion_mask = (cp.random.rand(*probabilidad_relacion.shape)
                     < probabilidad_relacion) & cargas_mask
    relacion_mask &= energias_mask[:, None]

    carga_compartida = (cargas[:, None] + cargas) / 2
    matriz_relaciones = cp.where(relacion_mask, carga_compartida, 0)
    liberar_memoria_gpu()
    return matriz_relaciones


def calcular_cargas(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> cp.ndarray:
    # Crear un array de cargas en la GPU utilizando CuPy
    cargas = cp.array([float(nodo.cargas) for nodo in nodos], dtype=cp.float32)

    # Aplicar la probabilidad de transición
    transicion_indices = cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] *= -1

    # Aplicar fluctuación
    fluctuacion = cp.random.uniform(-valores_sistema.FLUCTUACION_MAXIMA,
                                    valores_sistema.FLUCTUACION_MAXIMA, len(nodos))
    cargas += fluctuacion

    # Aplicar fluctuación aleatoria
    aleatorio_indices = cp.random.rand(len(nodos)) < 0.5
    cargas += fluctuacion * (1 - 2 * aleatorio_indices)

    # Aplicar probabilidad de túnel
    tunel_indices = (cargas > 0.5) & (cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0

    # Limitar las cargas a [-1, 1]
    cargas = cp.clip(cargas, -1, 1)
    liberar_memoria_gpu()
    return cargas
