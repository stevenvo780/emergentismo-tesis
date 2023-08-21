from typing import List
from types_universo import NodoInterface, IPhysicsRules, systemRules
import numpy as np
import cupy as cp


def intercambiar_cargas_matricial(valores_sistema: IPhysicsRules, nodos: List[NodoInterface], es_grupo_circular_matriz: cp.ndarray) -> cp.ndarray:
    # Crear una matriz de adyacencia para las cargas
    matriz_cargas = cp.zeros((len(nodos), len(nodos)), dtype=cp.float32)
    relaciones = [(i, next((idx for idx, n in enumerate(nodos) if n.id == rel.nodoId), -1),
                   rel.cargaCompartida) for i, nodo in enumerate(nodos) for rel in nodo.relaciones]
    
    # Usar operaciones de matriz para llenar la matriz de cargas
    indices_i, indices_j, valores = zip(*relaciones)
    matriz_cargas[indices_i, indices_j] = valores

    # Calcular las cargas compartidas
    cargas = cp.array([nodo.cargas for nodo in nodos], dtype=cp.float32)
    matriz_cargas += (cargas[:, None] + cargas) / 2

    # Aplicar el factor de estabilidad si es un grupo circular
    matriz_cargas *= (1 - es_grupo_circular_matriz * valores_sistema.FACTOR_ESTABILIDAD)

    return matriz_cargas


def calcular_energia_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    # Crear la matriz de adyacencia
    matriz_adyacencia = cp.zeros((len(nodos), len(nodos)))
    for idx, n in enumerate(nodos):
        for rel in n.relaciones:
            vecino_idx = next((i for i, v in enumerate(nodos)
                              if v.id == rel.nodoId), -1)
            if vecino_idx != -1:
                matriz_adyacencia[idx, vecino_idx] = rel.cargaCompartida

    # Calcular la carga en cadena para cada nodo
    carga_en_cadena = cp.sum(matriz_adyacencia, axis=1)

    # Calcular la energía para cada nodo
    energias = 1 - \
        cp.abs(cp.array([nodo.cargas for nodo in nodos])
               ) + cp.abs(carga_en_cadena)
    energias = cp.clip(energias, None, 1)

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

    return matriz_relaciones


def calcular_cargas(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> cp.ndarray:
    # Crear un array de cargas en la GPU utilizando CuPy
    cargas = cp.array([float(nodo.cargas) for nodo in nodos], dtype=cp.float32)

    # Aplicar la probabilidad de transición
    transicion_indices = cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] = -cargas[transicion_indices]

    # Aplicar fluctuación
    fluctuacion = (cp.random.rand(len(nodos)) * 2 - 1) * \
        valores_sistema.FLUCTUACION_MAXIMA
    cargas += fluctuacion

    # Aplicar fluctuación aleatoria
    aleatorio_indices = cp.random.rand(len(nodos)) < 0.5
    cargas[aleatorio_indices] -= fluctuacion[aleatorio_indices]
    cargas[~aleatorio_indices] += fluctuacion[~aleatorio_indices]

    # Aplicar probabilidad de túnel
    tunel_indices = (cargas > 0.5) & (cp.random.rand(
        len(nodos)) < valores_sistema.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0

    # Limitar las cargas a [-1, 1]
    cargas = cp.clip(cargas, -1, 1)
    return cargas
