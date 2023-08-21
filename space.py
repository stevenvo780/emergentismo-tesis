from types_universo import NodoInterface, systemRules, IPhysicsRules, Relacion
from time_procedural import calcular_energia_matricial, intercambiar_cargas_matricial, calcular_distancias_matricial, relacionar_nodos_matricial, calcular_cargas, liberar_memoria_gpu
from random import uniform
from typing import List
from concurrent.futures import ThreadPoolExecutor
from typing import List
import cupy as cp


def next_step(nodos: List[NodoInterface], valores_sistema: IPhysicsRules):
    matriz_distancias = calcular_distancias_matricial(nodos)
    matriz_relaciones = relacionar_nodos_matricial(
        valores_sistema, nodos, matriz_distancias)
    del matriz_distancias
    cargas_nuevas, _ = intercambiar_cargas_matricial(
        valores_sistema, nodos, matriz_relaciones)
    energias = calcular_energia_matricial(nodos, matriz_relaciones)
    liberar_memoria_gpu()
    for i, nodo in enumerate(nodos):
        nodo.cargas = cargas_nuevas[i].tolist()
        nodo.energia = energias[i].tolist()
        nodo.relaciones_matriz = matriz_relaciones[i]
    del matriz_relaciones
    del cargas_nuevas
    print(nodos[0].cargas)
    return nodos


def crear_nodo(i: int, j: int, cargas: float, energia: float) -> NodoInterface:
    return NodoInterface(
        id=f"nodo-{i}-{j}",
        cargas=cargas,
        energia=energia,
        relaciones_matriz=[],
    )


def expandir_espacio(nodos: List[NodoInterface]) -> List[NodoInterface]:
    for i in range(systemRules.CRECIMIENTO_X):
        for j in range(systemRules.COLUMNAS):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(systemRules.FILAS + i, j, cargas, energia)
            nodos.append(nodo)

    for i in range(systemRules.FILAS + systemRules.CRECIMIENTO_X):
        for j in range(systemRules.CRECIMIENTO_Y):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(i, systemRules.COLUMNAS +
                              j, cargas, energia)
            nodos.append(nodo)

    systemRules.FILAS += systemRules.CRECIMIENTO_X
    systemRules.COLUMNAS += systemRules.CRECIMIENTO_Y

    return nodos
