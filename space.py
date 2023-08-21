from types_universo import NodoInterface, systemRules, IPhysicsRules, Relacion
from time_procedural import calcular_energia_matricial, intercambiar_cargas_matricial, calcular_distancias_matricial, relacionar_nodos_matricial, calcular_cargas, liberar_memoria_gpu
from random import uniform
from typing import List
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np

def next_step(nodos: List[NodoInterface], valores_sistema: IPhysicsRules):
    print("Calculando distancias matriciales...")
    matriz_distancias = calcular_distancias_matricial(nodos)
    print("Calculando relaciones matriciales...")
    matriz_relaciones = relacionar_nodos_matricial(
        valores_sistema, nodos, matriz_distancias)
    del matriz_distancias
    print("Calculando nuevas cargas...")
    cargas_nuevas = calcular_cargas(nodos, valores_sistema)
    print("Calculando energías matriciales...")
    energias = calcular_energia_matricial(nodos, matriz_relaciones)
    print("Intercambiando cargas matriciales...")
    matriz_cargas = intercambiar_cargas_matricial(
        valores_sistema, nodos, matriz_relaciones)
    cargas_nuevas += np.sum(matriz_cargas, axis=1)
    del matriz_cargas
    liberar_memoria_gpu()
    for i, nodo in enumerate(nodos):
        nodo.cargas = cargas_nuevas[i].tolist()
        nodo.energia = energias[i].tolist()
        nodo.relaciones_matriz = matriz_relaciones[i]
    del matriz_relaciones
    del cargas_nuevas
    print("Proceso completado.")
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
            nodo = crear_nodo(i, systemRules.COLUMNAS + j, cargas, energia)
            nodos.append(nodo)

    systemRules.FILAS += systemRules.CRECIMIENTO_X
    systemRules.COLUMNAS += systemRules.CRECIMIENTO_Y

    return nodos
