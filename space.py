from types_universo import NodoInterface, systemRules, IPhysicsRules, Relacion
from time_procedural import calcular_energia_matricial, intercambiar_cargas_matricial, calcular_distancias_matricial, relacionar_nodos_matricial, calcular_cargas, liberar_memoria_gpu
from random import uniform
from typing import List
import cupy as cp


def next_step(nodos: List[NodoInterface], valores_sistema: IPhysicsRules):
    # 1. Calculate and apply charges
    cargas = calcular_cargas(nodos, valores_sistema)
    for i, nodo in enumerate(nodos):
        nodo.cargas = cargas[i]

    # 2. Calculate and apply energy
    energias = calcular_energia_matricial(nodos)
    for i, nodo in enumerate(nodos):
        nodo.energia = energias[i]

    # 3. Calculate and apply node relations
    matriz_distancias = calcular_distancias_matricial(nodos)
    matriz_relaciones = relacionar_nodos_matricial(
        valores_sistema, nodos, matriz_distancias)
    # Apply relations to nodes (This needs to be defined properly based on your logic)

    # 4. Calculate and apply charge exchange
    es_grupo_circular_matriz = cp.zeros((len(nodos), len(nodos)))  # This needs to be defined properly
    matriz_cargas = intercambiar_cargas_matricial(
        valores_sistema, nodos, es_grupo_circular_matriz)
    # Apply charge exchange to nodes (This needs to be defined properly based on your logic)

    return nodos

def crear_nodo(i: int, j: int, cargas: float, energia: float) -> NodoInterface:
    return NodoInterface(
        id=f"nodo-{i}-{j}",
        cargas=cargas,
        energia=energia,
        relaciones=[],
    )


def expandir_espacio(nodos: List[NodoInterface]) -> List[NodoInterface]:
    # Añadir filas en la parte inferior
    for i in range(systemRules.CRECIMIENTO_X):
        for j in range(systemRules.COLUMNAS):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(systemRules.FILAS + i, j, cargas, energia)
            nodos.append(nodo)

    # Añadir columnas a la derecha
    for i in range(systemRules.FILAS + systemRules.CRECIMIENTO_X):
        for j in range(systemRules.CRECIMIENTO_Y):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(i, systemRules.COLUMNAS + j, cargas, energia)
            nodos.append(nodo)

    # Actualizar los valores del sistema
    systemRules.FILAS += systemRules.CRECIMIENTO_X
    systemRules.COLUMNAS += systemRules.CRECIMIENTO_Y

    return nodos


def es_parte_de_grupo_circular(valores_sistema: IPhysicsRules, nodo: NodoInterface, vecinos):
    return (len(vecinos) >= systemRules.LIMITE_RELACIONAL and
            len(nodo.relaciones) >= systemRules.LIMITE_RELACIONAL)  # Cambio aquí
