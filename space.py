from types_universo import NodoInterface, systemRules, IPhysicsRules, Relacion
from time_procedural import calcular_energia_matricial, calcular_cargas, calcular_distancias_matricial, relacionar_nodos_matricial
from random import uniform
from typing import List
import cupy as cp

def next_step(universo):
    matriz_distancias = universo.matriz_distancias

    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()

    with stream1:
        matriz_relaciones = relacionar_nodos_matricial(
            universo.physics_rules, universo.energiasMatriz, universo.cargasMatriz, matriz_distancias)

    with stream2:
        cargas_nuevas = calcular_cargas(
            universo.cargasMatriz, matriz_relaciones, universo.physics_rules)

    with stream3:
        energias = calcular_energia_matricial(
            universo.energiasMatriz, matriz_relaciones)

    return cargas_nuevas, energias, matriz_relaciones


def crear_nodo(i: int, j: int, cargas: float, energia: float) -> NodoInterface:
    return NodoInterface(
        id=f"nodo-{i}-{j}",
        cargas=cargas,
        energia=energia,
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
