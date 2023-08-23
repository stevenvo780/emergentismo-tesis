from types_universo import NodoInterface, systemRules, IPhysicsRules, Relacion
from time_procedural import calcular_energia_matricial, calcular_cargas, calcular_distancias_matricial, relacionar_nodos_matricial
from random import uniform
from typing import List
import cupy as cp
import subprocess


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


def obtener_memoria_disponible():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                                '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        memoria_disponible = int(result.stdout.decode('utf-8').strip())
        return memoria_disponible * 1024 * 1024  # Convertir a bytes
    except Exception as e:
        print("Error al obtener la memoria disponible de la GPU:", e)
        return 0


def expandir_espacio(nodos: List[NodoInterface]) -> List[NodoInterface]:
    memoria_disponible = obtener_memoria_disponible()
    crecimiento_permitido = int(
        memoria_disponible // systemRules.MEMORIA_POR_FILA * systemRules.FILAS_POR_GB)

    crecimiento_x = min(systemRules.CRECIMIENTO_X, crecimiento_permitido)
    crecimiento_y = min(systemRules.CRECIMIENTO_Y, crecimiento_permitido)

    for i in range(crecimiento_x):
        for j in range(systemRules.COLUMNAS):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(systemRules.FILAS + i, j, cargas, energia)
            nodos.append(nodo)

    for i in range(systemRules.FILAS + crecimiento_x):
        for j in range(crecimiento_y):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(i, systemRules.COLUMNAS +
                              j, cargas, energia)
            nodos.append(nodo)

    systemRules.FILAS += crecimiento_x
    systemRules.COLUMNAS += crecimiento_y

    return nodos
