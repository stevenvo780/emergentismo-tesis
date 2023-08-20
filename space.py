from types_universo import NodoInterface, systemRules, IPhysicsRules
from time_procedural import calcular_energia, intercambiar_cargas, relacionar_nodos
import random
from concurrent.futures import ThreadPoolExecutor, wait

def cargas(nodo: NodoInterface, valores_sistema: IPhysicsRules):
    if random.random() < valores_sistema.PROBABILIDAD_TRANSICION:
        nodo.memoria.cargas = -nodo.memoria.cargas

    fluctuacion = (random.random() * 2 - 1) * valores_sistema.FLUCTUACION_MAXIMA
    nodo.memoria.cargas += fluctuacion

    if random.random() < 0.5:
        nodo.memoria.cargas -= fluctuacion
    else:
        nodo.memoria.cargas += fluctuacion

    if nodo.memoria.cargas > 0.5 and random.random() < valores_sistema.PROBABILIDAD_TUNEL:
        nodo.memoria.cargas = 0

    nodo.memoria.cargas = min(max(nodo.memoria.cargas, -1), 1)
    nodo.memoria.energia = 1 - abs(nodo.memoria.cargas)

def es_parte_de_grupo_circular(valores_sistema: IPhysicsRules, nodo: NodoInterface, vecinos):
    return (len(vecinos) >= systemRules.LIMITE_RELACIONAL and
            len(nodo.memoria.relaciones) >= systemRules.LIMITE_RELACIONAL)  # Cambio aquí

def obtener_vecinos(nodos, valores_sistema: IPhysicsRules, i, j):
    FILAS = systemRules.FILAS
    COLUMNAS = systemRules.COLUMNAS

    indices_vecinos = [
        (i - 1) * COLUMNAS + (j - 1) if i > 0 and j > 0 else -1,
        (i - 1) * COLUMNAS + j if i > 0 else -1,
        (i - 1) * COLUMNAS + (j + 1) if i > 0 and j < COLUMNAS - 1 else -1,
        i * COLUMNAS + (j - 1) if j > 0 else -1,
        i * COLUMNAS + (j + 1) if j < COLUMNAS - 1 else -1,
        (i + 1) * COLUMNAS + (j - 1) if i < FILAS - 1 and j > 0 else -1,
        (i + 1) * COLUMNAS + j if i < FILAS - 1 else -1,
        (i + 1) * COLUMNAS + (j + 1) if i < FILAS - 1 and j < COLUMNAS - 1 else -1,
    ]

    return [nodos[indice] for indice in indices_vecinos if 0 <= indice < len(nodos)]

def proceso_de_vida_o_muerte(nodo: NodoInterface):
    nodo.memoria.energia = calcular_energia(nodo)

def next_step(nodos, valores_sistema: IPhysicsRules):
    step = systemRules.FILAS // systemRules.NUM_THREADS
    result = nodos.copy()

    def process_rows(start_row, end_row):
        nonlocal result
        for i in range(start_row, end_row):
            for j in range(systemRules.COLUMNAS):
                nodo = result[i * systemRules.COLUMNAS + j]
                vecinos = obtener_vecinos(result, valores_sistema, i, j)
                if not vecinos or not nodo:
                    print('Error al relacionar los nodos:', len(nodos))
                    continue
                es_grupo_circular = es_parte_de_grupo_circular(valores_sistema, nodo, vecinos)
                cargas(nodo, valores_sistema)
                proceso_de_vida_o_muerte(nodo)
                relacionar_nodos(valores_sistema, nodo, vecinos)
                for vecino in vecinos:
                    if (
                        (nodo.memoria.cargas < 0 and vecino.memoria.cargas > 0) or
                        (nodo.memoria.cargas > 0 and vecino.memoria.cargas < 0)
                    ):
                        intercambiar_cargas(valores_sistema, nodo, vecino, es_grupo_circular)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_rows, i * step, (i + 1) * step if i != systemRules.NUM_THREADS - 1 else systemRules.FILAS) for i in range(systemRules.NUM_THREADS)]
        wait(futures)

    return result
