from types_universo import NodoInterface, IPhysicsRules
from time_procedural import calcular_energia, intercambiar_cargas, relacionar_nodos
from concurrent.futures import ThreadPoolExecutor, wait
import cupy as cp

# Funciones relacionadas con las cargas y la energía
def cargas(nodo: NodoInterface, valores_sistema: IPhysicsRules):
    fluctuacion = (cp.random.random() * 2 - 1) * valores_sistema.FLUCTUACION_MAXIMA

    if cp.random.random() < valores_sistema.PROBABILIDAD_TRANSICION:
        nodo.memoria.cargas = -nodo.memoria.cargas

    nodo.memoria.cargas += fluctuacion if cp.random.random() < 0.5 else -fluctuacion

    if nodo.memoria.cargas > 0.5 and cp.random.random() < valores_sistema.PROBABILIDAD_TUNEL:
        nodo.memoria.cargas = 0

    nodo.memoria.cargas = cp.min(cp.max(nodo.memoria.cargas, -1), 1).tolist()
    nodo.memoria.energia = (1 - cp.abs(nodo.memoria.cargas)).tolist()


def proceso_de_vida_o_muerte(nodo: NodoInterface):
    nodo.memoria.energia = calcular_energia(nodo)


# Funciones relacionadas con la estructura del nodo y los vecinos
def obtener_vecinos(nodos, valores_sistema: IPhysicsRules, i, j):
    FILAS, COLUMNAS = valores_sistema.FILAS, valores_sistema.COLUMNAS
    indices = [
        (i + di) * COLUMNAS + (j + dj)
        for di in range(-1, 2)
        for dj in range(-1, 2)
        if 0 <= i + di < FILAS and 0 <= j + dj < COLUMNAS and (di, dj) != (0, 0)
    ]
    return [nodos[indice] for indice in indices]


def es_parte_de_grupo_circular(valores_sistema: IPhysicsRules, nodo: NodoInterface, vecinos):
    return (len(vecinos) >= valores_sistema.LIMITE_RELACIONAL and
            len(nodo.memoria.relaciones) >= valores_sistema.LIMITE_RELACIONAL)


# Función principal para el siguiente paso en la simulación
def next_step(nodos, valores_sistema: IPhysicsRules, num_threads=4):
    step = valores_sistema.FILAS // num_threads

    def process_rows(start_row, end_row):
        for i in range(start_row, end_row):
            for j in range(valores_sistema.COLUMNAS):
                nodo = nodos[i * valores_sistema.COLUMNAS + j]
                vecinos = obtener_vecinos(nodos, valores_sistema, i, j)
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
        futures = [executor.submit(process_rows, i * step, (i + 1) * step if i != num_threads - 1 else valores_sistema.FILAS) for i in range(num_threads)]
        wait(futures)

    return nodos
