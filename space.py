from types_universo import NodoInterface, IPhysicsRules
from time_procedural import calcular_energia, intercambiar_cargas, relacionar_nodos
import random

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
    return (len(vecinos) >= valores_sistema.LIMITE_RELACIONAL and
            len(nodo.memoria.relaciones) >= valores_sistema.LIMITE_RELACIONAL)  # Cambio aquí

def obtener_vecinos(nodos, valores_sistema: IPhysicsRules, i, j):
    FILAS = valores_sistema.FILAS
    COLUMNAS = valores_sistema.COLUMNAS

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
    nodo.memoria.energia = calcular_energia(nodo)  # Cambio aquí

def next_step(nodos, valores_sistema: IPhysicsRules):
    nueva_generacion = nodos
    for i in range(valores_sistema.FILAS):
        for j in range(valores_sistema.COLUMNAS):
            nodo = nueva_generacion[i * valores_sistema.COLUMNAS + j]
            vecinos = obtener_vecinos(nueva_generacion, valores_sistema, i, j)
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

    return nueva_generacion
