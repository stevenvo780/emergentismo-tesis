from random import random, uniform
from typing import List
from types_universo import NodoInterface, IPhysicsRules, Memoria, Procesos, Relacion
import cupy as cp

# Funciones relacionadas con la creación de nodos
def crear_nodo(i: int, j: int, cargas: float, energia: float) -> NodoInterface:
    return NodoInterface(
        id=f"nodo-{i}-{j}",
        memoria=Memoria(
            cargas=cargas,
            energia=energia,
            edad=0,
            procesos=Procesos(
                relacionarNodos=relacionar_nodos,
                intercambiarCargas=intercambiar_cargas,
            ),
            relaciones=[],
        )
    )

# Funciones relacionadas con las cargas y la energía
def intercambiar_cargas(valores_sistema: IPhysicsRules, nodo_a: NodoInterface, nodo_b: NodoInterface, es_grupo_circular: bool) -> None:
    carga_compartida = (nodo_a.memoria.cargas + nodo_b.memoria.cargas) / 2

    if es_grupo_circular:
        carga_compartida *= (1 - valores_sistema.FACTOR_ESTABILIDAD)

    nodo_a.memoria.cargas = nodo_b.memoria.cargas = carga_compartida

    # Actualizar la carga compartida en la relación
    for rel in nodo_a.memoria.relaciones + nodo_b.memoria.relaciones:
        if rel.nodoId in {nodo_a.id, nodo_b.id}:
            rel.cargaCompartida = carga_compartida

    nodo_a.memoria.energia = calcular_energia(nodo_a)
    nodo_b.memoria.energia = calcular_energia(nodo_b)


def calcular_energia(nodo: NodoInterface) -> float:
    energia = 1 - cp.abs(nodo.memoria.cargas)
    carga_compartida = cp.sum(cp.abs(cp.array([rel.cargaCompartida for rel in nodo.memoria.relaciones])))
    return float(cp.min(energia + carga_compartida, 1))


# Funciones relacionadas con la estructura del nodo y los vecinos
def calcular_distancia(nodo_a: NodoInterface, nodo_b: NodoInterface) -> float:
    i_a, j_a = map(int, nodo_a.id.split('-')[1:])
    i_b, j_b = map(int, nodo_b.id.split('-')[1:])
    return ((i_a - i_b) ** 2 + (j_a - j_b) ** 2) ** 0.5


def relacionar_nodos(valores_sistema: IPhysicsRules, nodo: NodoInterface, vecinos: List[NodoInterface]):
    if nodo.memoria.energia > valores_sistema.ENERGIA:
        for vecino in vecinos:
            if vecino and vecino.memoria.energia > valores_sistema.ENERGIA and vecino.id != nodo.id and vecino.id > nodo.id:
                diferencia_cargas = abs(nodo.memoria.cargas - vecino.memoria.cargas)
                distancia = calcular_distancia(nodo, vecino)
                if distancia > valores_sistema.DISTANCIA_MAXIMA_RELACION:
                    continue

                probabilidad_relacion = (diferencia_cargas / 2) * (1 / distancia) * valores_sistema.FACTOR_RELACION

                if random() < probabilidad_relacion and ((nodo.memoria.cargas < 0 and vecino.memoria.cargas > 0) or (nodo.memoria.cargas > 0 and vecino.memoria.cargas < 0)):
                    relacion_existente = next((rel for rel in nodo.memoria.relaciones if rel.nodoId == vecino.id), None)
                    if not relacion_existente:
                        carga_compartida = (nodo.memoria.cargas + vecino.memoria.cargas) / 2
                        nodo.memoria.relaciones.append(Relacion(vecino.id, carga_compartida))

    # Reducir gradualmente la carga compartida y eliminar relaciones con carga cero
    nodo.memoria.relaciones = [rel for rel in nodo.memoria.relaciones if (
        abs(rel.cargaCompartida) >= valores_sistema.ENERGIA and nodo.memoria.energia > valores_sistema.ENERGIA)]


# Funciones para expandir el espacio
def expandir_espacio(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> List[NodoInterface]:
    # Añadir filas en la parte inferior
    for i in range(valores_sistema.CRECIMIENTO_X):
        for j in range(valores_sistema.COLUMNAS):
            cargas = cp.random.uniform(-1, 1).tolist()
            energia = (1 - cp.abs(cargas)).tolist()
            nodo = crear_nodo(valores_sistema.FILAS + i, j, cargas, energia)
            nodos.append(nodo)

    # Añadir columnas a la derecha
    for i in range(valores_sistema.FILAS + valores_sistema.CRECIMIENTO_X):
        for j in range(valores_sistema.CRECIMIENTO_Y):
            cargas = cp.random.uniform(-1, 1).tolist()
            energia = (1 - cp.abs(cargas)).tolist()
            nodo = crear_nodo(i, valores_sistema.COLUMNAS + j, cargas, energia)
            nodos.append(nodo)

    # Actualizar los valores del sistema
    valores_sistema.FILAS += valores_sistema.CRECIMIENTO_X
    valores_sistema.COLUMNAS += valores_sistema.CRECIMIENTO_Y

    return nodos
