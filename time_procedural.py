from random import random
from typing import List
from types_universo import NodoInterface, IPhysicsRules, Memoria, Procesos, Relacion
from random import uniform


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


def intercambiar_cargas(valores_sistema: IPhysicsRules, nodo_a: NodoInterface, nodo_b: NodoInterface, es_grupo_circular: bool) -> None:
    carga_compartida = (nodo_a.memoria.cargas + nodo_b.memoria.cargas) / 2

    if es_grupo_circular:
        carga_compartida = carga_compartida * \
            (1 - valores_sistema.FACTOR_ESTABILIDAD)

    nodo_a.memoria.cargas = carga_compartida
    nodo_b.memoria.cargas = carga_compartida

    # Actualizar la carga compartida en la relación
    for rel in nodo_a.memoria.relaciones:
        if rel.nodoId == nodo_b.id:
            rel.cargaCompartida = carga_compartida

    for rel in nodo_b.memoria.relaciones:
        if rel.nodoId == nodo_a.id:
            rel.cargaCompartida = carga_compartida

    nodo_a.memoria.energia = calcular_energia(nodo_a)
    nodo_b.memoria.energia = calcular_energia(nodo_b)


def calcular_energia(nodo: NodoInterface) -> float:
    energia = 1 - abs(nodo.memoria.cargas)
    for rel in nodo.memoria.relaciones:
        energia += abs(rel.cargaCompartida)
    return min(energia, 1)


def calcular_distancia(nodo_a: NodoInterface, nodo_b: NodoInterface) -> float:
    i_a, j_a = map(int, nodo_a.id.split('-')[1:])
    i_b, j_b = map(int, nodo_b.id.split('-')[1:])
    return ((i_a - i_b) ** 2 + (j_a - j_b) ** 2) ** 0.5


def relacionar_nodos(valores_sistema: IPhysicsRules, nodo: NodoInterface, vecinos: List[NodoInterface]):
    if nodo.memoria.energia > valores_sistema.ENERGIA:
        for vecino in vecinos:
            if (vecino and
                vecino.memoria.energia > valores_sistema.ENERGIA and
                vecino.id != nodo.id and
                    vecino.id > nodo.id):
                diferencia_cargas = abs(
                    nodo.memoria.cargas - vecino.memoria.cargas)
                distancia = calcular_distancia(nodo, vecino)
                if distancia > valores_sistema.DISTANCIA_MAXIMA_RELACION:
                    continue

                probabilidad_relacion = (
                    (diferencia_cargas / 2) *
                    (1 / distancia) *
                    valores_sistema.FACTOR_RELACION
                )

                if (random() < probabilidad_relacion and
                    ((nodo.memoria.cargas < 0 and vecino.memoria.cargas > 0) or
                     (nodo.memoria.cargas > 0 and vecino.memoria.cargas < 0))):
                    relacion_existente = next(
                        (rel for rel in nodo.memoria.relaciones if rel.nodoId == vecino.id), None)
                    if not relacion_existente:
                        carga_compartida = (
                            nodo.memoria.cargas + vecino.memoria.cargas) / 2
                        nodo.memoria.relaciones.append(
                            Relacion(vecino.id, carga_compartida))

    # Reducir gradualmente la carga compartida y eliminar relaciones con carga cero
    nodo.memoria.relaciones = [rel for rel in nodo.memoria.relaciones if (
        abs(rel.cargaCompartida) >= valores_sistema.ENERGIA and nodo.memoria.energia > valores_sistema.ENERGIA)]

    # Reducir gradualmente la carga compartida y eliminar relaciones con carga cero
    nodo.memoria.relaciones = [rel for rel in nodo.memoria.relaciones if (
        abs(rel.cargaCompartida) >= valores_sistema.ENERGIA and nodo.memoria.energia > valores_sistema.ENERGIA)]


def expandir_espacio(nodos: List[NodoInterface], valores_sistema: IPhysicsRules) -> List[NodoInterface]:
    # Añadir filas en la parte inferior
    for i in range(valores_sistema.CRECIMIENTO_X):
        for j in range(valores_sistema.COLUMNAS):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(valores_sistema.FILAS + i, j, cargas, energia)
            nodos.append(nodo)

    # Añadir columnas a la derecha
    for i in range(valores_sistema.FILAS + valores_sistema.CRECIMIENTO_X):
        for j in range(valores_sistema.CRECIMIENTO_Y):
            cargas = uniform(-1, 1)
            energia = 1 - abs(cargas)
            nodo = crear_nodo(i, valores_sistema.COLUMNAS + j, cargas, energia)
            nodos.append(nodo)

    # Actualizar los valores del sistema
    valores_sistema.FILAS += valores_sistema.CRECIMIENTO_X
    valores_sistema.COLUMNAS += valores_sistema.CRECIMIENTO_Y

    return nodos
