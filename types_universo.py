from typing import List, Callable


class IPhysicsRules:
    FILAS: int
    COLUMNAS: int
    PROBABILIDAD_VIDA_INICIAL: float
    LIMITE_EDAD: int
    REDUCCION_CARGA: float
    CRECIMIENTO_X: int
    CRECIMIENTO_Y: int
    UMBRAL_CARGA: float
    FACTOR_ESTABILIDAD: float
    LIMITE_RELACIONAL: int
    DISTANCIA_MAXIMA_RELACION: int
    ESPERADO_EMERGENTE: int
    FACTOR_RELACION: int
    ENERGIA: float
    PROBABILIDAD_TRANSICION: float
    FLUCTUACION_MAXIMA: float
    PROBABILIDAD_TUNEL: float


class PhysicsRules(IPhysicsRules):
    def __init__(self,
                 FILAS=100,
                 COLUMNAS=100,
                 PROBABILIDAD_VIDA_INICIAL=0.99999,
                 LIMITE_EDAD=5,
                 REDUCCION_CARGA=0.01,
                 CRECIMIENTO_X=2,
                 CRECIMIENTO_Y=2,
                 UMBRAL_CARGA=0.0001,
                 FACTOR_ESTABILIDAD=0.2,
                 LIMITE_RELACIONAL=3,
                 DISTANCIA_MAXIMA_RELACION=6,
                 ESPERADO_EMERGENTE=7,
                 FACTOR_RELACION=10,
                 ENERGIA=0.01,
                 PROBABILIDAD_TRANSICION=0.01,
                 FLUCTUACION_MAXIMA=0.01,
                 PROBABILIDAD_TUNEL=0.01):
        self.FILAS = FILAS
        self.COLUMNAS = COLUMNAS
        self.PROBABILIDAD_VIDA_INICIAL = PROBABILIDAD_VIDA_INICIAL
        self.LIMITE_EDAD = LIMITE_EDAD
        self.REDUCCION_CARGA = REDUCCION_CARGA
        self.CRECIMIENTO_X = CRECIMIENTO_X
        self.CRECIMIENTO_Y = CRECIMIENTO_Y
        self.UMBRAL_CARGA = UMBRAL_CARGA
        self.FACTOR_ESTABILIDAD = FACTOR_ESTABILIDAD
        self.LIMITE_RELACIONAL = LIMITE_RELACIONAL
        self.DISTANCIA_MAXIMA_RELACION = DISTANCIA_MAXIMA_RELACION
        self.ESPERADO_EMERGENTE = ESPERADO_EMERGENTE
        self.FACTOR_RELACION = FACTOR_RELACION
        self.ENERGIA = ENERGIA
        self.PROBABILIDAD_TRANSICION = PROBABILIDAD_TRANSICION
        self.FLUCTUACION_MAXIMA = FLUCTUACION_MAXIMA
        self.PROBABILIDAD_TUNEL = PROBABILIDAD_TUNEL

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in vars(self).items()]
        return '\n'.join(attributes)


class SystemRules:
    TIEMPO_LIMITE_ESTRUCTURA = 50
    OBSERVACION_RELACIONES = 1


class NodoInterface:
    def __init__(self, id: str, memoria: 'Memoria'):
        self.id = id
        self.memoria = memoria


class Memoria:
    def __init__(self, cargas: float, energia: float, edad: int, procesos: 'Procesos', relaciones: List['Relacion']):
        self.cargas = cargas
        self.energia = energia
        self.edad = edad
        self.procesos = procesos
        self.relaciones = relaciones


class Procesos:
    def __init__(self, relacionarNodos: Callable[['IPhysicsRules', 'NodoInterface', List['NodoInterface']], None],
                 intercambiarCargas: Callable[['IPhysicsRules', 'NodoInterface', 'NodoInterface', bool], None]):
        self.relacionarNodos = relacionarNodos
        self.intercambiarCargas = intercambiarCargas


class Relacion:
    def __init__(self, nodoId: str, cargaCompartida: float):
        self.nodoId = nodoId
        self.cargaCompartida = cargaCompartida
