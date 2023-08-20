from typing import List, Callable


class IPhysicsRules:
    FILAS: int
    COLUMNAS: int
    PROBABILIDAD_VIDA_INICIAL: float
    REDUCCION_CARGA: float
    UMBRAL_CARGA: float
    FACTOR_ESTABILIDAD: float
    ENERGIA: float
    PROBABILIDAD_TRANSICION: float
    FLUCTUACION_MAXIMA: float
    PROBABILIDAD_TUNEL: float


class PhysicsRules(IPhysicsRules):
    def __init__(self,
                 FILAS=100,
                 COLUMNAS=100,
                 PROBABILIDAD_VIDA_INICIAL=0.99999,
                 UMBRAL_CARGA=0.1,
                 ENERGIA=0.01,
                 PROBABILIDAD_TRANSICION=0.01,
                 FLUCTUACION_MAXIMA=0.01,
                 PROBABILIDAD_TUNEL=0.01,
                 FACTOR_ESTABILIDAD=0.2,
                 ):
        self.FILAS = FILAS
        self.COLUMNAS = COLUMNAS
        self.PROBABILIDAD_VIDA_INICIAL = PROBABILIDAD_VIDA_INICIAL
        self.UMBRAL_CARGA = UMBRAL_CARGA
        self.FACTOR_ESTABILIDAD = FACTOR_ESTABILIDAD
        self.ENERGIA = ENERGIA
        self.PROBABILIDAD_TRANSICION = PROBABILIDAD_TRANSICION
        self.FLUCTUACION_MAXIMA = FLUCTUACION_MAXIMA
        self.PROBABILIDAD_TUNEL = PROBABILIDAD_TUNEL

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in vars(self).items()]
        return '\n'.join(attributes)


class SystemRules:
    TIEMPO_LIMITE_ESTRUCTURA = 50
    PUNTAGE_MINIMO_REINICIO = 1000
    LIMITE_RELACIONAL=3
    DISTANCIA_MAXIMA_RELACION=6
    ESPERADO_EMERGENTE=7
    FACTOR_RELACION=10
    CRECIMIENTO_X=2
    CRECIMIENTO_Y=2
    NEURONAL_FACTOR=0.05
    NEURONAS_CANTIDAD=10
    TASA_APRENDIZAJE=0.5
    FILAS=100
    COLUMNAS=100


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
