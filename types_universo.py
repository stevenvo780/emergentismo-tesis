from typing import List
import cupy as cp

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
    FACTOR_RELACION: int

class PhysicsRules(IPhysicsRules):
    def __init__(self,
                 PROBABILIDAD_VIDA_INICIAL=0.99999,
                 UMBRAL_CARGA=0.1,
                 ENERGIA=0.01,
                 PROBABILIDAD_TRANSICION=0.01,
                 FLUCTUACION_MAXIMA=0.01,
                 PROBABILIDAD_TUNEL=0.01,
                 FACTOR_ESTABILIDAD=0.2,
                 FACTOR_RELACION=10,
                 ):
        self.PROBABILIDAD_VIDA_INICIAL = PROBABILIDAD_VIDA_INICIAL
        self.UMBRAL_CARGA = UMBRAL_CARGA
        self.FACTOR_ESTABILIDAD = FACTOR_ESTABILIDAD
        self.ENERGIA = ENERGIA
        self.PROBABILIDAD_TRANSICION = PROBABILIDAD_TRANSICION
        self.FLUCTUACION_MAXIMA = FLUCTUACION_MAXIMA
        self.PROBABILIDAD_TUNEL = PROBABILIDAD_TUNEL
        self.FACTOR_RELACION = FACTOR_RELACION

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in vars(self).items()]
        return '\n'.join(attributes)


class ISystemRules:
    FILAS = int,
    COLUMNAS = int,
    INTERVALO_ENTRENAMIENTO = int
    PUNTAGE_MINIMO_REINICIO = int
    LIMITE_RELACIONAL = int
    DISTANCIA_MAXIMA_RELACION = int
    FACTOR_RELACION_LIMIT = int
    CRECIMIENTO_X = int
    CRECIMIENTO_Y = int
    NEURONAL_FACTOR = float
    NEURONAS_CANTIDAD = float
    TASA_APRENDIZAJE = float
    NUM_THREADS = int
    MEJOR_PUNTAJE = int
    LIMITE_INTERCAMBIO= float


class SystemRules(ISystemRules):
    def __init__(self,
                 GIRD_SIZE=100,
                 FILAS=100,
                 COLUMNAS=100,
                 INTERVALO_ENTRENAMIENTO=50,
                 PUNTAGE_MINIMO_REINICIO=20000,
                 LIMITE_RELACIONAL=6,
                 DISTANCIA_MAXIMA_RELACION=1,
                 RECOMPENSA_EXTRA_CERRADA=10000,
                 RECOMPENSA_POR_RELACION=0.1,
                 PENALIZACION_POR_RELACIONES=1000,
                 UMBRAL_PROPORCION=0.1,
                 UMBRAL_CONJUNTOS_CERRADOS=1,
                 NEURONAL_FACTOR_INCREASE=0.05,
                 FACTOR_RELACION_LIMIT=10,
                 CRECIMIENTO_X=2,
                 CRECIMIENTO_Y=2,
                 NEURONAL_FACTOR=0.05,
                 NEURONAS_CANTIDAD=8,
                 TASA_APRENDIZAJE=0.5,
                 NUM_THREADS=12,
                 MEJOR_PUNTAJE=0,
                 LIMITE_INTERCAMBIO=10,
                 ):
        self.GIRD_SIZE = GIRD_SIZE
        self.FILAS = FILAS
        self.COLUMNAS = COLUMNAS
        self.INTERVALO_ENTRENAMIENTO = INTERVALO_ENTRENAMIENTO
        self.PUNTAGE_MINIMO_REINICIO = PUNTAGE_MINIMO_REINICIO
        self.LIMITE_RELACIONAL = LIMITE_RELACIONAL
        self.DISTANCIA_MAXIMA_RELACION = DISTANCIA_MAXIMA_RELACION
        self.RECOMPENSA_EXTRA_CERRADA = RECOMPENSA_EXTRA_CERRADA
        self.RECOMPENSA_POR_RELACION = RECOMPENSA_POR_RELACION
        self.PENALIZACION_POR_RELACIONES = PENALIZACION_POR_RELACIONES
        self.UMBRAL_CONJUNTOS_CERRADOS = UMBRAL_CONJUNTOS_CERRADOS
        self.NEURONAL_FACTOR_INCREASE = NEURONAL_FACTOR_INCREASE
        self.FACTOR_RELACION_LIMIT = FACTOR_RELACION_LIMIT
        self.CRECIMIENTO_X = CRECIMIENTO_X
        self.CRECIMIENTO_Y = CRECIMIENTO_Y
        self.NEURONAL_FACTOR = NEURONAL_FACTOR
        self.NEURONAS_CANTIDAD = NEURONAS_CANTIDAD
        self.TASA_APRENDIZAJE = TASA_APRENDIZAJE
        self.NUM_THREADS = NUM_THREADS
        self.MEJOR_PUNTAJE = MEJOR_PUNTAJE
        self.UMBRAL_PROPORCION = UMBRAL_PROPORCION
        self.LIMITE_INTERCAMBIO = LIMITE_INTERCAMBIO

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in vars(self).items()]
        return '\n'.join(attributes)


class NodoInterface:
    def __init__(self, id: str, cargas: float, energia: float):
        self.id = id
        self.cargas = cargas
        self.energia = energia
        self.relaciones = []

class Relacion:
    def __init__(self, nodoId: str, cargaCompartida: float):
        self.nodoId = nodoId
        self.cargaCompartida = cargaCompartida


systemRules = SystemRules()
