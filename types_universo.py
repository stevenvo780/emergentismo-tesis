class PhysicsRules():
    def __init__(self,
                 PROBABILIDAD_VIDA_INICIAL=0.99999,
                 UMBRAL_CARGA=0.1,
                 ENERGIA=0.00001,
                 PROBABILIDAD_TRANSICION=0.01,
                 FLUCTUACION_MAXIMA=0.01,
                 PROBABILIDAD_TUNEL=0.01,
                 FACTOR_ESTABILIDAD=0.1,
                 PROBABILIDAD_SUPERVIVENCIA=0.5,
                 ):
        self.PROBABILIDAD_VIDA_INICIAL = PROBABILIDAD_VIDA_INICIAL
        self.UMBRAL_CARGA = UMBRAL_CARGA
        self.ENERGIA = ENERGIA
        self.PROBABILIDAD_TRANSICION = PROBABILIDAD_TRANSICION
        self.FLUCTUACION_MAXIMA = FLUCTUACION_MAXIMA
        self.PROBABILIDAD_TUNEL = PROBABILIDAD_TUNEL
        self.FACTOR_ESTABILIDAD = FACTOR_ESTABILIDAD
        self.PROBABILIDAD_SUPERVIVENCIA = PROBABILIDAD_SUPERVIVENCIA

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in vars(self).items()]
        return '\n'.join(attributes)

class SystemRules:
    def __init__(self,
                 # GRID
                 GIRD_SIZE=100,
                 FILAS=100,
                 COLUMNAS=100,
                 CRECIMIENTO_X=2,
                 CRECIMIENTO_Y=2,

                 # Red evolutiva
                 MEJOR_RECOMPENSA=0,
                 NEURONAS_PROFUNDIDAD=16,
                 NEURONAS_DENSIDAD_ENTRADA=12,
                 INTERVALO_ENTRENAMIENTO=2000,
                 PORCENTAJE_POBLACION_MUTACION=0.2,
                 RECOMPENSA_EXTRA_CERRADA=0.1,
                 RECOMPENSA_POR_RELACION=0.00001,
                 PENALIZACION_RELACIONES_SINFORMA=1000,
                 UMBRAL_PROPORCION_ESTRUCUTRAS_CERRADAS=0.1,
                 VARIACION_NEURONAL_GRANDE=0.1,
                 VARIACION_NEURONAL_PEQUEÑA=0.05,
                 FACTOR_RELACION_LIMIT=10,
                 TASA_APRENDIZAJE=0.5,

                 # Configuraciones para evitar errores
                 LIMITE_INTERCAMBIO=1,
                 GENERACIONES_PARA_REINICIO=50,
                 TOLERANCIA_ENERGIA=1,
                 MEMORIA_POR_FILA=4,
                 FILAS_POR_GB=1,
                 ):
        # GRID
        self.GIRD_SIZE = GIRD_SIZE
        self.FILAS = FILAS
        self.COLUMNAS = COLUMNAS
        self.CRECIMIENTO_X = CRECIMIENTO_X
        self.CRECIMIENTO_Y = CRECIMIENTO_Y

        # Red evolutiva
        self.MEJOR_RECOMPENSA = MEJOR_RECOMPENSA
        self.NEURONAS_PROFUNDIDAD = NEURONAS_PROFUNDIDAD
        self.NEURONAS_DENSIDAD_ENTRADA = NEURONAS_DENSIDAD_ENTRADA
        self.INTERVALO_ENTRENAMIENTO = INTERVALO_ENTRENAMIENTO
        self.PORCENTAJE_POBLACION_MUTACION = PORCENTAJE_POBLACION_MUTACION
        self.RECOMPENSA_EXTRA_CERRADA = RECOMPENSA_EXTRA_CERRADA
        self.RECOMPENSA_POR_RELACION = RECOMPENSA_POR_RELACION
        self.PENALIZACION_RELACIONES_SINFORMA = PENALIZACION_RELACIONES_SINFORMA
        self.UMBRAL_PROPORCION_ESTRUCUTRAS_CERRADAS = UMBRAL_PROPORCION_ESTRUCUTRAS_CERRADAS
        self.VARIACION_NEURONAL_GRANDE = VARIACION_NEURONAL_GRANDE
        self.VARIACION_NEURONAL_PEQUEÑA = VARIACION_NEURONAL_PEQUEÑA
        self.FACTOR_RELACION_LIMIT = FACTOR_RELACION_LIMIT
        self.TASA_APRENDIZAJE = TASA_APRENDIZAJE

        # Configuraciones para evitar errores
        self.LIMITE_INTERCAMBIO = LIMITE_INTERCAMBIO
        self.GENERACIONES_PARA_REINICIO = GENERACIONES_PARA_REINICIO
        self.TOLERANCIA_ENERGIA = TOLERANCIA_ENERGIA
        self.MEMORIA_POR_FILA = MEMORIA_POR_FILA
        self.FILAS_POR_GB = FILAS_POR_GB

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
