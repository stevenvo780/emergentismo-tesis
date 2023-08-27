class PhysicsRules():
    def __init__(self,
                 PROBABILIDAD_VIDA_INICIAL=0.99999,
                 UMBRAL_CARGA=0.1,
                 ENERGIA=0.00001,
                 FLUCTUACION_MAXIMA=0.01,
                 FACTOR_ESTABILIDAD=0.1,
                 CONSTANTE_HUBBLE=0.5,
                 LONGITUD_DE_DECAY=0.5,
                 RUIDO_MAXIMO=0.1,
                 ):
        self.PROBABILIDAD_VIDA_INICIAL = PROBABILIDAD_VIDA_INICIAL
        self.UMBRAL_CARGA = UMBRAL_CARGA
        self.ENERGIA = ENERGIA
        self.FLUCTUACION_MAXIMA = FLUCTUACION_MAXIMA
        self.FACTOR_ESTABILIDAD = FACTOR_ESTABILIDAD
        self.CONSTANTE_HUBBLE = CONSTANTE_HUBBLE
        self.LONGITUD_DE_DECAY = LONGITUD_DE_DECAY
        self.RUIDO_MAXIMO = RUIDO_MAXIMO

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
                 MEJOR_TOTAL_RECOMPENSA=0,
                 # SIEMPRE DEBE SER PAR
                 POPULATION_SIZE=6,
                 NEURONAS_PROFUNDIDAD=24,
                 NEURONAS_DENSIDAD_ENTRADA=12,
                 INTERVALO_ENTRENAMIENTO=1000,
                 FACTOR_ENTROPIA=10,
                 VARIACION_NEURONAL_GRANDE=0.001,
                 VARIACION_NEURONAL_PEQUEÑA=0.0005,
                 TASA_APRENDIZAJE=0.7,
                 GENERACIONES_PARA_AUMENTO_MUTACION=10,
                 GENERACIONES_PARA_REINICIO=100,
                 GENERACIONES_PARA_TERMINAR=1000,

                 # Configuraciones para evitar errores
                 LIMITE_INTERCAMBIO=1,
                 MEMORIA_POR_FILA=1048,
                 FILAS_POR_MB=200,
                 CONSTANTE_HUBBLE=500,
                 ):
        # GRID
        self.GIRD_SIZE = GIRD_SIZE
        self.FILAS = FILAS
        self.COLUMNAS = COLUMNAS
        self.CRECIMIENTO_X = CRECIMIENTO_X
        self.CRECIMIENTO_Y = CRECIMIENTO_Y

        # Red evolutiva
        self.MEJOR_TOTAL_RECOMPENSA = MEJOR_TOTAL_RECOMPENSA
        self.POPULATION_SIZE = POPULATION_SIZE
        self.NEURONAS_PROFUNDIDAD = NEURONAS_PROFUNDIDAD
        self.NEURONAS_DENSIDAD_ENTRADA = NEURONAS_DENSIDAD_ENTRADA
        self.INTERVALO_ENTRENAMIENTO = INTERVALO_ENTRENAMIENTO
        self.FACTOR_ENTROPIA = FACTOR_ENTROPIA
        self.VARIACION_NEURONAL_GRANDE = VARIACION_NEURONAL_GRANDE
        self.VARIACION_NEURONAL_PEQUEÑA = VARIACION_NEURONAL_PEQUEÑA
        self.TASA_APRENDIZAJE = TASA_APRENDIZAJE
        self.GENERACIONES_PARA_AUMENTO_MUTACION = GENERACIONES_PARA_AUMENTO_MUTACION
        self.GENERACIONES_PARA_TERMINAR = GENERACIONES_PARA_TERMINAR

        # Configuraciones para evitar errores
        self.LIMITE_INTERCAMBIO = LIMITE_INTERCAMBIO
        self.GENERACIONES_PARA_REINICIO = GENERACIONES_PARA_REINICIO
        self.MEMORIA_POR_FILA = MEMORIA_POR_FILA
        self.FILAS_POR_MB = FILAS_POR_MB
        self.CONSTANTE_HUBBLE = CONSTANTE_HUBBLE

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
