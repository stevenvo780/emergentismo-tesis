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
                 MEJOR_RECOMPENSA=0,
                 NEURONAS_PROFUNDIDAD=60,
                 NEURONAS_DENSIDAD_ENTRADA=12,
                 INTERVALO_ENTRENAMIENTO=2000,
                 PORCENTAJE_POBLACION_MUTACION=0.2,
                 FACTOR_ENTROPIA=10,
                 VARIACION_NEURONAL_GRANDE=0.1,
                 VARIACION_NEURONAL_PEQUEÑA=0.05,
                 FACTOR_RELACION_LIMIT=10,
                 TASA_APRENDIZAJE=0.5,

                 # Configuraciones para evitar errores
                 LIMITE_INTERCAMBIO=1,
                 GENERACIONES_PARA_REINICIO=10,
                 TOLERANCIA_ENERGIA=1,
                 MEMORIA_POR_FILA=1048,
                 FILAS_POR_MB=100,
                 CONSTANTE_HUBBLE=500,
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
        self.FACTOR_ENTROPIA = FACTOR_ENTROPIA
        self.VARIACION_NEURONAL_GRANDE = VARIACION_NEURONAL_GRANDE
        self.VARIACION_NEURONAL_PEQUEÑA = VARIACION_NEURONAL_PEQUEÑA
        self.FACTOR_RELACION_LIMIT = FACTOR_RELACION_LIMIT
        self.TASA_APRENDIZAJE = TASA_APRENDIZAJE

        # Configuraciones para evitar errores
        self.LIMITE_INTERCAMBIO = LIMITE_INTERCAMBIO
        self.GENERACIONES_PARA_REINICIO = GENERACIONES_PARA_REINICIO
        self.TOLERANCIA_ENERGIA = TOLERANCIA_ENERGIA
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
