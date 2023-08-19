from universo import Universo
from types_universo import PhysicsRules, SystemRules
import random
from threading import Thread


class Entrenador:
    def __init__(self):
        self.universo = Universo()
        self.tiempoLimiteSinEstructuras = SystemRules.TIEMPO_LIMITE_ESTRUCTURA
        self.pesos = {key: getattr(PhysicsRules, key) for key in dir(PhysicsRules) if not key.startswith(
            "__") and not isinstance(getattr(PhysicsRules, key), (int, float))}
        self.tasaDeAprendizaje = 0.05
        self.tiempoSinEstructuras = 0

    def iniciarEntrenamiento(self):
        self.entrenamiento_thread = Thread(target=self.entrenamientoPerpetuo)
        self.entrenamiento_thread.start()

    def entrenamientoPerpetuo(self):
        while True:  # Puedes agregar una condición de salida si lo deseas
            self.nextStepRecursivo()

    def nextStepRecursivo(self):
        self.universo.next()
        self.universo.tiempo += 1
        visualizar = self.universo.tiempo % self.tiempoLimiteSinEstructuras
        if visualizar == 0:
            self.entrenarPerpetuo()
        self.nextStepRecursivo()

    def calcularRecompensa(self, nodos):
        return self.detectarEstructuras(nodos)

    def actualizarPesos(self, recompensa):
        claves = [clave for clave in self.pesos]
        for clave in claves:
            if clave in ['COLUMNAS', 'FILAS', 'CRECIMIENTO_X', 'CRECIMIENTO_Y']:
                return  # No modificar estos valores
            ajuste = random.random() * 0.1 - 0.05
            if clave in ['LIMITE_RELACIONAL', 'DISTANCIA_MAXIMA_RELACION', 'ESPERADO_EMERGENTE', 'FACTOR_RELACION']:
                ajuste = round(ajuste)
            cambio = self.tasaDeAprendizaje * recompensa * ajuste
            self.pesos[clave] += cambio

    def actualizarConfiguracion(self, tiempoLimiteSinEstructuras, tasaDeAprendizaje):
        self.tiempoLimiteSinEstructuras = tiempoLimiteSinEstructuras
        self.tasaDeAprendizaje = tasaDeAprendizaje

    def detectarEstructuras(self, nodos):
        numeroDeEstructuras = 0
        nodosVisitados = set()
        for nodo in nodos:
            if nodo.id in nodosVisitados:
                continue
            nodosRelacionados = [rel.nodoId for rel in nodo.memoria.relaciones]
            if len(nodosRelacionados) >= self.universo.valoresSistema.ESPERADO_EMERGENTE:
                esEstructuraValida = all([any(nodoRelacionado.id == idRelacionado and nodoRelacionado.memoria.energia >
                                         self.universo.valoresSistema.ENERGIA for nodoRelacionado in nodos) for idRelacionado in nodosRelacionados])
                if esEstructuraValida:
                    for idRelacionado in nodosRelacionados:
                        nodosVisitados.add(idRelacionado)
                    numeroDeEstructuras += 1
        return numeroDeEstructuras

    def reiniciarUniverso(self):
        valoresSistema = PhysicsRules()
        for clave, valor in self.pesos.items():
            setattr(valoresSistema, clave, valor)
        self.universo = Universo(valoresSistema)
        self.universo.valoresSistema.COLUMNAS = PhysicsRules.COLUMNAS
        self.universo.valoresSistema.FILAS = PhysicsRules.FILAS

    def entrenarPerpetuo(self):
        if self.hayEstructuras(self.universo.nodos):
            self.tiempoSinEstructuras = 0
        else:
            self.tiempoSinEstructuras += 1
            if self.tiempoSinEstructuras >= self.tiempoLimiteSinEstructuras:
                self.reiniciarUniverso()
                self.tiempoSinEstructuras = 0
        recompensa = self.calcularRecompensa(self.universo.nodos)
        self.actualizarPesos(recompensa)

    def hayEstructuras(self, nodos):
        return self.detectarEstructuras(nodos) > 0
