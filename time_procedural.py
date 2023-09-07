from typing import List
from types_universo import PhysicsRules, neuronalRules, NodoInterface
import cupy as cp
import cupy

device_limit = 32 * 1024**3
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=device_limit)
cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)

with cp.cuda.Device(0):
    def calcular_cargas(cargas: cp.ndarray, matriz_distancias: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        interacciones_distancia = calcular_interacciones_distancia(
            cargas, matriz_distancias, physics_rules)
        vecinos_vivos = cp.sum(matriz_distancias == 1,
                               axis=1).reshape(cargas.shape)

        nacimiento = (cargas == 0) & (cp.random.rand(*cargas.shape)
                                      < physics_rules.PROBABILIDAD_VIDA_INICIAL * vecinos_vivos)
        supervivencia = (cargas != 0) & (cp.random.rand(*cargas.shape) < 1 / (
            1 + cp.exp(-physics_rules.FACTOR_ESTABILIDAD * (vecinos_vivos - (physics_rules.CONSTANTE_INTERACCION_VECINOS * 10)))))
        muerte = ~nacimiento & ~supervivencia

        cargas[nacimiento] = cp.random.uniform(-1,
                                               1, cp.sum(nacimiento).item())
        cargas[muerte] = 0
        fluctuacion = cp.sin(cargas) * cp.random.uniform(-physics_rules.FLUCTUACION_MAXIMA,
                                                         physics_rules.FLUCTUACION_MAXIMA, cargas.shape)

        cargas += interacciones_distancia + fluctuacion
        return cp.clip(cargas, -neuronalRules.LIMITE_INTERCAMBIO, neuronalRules.LIMITE_INTERCAMBIO)

    def calcular_interacciones_distancia(cargas: cp.ndarray, matriz_distancias: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        LONGITUD_DE_DECAY_ESCALADA = 1 + 9 * physics_rules.LONGITUD_DE_DECAY
        decaimiento = cp.exp(-matriz_distancias / LONGITUD_DE_DECAY_ESCALADA)
        ruido_distancia = cp.random.uniform(
            -physics_rules.RUIDO_MAXIMO, physics_rules.RUIDO_MAXIMO, matriz_distancias.shape)
        return cp.sum(decaimiento * ruido_distancia, axis=1).reshape(cargas.shape)

    def calcular_energia(energias: cp.ndarray, cargas: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        return cp.where(cp.abs(energias + cargas) > physics_rules.UMBRAL_CARGA, 1, 0)

    def calcular_distancias_matricial(filas, columnas):
        x, y = cp.meshgrid(cp.arange(filas), cp.arange(columnas))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        return cp.sqrt(dx ** 2 + dy ** 2)

    def calcular_relaciones_matricial(physics_rules: PhysicsRules, cargas: cp.ndarray, matriz_distancias: cp.ndarray) -> cp.ndarray:
        mask = matriz_distancias == 1
        diferencia_cargas = cargas[:, cp.newaxis] - cargas
        relacion_mask = mask & (cp.abs(diferencia_cargas)
                                > physics_rules.ENERGIA)

        # Calcula el intercambio de cargas entre nodos vecinos
        intercambio = physics_rules.FACTOR_ESTABILIDAD * diferencia_cargas
        intercambio = cp.where(relacion_mask, intercambio, 0)

        # Aplicar el intercambio de cargas a cada nodo
        intercambio_nodos = cp.sum(intercambio, axis=1)

        return intercambio_nodos

    def calcular_entropia_condicional(cargas: cp.ndarray, energias: cp.ndarray, matriz_distancias: cp.ndarray) -> float:
        vecinos_indices = cp.where(matriz_distancias == 1)
        cargas_vecinos = cargas[vecinos_indices[0]]
        energias_vecinos = energias[vecinos_indices[0]]

        # Asegurar que los histogramas sumen 1 (probabilidades)
        n = len(cargas_vecinos)
        num_bins = 10

        p_x_y, _, _ = cp.histogram2d(
            cargas_vecinos.ravel(), energias_vecinos.ravel(), bins=num_bins)
        p_x_y /= p_x_y.sum()

        p_x = cp.sum(p_x_y, axis=1)
        p_y = cp.sum(p_x_y, axis=0)

        # Evitar la división por cero y los logaritmos de cero
        # un pequeño valor para evitar log(0) y divisiones por cero
        epsilon = 1e-10
        p_x_y_safe = cp.where(p_x_y > epsilon, p_x_y, epsilon)
        p_y_safe = cp.where(p_y > epsilon, p_y, epsilon)

        entropia_condicional = -cp.sum(p_x_y * cp.log2(p_x_y_safe / p_y_safe))

        return float(entropia_condicional)
