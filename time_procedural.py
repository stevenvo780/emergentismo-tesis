from typing import List
from types_universo import PhysicsRules, systemRules, NodoInterface
import cupy as cp
import cupy

# Configurar el límite de la memoria del dispositivo (por ejemplo, 4 GiB)
device_limit = 16 * 1024**3

# Configurar el gestor de memoria de CuPy
mempool = cp.get_default_memory_pool()

# Elegir el dispositivo con el que trabajar (por ejemplo, el 0)
with cp.cuda.Device(0):
    mempool.set_limit(size=device_limit)
    cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)
    def calcular_cargas(cargas: cp.ndarray, matriz_distancias: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        interacciones_distancia = calcular_interacciones_distancia(
            cargas, matriz_distancias, physics_rules)
        vecinos_vivos = cp.sum(matriz_distancias == 1, axis=1)

        nacimiento = (cargas == 0) & (cp.random.rand(len(cargas)) <
                                      physics_rules.PROBABILIDAD_VIDA_INICIAL * vecinos_vivos)
        supervivencia = (cargas != 0) & (cp.random.rand(len(cargas)) < 1 / (1 +
                                                                            cp.exp(-physics_rules.FACTOR_ESTABILIDAD * (vecinos_vivos - physics_rules.CONSTANTE_HUBBLE))))
        muerte = ~nacimiento & ~supervivencia

        cargas_nuevas = cargas.copy()
        cargas_nuevas[nacimiento] = cp.random.uniform(
            -1, 1, cp.sum(nacimiento).item())
        cargas_nuevas[muerte] = 0
        fluctuacion = cp.sin(cargas_nuevas) * cp.random.uniform(-physics_rules.FLUCTUACION_MAXIMA,
                                                                physics_rules.FLUCTUACION_MAXIMA, len(cargas_nuevas), dtype=cp.float32)

        # Añade la interacción a distancia
        cargas_nuevas += interacciones_distancia

        cargas_nuevas += fluctuacion
        return cp.clip(cargas_nuevas, -systemRules.LIMITE_INTERCAMBIO, systemRules.LIMITE_INTERCAMBIO)

    def calcular_interacciones_distancia(cargas: cp.ndarray, matriz_distancias: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        decaimiento = cp.exp(-matriz_distancias /
                             physics_rules.LONGITUD_DE_DECAY)
        ruido_distancia = cp.random.uniform(-physics_rules.RUIDO_MAXIMO,
                                            physics_rules.RUIDO_MAXIMO, cargas.shape)
        # Asegúrate de que la dimensión sea correcta
        interacciones = cp.sum(decaimiento * ruido_distancia, axis=1)

        return interacciones

    def calcular_energia(energias: cp.ndarray, cargas: cp.ndarray, physics_rules: PhysicsRules) -> cp.ndarray:
        # Usar una función escalón para la energía
        return cp.where(cp.abs(energias + cargas) > physics_rules.UMBRAL_CARGA, 1, 0)

    def calcular_relaciones_matricial(physics_rules: PhysicsRules, cargas: cp.ndarray, matriz_distancias: cp.ndarray) -> cp.ndarray:
        mask = matriz_distancias == 1
        diferencia_cargas = cp.where(mask, cp.abs(cargas[:, None] - cargas), 0)
        relacion_mask = mask & (diferencia_cargas > physics_rules.ENERGIA)
        return cp.where(relacion_mask, cp.exp(diferencia_cargas), 0)

    def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
        coords = cp.array([nodo.id.split('-')[1:]
                           for nodo in nodos], dtype=cp.int16)
        diff = coords[:, None, :] - coords
        return cp.sqrt(cp.sum(diff ** 2, axis=-1))

    def calcular_entropia_condicional(cargas: cp.ndarray, energias: cp.ndarray, matriz_distancias: cp.ndarray) -> float:
        n = len(cargas)

        # Obtener los índices de los vecinos
        vecinos_indices = cp.where(matriz_distancias == 1)
        cargas_vecinos = cargas[vecinos_indices[1]]
        energias_vecinos = energias[vecinos_indices[1]]

        # Define el número de bins (intervalos) para el histograma
        num_bins = 10  # Ajusta esto según tus datos y preferencias

        # Calcular las frecuencias conjuntas de cargas y energías de los vecinos
        p_x_y, _, _ = cp.histogram2d(
            cargas_vecinos, energias_vecinos, bins=num_bins)
        p_x_y /= n
        p_x = cp.sum(p_x_y, axis=1)
        p_y = cp.sum(p_x_y, axis=0)

        # Evitar la división por cero y los logaritmos de cero
        p_x_y_safe = cp.where(p_x_y > 0, p_x_y, 1)
        p_y_safe = cp.where(p_y > 0, p_y, 1)

        entropia_condicional = -cp.sum(p_x_y * cp.log2(p_x_y_safe / p_y_safe))

        return entropia_condicional.item()
