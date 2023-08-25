from types_universo import NodoInterface, systemRules
from time_procedural import calcular_cargas, calcular_energia
from random import uniform
from typing import List
import cupy as cp
import cupy
import subprocess

pool = cp.cuda.memory.SingleDeviceMemoryPool(cp.cuda.memory._malloc)
cp.cuda.set_allocator(pool.malloc)

mempool = cupy.get_default_memory_pool()

# Establecer el límite de memoria fuera del bloque with
mempool.set_limit(size=16*1024**3)

with cupy.cuda.Device(0):
    # No es necesario establecer el límite de memoria aquí

    def next_step(universo):
        matriz_distancias = universo.matriz_distancias

        with cp.cuda.Stream():
            cargas_nuevas = calcular_cargas(
                universo.cargasMatriz, matriz_distancias, universo)

        with cp.cuda.Stream():
            energias = calcular_energia(
                universo.energiasMatriz, cargas_nuevas, universo.physics_rules)
        mempool.free_all_blocks()
        return cargas_nuevas, energias

    def crear_nodo(i: int, j: int, cargas: float, energia: float) -> NodoInterface:
        return NodoInterface(
            id=f"nodo-{i}-{j}",
            cargas=cargas,
            energia=energia,
        )

    def obtener_memoria_disponible():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
            memoria_disponible = int(result.stdout.decode('utf-8').strip())
            return memoria_disponible  # Convertir a bytes
        except Exception as e:
            print("Error al obtener la memoria disponible de la GPU:", e)
            return 0

    def expandir_espacio(nodos: List[NodoInterface]) -> List[NodoInterface]:
        memoria_disponible = obtener_memoria_disponible()
        crecimiento_permitido = int(
            memoria_disponible // systemRules.MEMORIA_POR_FILA * systemRules.FILAS_POR_MB)
        crecimiento_x = min(systemRules.CRECIMIENTO_X, crecimiento_permitido)
        crecimiento_y = min(systemRules.CRECIMIENTO_Y, crecimiento_permitido)

        for i in range(crecimiento_x):
            for j in range(systemRules.COLUMNAS):
                cargas = uniform(-1, 1)
                energia = 1 - abs(cargas)
                nodo = crear_nodo(systemRules.FILAS + i, j, cargas, energia)
                nodos.append(nodo)

        for i in range(systemRules.FILAS + crecimiento_x):
            for j in range(crecimiento_y):
                cargas = uniform(-1, 1)
                energia = 1 - abs(cargas)
                nodo = crear_nodo(i, systemRules.COLUMNAS +
                                  j, cargas, energia)
                nodos.append(nodo)

        systemRules.FILAS += crecimiento_x
        systemRules.COLUMNAS += crecimiento_y

        return nodos
