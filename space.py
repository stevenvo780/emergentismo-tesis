from types_universo import NodoInterface, neuronalRules
from time_procedural import calcular_cargas, calcular_energia
from random import uniform
from typing import List
import cupy as cp
import subprocess


def next_step(universo):
    matriz_distancias = universo.matriz_distancias

    with cp.cuda.Stream():
        cargas_nuevas = calcular_cargas(
            universo.cargasMatriz, matriz_distancias, universo.physics_rules)

    with cp.cuda.Stream():
        energias = calcular_energia(
            universo.energiasMatriz, cargas_nuevas, universo.physics_rules)
    return cargas_nuevas, energias


def expandir_espacio(self, cargasMatriz: cp.ndarray) -> cp.ndarray:

    memoria_disponible = obtener_memoria_disponible()
    crecimiento_permitido = int(
        memoria_disponible // neuronalRules.MEMORIA_POR_FILA * neuronalRules.FILAS_POR_MB)
    crecimiento_x = min(self.procedural_rules.CRECIMIENTO_X //
                        2, crecimiento_permitido // 2)
    crecimiento_y = min(self.procedural_rules.CRECIMIENTO_Y //
                        2, crecimiento_permitido // 2)

    if crecimiento_x > 0 and self.procedural_rules.COLUMNAS > 0:
        new_rows = cp.array(
            [uniform(-1, 1) for _ in range(self.procedural_rules.COLUMNAS * crecimiento_x * 2)]).reshape(crecimiento_x * 2, -1)
        cargasMatriz = cp.vstack(
            (new_rows[:crecimiento_x, :], cargasMatriz, new_rows[crecimiento_x:, :]))

    if crecimiento_y > 0 and self.procedural_rules.FILAS + 2 * crecimiento_x > 0:
        new_columns = cp.array([uniform(-1, 1) for _ in range(
            (self.procedural_rules.FILAS + 2 * crecimiento_x) * crecimiento_y * 2)]).reshape(-1, crecimiento_y * 2)
        cargasMatriz = cp.hstack(
            (new_columns[:, :crecimiento_y], cargasMatriz, new_columns[:, crecimiento_y:]))

    self.procedural_rules.FILAS += 2 * crecimiento_x
    self.procedural_rules.COLUMNAS += 2 * crecimiento_y

    return cargasMatriz


def obtener_memoria_disponible():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                                '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)  # type: ignore
        memoria_disponible = int(result.stdout.decode('utf-8').strip())
        return memoria_disponible  # Convertir a megabytes
    except Exception as e:
        print("Error al obtener la memoria disponible de la GPU:", e)
        return 0
