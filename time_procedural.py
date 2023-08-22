from typing import List
from types_universo import NodoInterface, IPhysicsRules, SystemRules, systemRules
import cupy as cp


def intercambiar_cargas_matricial(cargas: cp.ndarray, matriz_relaciones):
    diferencias_cargas = (cargas[:, None] - cargas) / 2
    diferencias_cargas = cp.clip(
        diferencias_cargas, -systemRules.LIMITE_INTERCAMBIO, systemRules.LIMITE_INTERCAMBIO)

    mask_intercambio = (diferencias_cargas != 0) & \
                       (cargas[:, None] * cargas < 0) & \
                       (matriz_relaciones > 0)

    diferencias_cargas[mask_intercambio == False] = 0

    cargas_nuevas = cargas - cp.sum(diferencias_cargas, axis=1)

    # Me gustaria saber el intercambio de cargas es bueno guardarlo o calcularlo
    # matriz_cargas = matriz_relaciones * diferencias_cargas

    return cargas_nuevas


def calcular_energia_matricial(energias: cp.ndarray, matriz_relaciones: cp.ndarray) -> cp.ndarray:
    carga_en_cadena_abs = cp.abs(cp.sum(matriz_relaciones, axis=1))
    energias = 1 - \
        cp.abs(energias) + carga_en_cadena_abs
    energias = cp.clip(energias, None, 1)
    liberar_memoria_gpu()
    return energias


def calcular_distancias_matricial(nodos: List[NodoInterface]) -> cp.ndarray:
    coords = cp.array([list(map(int, nodo.id.split('-')[1:]))
                      for nodo in nodos], dtype=cp.float16)
    diff = coords[:, None, :] - coords
    matriz_distancias = cp.sqrt(cp.sum(diff ** 2, axis=-1))
    liberar_memoria_gpu()
    return matriz_distancias


def relacionar_nodos_matricial(physics_rules: IPhysicsRules, energias: cp.ndarray, cargas: cp.ndarray, matriz_distancias: cp.ndarray) -> cp.ndarray:
    mask = matriz_distancias == 1
    diferencia_energias = cp.where(
        mask, cp.abs(energias[:, None] - energias), 0)
    diferencia_cargas = cp.where(mask, cp.abs(cargas[:, None] - cargas), 0)
    relacion_mask = mask & (diferencia_energias > physics_rules.ENERGIA)
    carga_compartida = cp.where(relacion_mask, diferencia_cargas, 0)
    liberar_memoria_gpu()

    return carga_compartida


def simulacion_estocastica(cargas: cp.ndarray, physics_rules: IPhysicsRules) -> cp.ndarray:
    transicion_indices = cp.random.rand(
        len(cargas)) < physics_rules.PROBABILIDAD_TRANSICION
    cargas[transicion_indices] *= -1

    fluctuacion = cp.random.uniform(-physics_rules.FLUCTUACION_MAXIMA,
                                    physics_rules.FLUCTUACION_MAXIMA, len(cargas), dtype=cp.float32)
    aleatorio_factor = cp.random.choice([1, -1], len(cargas))
    cargas += fluctuacion * aleatorio_factor

    tunel_indices = (cargas > 0.5) & (cp.random.rand(
        len(cargas)) < physics_rules.PROBABILIDAD_TUNEL)
    cargas[tunel_indices] = 0

    # cargas = cp.clip(cargas, -1, 1)
    liberar_memoria_gpu()
    return cargas


def calcular_cargas(cargas: cp.ndarray, matriz_relaciones: cp.ndarray, physics_rules: IPhysicsRules) -> cp.ndarray:
    cargas_nuevas = intercambiar_cargas_matricial(
        cargas, matriz_relaciones)
    return simulacion_estocastica(cargas_nuevas, physics_rules)


def liberar_memoria_gpu():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.MemoryPool().free_all_blocks()
