from threading import Lock
import json
import os
from types_universo import neuronalRules
lock_guardar = Lock()


def save_matrices_to_json(energiasMatriz, cargasMatriz):
    with lock_guardar:
        with open('energiasMatriz.json', 'w') as file:
            json.dump(energiasMatriz.tolist(), file)

        with open('cargasMatriz.json', 'w') as file:
            json.dump(cargasMatriz.tolist(), file)


def guardar_mejor_universo(self, index_universo):
    mejores_nuevos_valores = self.universos[index_universo].physics_rules
    mejor_universo = {key: float(value) for key, value in vars(
        mejores_nuevos_valores).items()}
    with open('mejor_universo.json', 'w') as file:
        json.dump(mejor_universo, file)


def guardar_poblacion(self):
    if not os.path.exists('poblacion_guardada'):
        os.mkdir('poblacion_guardada')

    subdirectorio = 'poblacion_guardada'
    if not os.path.exists(subdirectorio):
        os.mkdir(subdirectorio)

    for i, modelo in enumerate(self.poblacion):
        modelo.save_weights(f'{subdirectorio}/red_neuronal_{i}.h5')


def cargar_poblacion(self):
    subdirectorio = 'poblacion_guardada'
    if os.path.exists(subdirectorio):
        for i in range(len(self.poblacion)):
            try:
                self.poblacion[i].load_weights(
                    f'{subdirectorio}/red_neuronal_{i}.h5')
            except:
                print(
                    f"No se pudo cargar la red neuronal {i} para la población.")
    else:
        print("No hay una población guardada para cargar.")


def cargar_mejor_puntaje(self):
    try:
        with open('system_rules.json', 'r') as file:
            rules = json.load(file)
            self.mejor_maxima_recompensa = rules.get("mejor_maxima_recompensa", 0.0)
            self.actual_total_recompensa = rules.get("actual_total_recompensa", 0)
            self.generaciones_sin_mejora = rules.get("generaciones_sin_mejora", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        self.guardar_mejor_puntaje()


def guardar_mejor_puntaje(self):
    rules = {
        "mejor_maxima_recompensa": self.mejor_maxima_recompensa,
        "actual_total_recompensa": self.actual_total_recompensa,
        "generaciones_sin_mejora": self.generaciones_sin_mejora
    }
    with open('system_rules.json', 'w') as file:
        json.dump(rules, file)
