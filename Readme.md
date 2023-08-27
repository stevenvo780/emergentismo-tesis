# Algoritmo de Entrenamiento del Universo

## Descripción General

Este algoritmo simula el comportamiento de un "universo" con reglas físicas específicas. Utiliza algoritmos genéticos y redes neuronales para optimizar ciertas variables del universo con el objetivo de maximizar una función de recompensa.

## Dependencias

- numpy
- universo (módulo personalizado)
- types_universo (módulo personalizado)
- time_procedural (módulo personalizado)
- random
- threading
- keras
- json
- os

## Clases Principales

### Entrenador

Esta es la clase principal que maneja la simulación y entrenamiento del universo.

#### Métodos Importantes

- `big_bang()`: Inicia la simulación del universo y aplica el algoritmo genético.
- `entrenar()`: Entrena las redes neuronales en función de la recompensa generada por el universo.
- `mutate()`: Aplica mutaciones a una red neuronal.
- `evolve_population()`: Evoluciona la población de redes neuronales.
- `crossover()`: Cruza dos redes neuronales para generar hijos.
- `predecir_valores()`: Utiliza la red neuronal para predecir nuevos valores para las variables del universo.
- `aplicar_mutaciones()`: Aplica mutaciones a la nueva población.
- `mantener_elite()`: Mantiene los mejores individuos de la población.
- `crear_red_neuronal()`: Crea una nueva red neuronal con una arquitectura específica.
- `crear_nueva_poblacion()`: Crea una nueva población de redes neuronales.
- `reiniciar_poblacion()`: Reinicia la población si no hay mejoras.
- `reiniciarUniverso()`: Reinicia el universo con nuevos valores.
- `cargar_mejor_puntaje()`, `guardar_mejor_puntaje()`, `cargar_poblacion()`, `guardar_poblacion()`, `cargar_mejor_universo()`, `guardar_mejor_universo()`: Métodos para guardar y cargar el estado.

#### Variables Importantes

- `mejor_maxima_recompensa`: Mejor recompensa obtenida hasta ahora.
- `actual_total_recompensa`: Recompensa total de la generación actual.
- `generaciones_sin_mejora`: Contador de generaciones sin mejora.
- `poblaciones`: Diccionario que contiene la población de redes neuronales para cada variable.
- `recompensas_por_clave`: Diccionario que contiene las recompensas obtenidas por cada red neuronal.

## Funciones Auxiliares

- `save_matrices_to_json()`: Guarda matrices en un archivo JSON.

## Variables Globales

- `lock_guardar`: Bloqueo para guardar el estado.

# Simulación del Universo en Python con Cupy

## Descripción

Este proyecto simula un universo utilizando `Cupy` para acelerar los cálculos matriciales en la GPU.
El universo consta de una matriz de `cargas` y otra de `energías`, las cuales evolucionan a lo largo del tiempo.

## Clases Principales

### Universo

Esta clase representa el universo en sí.

- `physics_rules`: Reglas físicas que definen el comportamiento del universo.
- `id`: ID único para cada instancia del universo.
- `tiempo`: Contador de tiempo para la simulación.
- `cargasMatriz`: Matriz que almacena las cargas de cada punto en el universo.
- `energiasMatriz`: Matriz que almacena las energías de cada punto.
- `matriz_distancias`: Matriz de distancias entre los puntos.

### Métodos Importantes en Universo

- `determinacionesDelSistema()`: Inicializa las matrices de carga y energía.
- `obtener_relaciones()`: Calcula las relaciones entre los puntos en función de sus cargas y energías.
- `next()`: Actualiza el universo para el siguiente paso de tiempo.

## Funciones Utilitarias

### next_step

Avanza el estado del universo al siguiente paso de tiempo.

### crear_nodo

Crea un nuevo nodo en el universo con cargas y energías dadas.

### obtener_memoria_disponible

Obtiene la memoria disponible en la GPU.

### expandir_espacio

Expande el espacio del universo.

## Cálculos en GPU

Se utilizan múltiples técnicas para acelerar los cálculos, incluyendo el uso de `Cupy` para realizar operaciones matriciales en la GPU.

## Requisitos

- Python 3.x
- Cupy

## Instalación

Instalar las dependencias usando pip:
\```
pip install cupy
\```

## Ejecución

Ejecutar el archivo principal para iniciar la simulación.
