import json
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tkinter as tk
import networkx as nx
import os
from tkinter import ttk
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from time_procedural import calcular_relaciones_matricial, calcular_distancias_matricial
from typing import List
from types_universo import PhysicsRules, NodoInterface


def open_relaciones_window():
    new_window = tk.Toplevel(root)
    new_window.title("Seleccionar Rango de Relaciones")

    row_label = tk.Label(new_window, text="Fila Inicial:")
    row_label.pack(padx=5, pady=5)

    row_entry = tk.Entry(new_window)
    row_entry.pack(padx=5, pady=5)

    col_label = tk.Label(new_window, text="Columna Inicial:")
    col_label.pack(padx=5, pady=5)

    col_entry = tk.Entry(new_window)
    col_entry.pack(padx=5, pady=5)

    def on_submit():
        start_row = int(row_entry.get())
        start_col = int(col_entry.get())
        load_relaciones(start_row=start_row, start_col=start_col)
        new_window.destroy()

    submit_button = tk.Button(new_window, text="Enviar", command=on_submit)
    submit_button.pack(padx=5, pady=5)


def plot_matrix(matriz, title, start_row=0, start_col=0, size=1000):
    if len(matriz.shape) == 1:  # Si es unidimensional
        size = int(np.sqrt(len(matriz)))
        if size * size != len(matriz):
            print(
                "La longitud de la matriz no es un cuadrado perfecto. No se puede redimensionar.")
            return
        # Redimensionar a la forma cuadrada
        matriz = np.reshape(matriz, (size, size))

    sub_matriz = matriz[start_row:start_row + size, start_col:start_col + size]
    plt.imshow(sub_matriz, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_relaciones(matriz_relaciones, title='Matriz Relaciones', start_row=0, start_col=0, size=1000):
    G = nx.Graph()
    sub_matriz = matriz_relaciones[start_row:start_row +
                                   size, start_col:start_col + size]

    # Agregar nodos
    for i in range(size):
        G.add_node(i)

    # Agregar aristas basadas en la sub_matriz de relaciones
    for i in range(size):
        for j in range(size):
            if sub_matriz[i][j] != 0:
                G.add_edge(i, j, weight=sub_matriz[i][j])

    pos = nx.spring_layout(G)  # Posición de los nodos
    nx.draw(G, pos, with_labels=True)
    plt.title(title)
    plt.show()


def load_file(filename, progress_bar):
    chunk_size = 1024
    file_size = os.path.getsize(filename)
    progress_bar['maximum'] = file_size
    file_content = ""

    def read_chunk(file, size):
        return file.read(size)

    with open(filename, 'r') as f:
        with ThreadPoolExecutor() as executor:
            while chunk := executor.submit(read_chunk, f, chunk_size).result():
                file_content += chunk
                progress_bar['value'] += len(chunk)
                root.update_idletasks()

    data = json.loads(file_content)
    return np.array(data)


def load_and_plot_thread(filename, plot_function, title, progress_bar, start_row=0, start_col=0):
    matriz = load_file(filename, progress_bar)
    plot_function(matriz, title, start_row=start_row, start_col=start_col)
    progress_bar.pack_forget()


def load_and_plot(filename, plot_function, title, start_row=0, start_col=0):
    progress_bar = ttk.Progressbar(
        root, orient='horizontal', mode='determinate')
    progress_bar.pack(pady=5)
    thread = Thread(target=load_and_plot_thread, args=(
        filename, plot_function, title, progress_bar, start_row, start_col))
    thread.start()


def load_energias():
    load_and_plot('energiasMatriz.json', plot_matrix, 'Energias Matriz')


def load_cargas():
    load_and_plot('cargasMatriz.json', plot_matrix, 'Cargas Matriz')


def load_relaciones(start_row=0, start_col=0):
    progress_bar = ttk.Progressbar(
        root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=5)
    progress_bar.start()

    def calculate_and_plot():
        # Cargar las reglas de física desde el archivo JSON
        with open('mejor_universo.json', 'r') as file:
            physics_rules_data = json.load(file)
            # Asegúrate de adaptar esto a la estructura de tus datos
            physics_rules = PhysicsRules(**physics_rules_data)

        # Cargar cargas y energías desde sus archivos
        with open('cargasMatriz.json', 'r') as file:
            cargas = cp.array(json.load(file))
        with open('energiasMatriz.json', 'r') as file:
            energias = cp.array(json.load(file))

        matriz_distancias = calcular_distancias_matricial(
            len(cargas), len(cargas))
        matriz_relaciones = calcular_relaciones_matricial(
            physics_rules, cargas, matriz_distancias)
        plot_relaciones(matriz_relaciones, 'Matriz Relaciones',
                        start_row=start_row, start_col=start_col)
        progress_bar.pack_forget()

    thread = Thread(target=calculate_and_plot)
    thread.start()


root = tk.Tk()
root.title("Visualizador de Matrices")
root.geometry("400x300")  # Ajusta el tamaño según tus preferencias

title_label = tk.Label(
    root, text="Visualizador de Matrices Universo", font=("Helvetica", 16))
title_label.pack(pady=10)

energias_button = tk.Button(
    root, text="Mostrar Energias Matriz", command=load_energias)
energias_button.pack(pady=5)

cargas_button = tk.Button(
    root, text="Mostrar Cargas Matriz", command=load_cargas)
cargas_button.pack(pady=5)

relaciones_button = tk.Button(
    root, text="Mostrar Matriz Relaciones", command=open_relaciones_window)
relaciones_button.pack(pady=5)

root.mainloop()
