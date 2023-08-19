import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from entrenador import Entrenador
from types_universo import SystemRules


class App:
    def __init__(self, root):
        self.root = root
        self.entrenador = Entrenador()
        self.entrenador.iniciarEntrenamiento()
        self.root.title("Simulador del Universo")

        # Crear controles
        self.ver_grafico_button = tk.Button(
            self.root, text="Ver Gráfico", command=self.ver_grafico)
        self.ver_grafico_button.pack()

        self.ver_grid_button = tk.Button(
            self.root, text="Ver Grid", command=self.ver_grid)
        self.ver_grid_button.pack()

        self.root.after(500, self.update_grid)

    def ver_grafico(self):
        # Lógica para visualizar el gráfico
        pass

    def ver_grid(self):
        gridSize = int(len(self.entrenador.universo.nodos) ** 0.5)
        self.image_data = np.zeros((gridSize, gridSize, 3), dtype=np.uint8)
        self.fig, self.ax = plt.subplots(figsize=(gridSize / 10, gridSize / 10))
        self.im = self.ax.imshow(self.image_data, interpolation='none')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def update_grid(self):
        if not hasattr(self, 'image_data'):
            self.root.after(500, self.update_grid)
            return

        gridSize = int(len(self.entrenador.universo.nodos) ** 0.5)
        for index, nodo in enumerate(self.entrenador.universo.nodos):
            x = index % gridSize
            y = index // gridSize

            if nodo.memoria.energia > self.entrenador.universo.valoresSistema.ENERGIA and len(nodo.memoria.relaciones) > SystemRules.OBSERVACION_RELACIONES:
                color = [255, 255, 0]
            else:
                if nodo.memoria.cargas > 0:
                    blueComponent = max(
                        0, min(255, int(255 * nodo.memoria.cargas)))
                    color = [0, 200, blueComponent]
                else:
                    greyComponent = max(
                        0, min(255, 200 - int(255 * abs(nodo.memoria.cargas))))
                    color = [greyComponent, greyComponent, greyComponent]

            self.image_data[y, x, :] = color

        self.im.set_array(self.image_data)
        self.canvas.draw_idle()

        self.root.after(500, self.update_grid)


root = tk.Tk()
app = App(root)
root.mainloop()
