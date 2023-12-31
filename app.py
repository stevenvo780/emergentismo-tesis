import time
import pygame
from entrenador import Entrenador
from types_universo import neuronalRules
import cupy as cp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
pygame.init()


class ConfigWindow:
    def __init__(self, entrenador, screen):
        self.entrenador = entrenador
        self.screen = screen
        self.button_color = (50, 200, 50)
        self.text_color = (255, 255, 255)
        self.button_font = pygame.font.Font(None, 26)
        self.button_rect = pygame.Rect(340, 130, 60, 30)
        self.exit_button_rect = pygame.Rect(340, 190, 60, 30)
        self.exit_button_label = self.button_font.render(
            "Salir", True, self.text_color)
        self.paused = False
        self.update_button()

    def update_button(self):
        self.button_label = self.button_font.render(
            "Pausar" if not self.paused else "Reanudar", True, self.text_color)

    def update_screen(self, screen):
        self.screen = screen

    def run(self):
        self.update_configurations()

    def refresh(self, entrenador: Entrenador):
        see_universo = {}
        try:
            self.entrenador.universos[0]
            universo_max = max(self.entrenador.universos, key=lambda u: u.tiempo)
            see_universo = universo_max
        except IndexError:
            return
        self.entrenador = entrenador
        self.screen.fill((255, 255, 255))  # Fondo blanco
        procedural_rules = vars(
            see_universo.procedural_rules).items()
        physics_rules = vars(
            see_universo.physics_rules).items()
        system_rules = vars(neuronalRules).items()
        self.font = pygame.font.Font(None, 20)
        time_label = self.font.render(
            f"Tiempo: {see_universo.tiempo}", True, (0, 0, 0))

        self.screen.blit(time_label, (10, 10))
        claves_mostrar = [
            'recompensa_actual_generacion',
            'actual_total_recompensa',
            'mejor_maxima_recompensa',
            'generaciones_sin_mejora',
        ]

        for i, clave in enumerate(claves_mostrar):
            # 'N/A' en caso de que la clave no exista
            valor = getattr(self.entrenador, clave, 'N/A')
            label = self.font.render(f"{clave}: {valor}", True, (0, 0, 0))
            self.screen.blit(label, (10, 25 + i * 20))
        self.font = pygame.font.Font(None, 18)
        for i, (attribute, value) in enumerate(system_rules):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (6, 120 + i * 18))

        for i, (attribute, value) in enumerate(procedural_rules):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (6, (410 + i * 18)))

        for i, (attribute, value) in enumerate(physics_rules):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (6, (570 + i * 18)))

        # Dibujar el botón de pausa/reanudación (ya existente)
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        self.screen.blit(self.button_label,
                         (self.button_rect.x + 10, self.button_rect.y + 10))

        # Dibujar el nuevo botón "Salir"
        pygame.draw.rect(self.screen, self.button_color, self.exit_button_rect)
        self.screen.blit(self.exit_button_label,
                         (self.exit_button_rect.x + 10, self.exit_button_rect.y + 10))

    def update_configurations(self):
        see_universo = {}
        try:
            self.entrenador.universos[0]
            universo_max = max(self.entrenador.universos, key=lambda u: u.tiempo)
            see_universo = universo_max
        except IndexError:
            return
        for i, (attribute, value) in enumerate(vars(see_universo.physics_rules).items()):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(
                label, (self.screen.get_width() // 2 + 10, 10 + i * 20))


class App:
    def __init__(self, entrenador: Entrenador):
        self.entrenador = entrenador
        self.cellSize = 10
        self.view_offset = [0, 0]
        self.screenSize = [1600, 820]
        self.screen = pygame.display.set_mode(
            self.screenSize, pygame.RESIZABLE)
        pygame.display.set_caption("Simulador del Universo")
        self.keys_pressed = {
            pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False}
        self.zoom_level = 1
        universe_width = int(self.screenSize[0] * 0.8)
        config_width = self.screenSize[0] - universe_width
        self.universe_screen = pygame.Surface(
            (universe_width, self.screenSize[1]))
        self.config_screen = pygame.Surface((config_width, self.screenSize[1]))
        self.config_window = ConfigWindow(self.entrenador, self.config_screen)
        self.update_surface_dimensions()

    def update_surface_dimensions(self):
        # Cambio en la proporción para configuración
        config_width = int(self.screenSize[0] * 0.3)
        universe_width = self.screenSize[0] - config_width
        self.config_screen = pygame.Surface((config_width, self.screenSize[1]))
        self.universe_screen = pygame.Surface(
            (universe_width, self.screenSize[1]))
        self.config_window.update_screen(self.config_screen)

    def run(self):
        running = True
        dragging = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.config_window.button_rect.collidepoint((mouse_pos[0], mouse_pos[1])):
                        self.config_window.paused = not self.config_window.paused
                        self.config_window.update_button()
                        if self.config_window.paused:
                            self.entrenador.pausarEntrenamiento()
                        else:
                            self.entrenador.reanudarEntrenamiento()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.config_window.exit_button_rect.collidepoint((mouse_pos[0], mouse_pos[1])):
                        self.config_window.paused = not self.config_window.paused
                        self.entrenador.run = False
                        running = False
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        self.keys_pressed[event.key] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Botón izquierdo
                        dragging = True
                        mouse_x, mouse_y = event.pos
                    elif event.button == 4:  # Rueda hacia arriba
                        self.zoom_level += 0.1
                    elif event.button == 5:  # Rueda hacia abajo
                        self.zoom_level -= 0.1
                        if self.zoom_level < 0.1:
                            self.zoom_level = 0.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Botón izquierdo
                        dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx, dy = event.rel
                    self.view_offset[0] -= dx
                    self.view_offset[1] -= dy
                elif event.type == pygame.VIDEORESIZE:
                    self.screenSize = event.size
                    self.screen = pygame.display.set_mode(
                        self.screenSize, pygame.RESIZABLE)
                    self.update_surface_dimensions()
            # Desplazamiento continuo con las flechas
            if self.keys_pressed[pygame.K_LEFT]:
                self.view_offset[0] -= self.cellSize
            if self.keys_pressed[pygame.K_RIGHT]:
                self.view_offset[0] += self.cellSize
            if self.keys_pressed[pygame.K_UP]:
                self.view_offset[1] -= self.cellSize
            if self.keys_pressed[pygame.K_DOWN]:
                self.view_offset[1] += self.cellSize

            self.update_grid()
            self.config_window.refresh(self.entrenador)
            # Coloca primero la ventana de configuración
            self.screen.blit(self.config_screen, (0, 0))
            self.screen.blit(self.universe_screen,
                             (self.config_screen.get_width(), 0))
            pygame.display.flip()
            # Reducir el delay para un movimiento más suave
            pygame.time.delay(100)

        pygame.quit()

    def update_grid(self):
        self.universe_screen.fill((0, 0, 0))
        see_universo = {}
        try:
            self.entrenador.universos[0]
            universo_max = max(self.entrenador.universos, key=lambda u: u.tiempo)
            see_universo = universo_max
        except IndexError:
            return

        cargas = cp.asnumpy(see_universo.cargasMatriz)
        energias = cp.asnumpy(see_universo.energiasMatriz)

        # Ajustar las cargas y energías al rango 0-255
        min_carga, max_carga = np.min(cargas), np.max(cargas)
        min_energia, max_energia = np.min(energias), np.max(energias)
        cargas = ((cargas - min_carga) / (max_carga - min_carga)
                  * 255).astype(np.uint8)
        energias = ((energias - min_energia) /
                    (max_energia - min_energia) * 255).astype(np.uint8)

        for index, (carga, energia) in enumerate(zip(cargas.flat, energias.flat)):
            cellSize = self.cellSize * self.zoom_level
            x = (
                index % see_universo.procedural_rules.FILAS) * cellSize - self.view_offset[0]
            y = (index // see_universo.procedural_rules.COLUMNAS) * \
                cellSize - self.view_offset[1]

            if x + self.cellSize < 0 or x > self.universe_screen.get_width() or y + self.cellSize < 0 or y > self.screenSize[1]:
                continue

            color = (energia, carga, carga)
            pygame.draw.rect(self.universe_screen, color,
                             (x, y, cellSize, cellSize))


if __name__ == '__main__':
    entrenador = Entrenador()
    entrenador.iniciarEntrenamiento()
    while len(entrenador.universos) == 0:
        time.sleep(1)
    app = App(entrenador)
    app.run()
