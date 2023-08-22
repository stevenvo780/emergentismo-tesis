import pygame
from entrenador import Entrenador
from types_universo import systemRules
import cupy as cp
pygame.init()


class ConfigWindow:
    def __init__(self, entrenador, screen):
        self.entrenador = entrenador
        self.screen = screen
        self.font = pygame.font.Font(None, 24)

    def update_screen(self, screen):
        self.screen = screen

    def run(self):
        self.update_configurations()

    def refresh(self, entrenador):
        self.entrenador = entrenador
        self.screen.fill((255, 255, 255))  # Fondo blanco
        physics_rules = vars(self.entrenador.universo.physics_rules).items()
        system_rules = vars(systemRules).items()
        time_label = self.font.render(
            f"Tiempo: {self.entrenador.universo.tiempo}", True, (0, 0, 0))
        time_structure_label = self.font.render(
            f"Tiempo sin estructura: {self.entrenador.tiempoSinEstructuras}", True, (0, 0, 0))
        id_label = self.font.render(
            f"ID: {self.entrenador.universo.id}", True, (0, 0, 0))

        self.screen.blit(time_label, (10, 10))
        self.screen.blit(time_structure_label, (10, 35))
        self.screen.blit(id_label, (10, 55))

        for i, (attribute, value) in enumerate(system_rules):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (10, 80 + i * 20))

        for i, (attribute, value) in enumerate(physics_rules):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (10, (500 + i * 20)))

    def update_configurations(self):
        for i, (attribute, value) in enumerate(vars(self.entrenador.universo.physics_rules).items()):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(
                label, (self.screen.get_width() // 2 + 10, 10 + i * 20))


class App:
    def __init__(self, entrenador):
        self.entrenador = entrenador
        self.gridSize = int(len(self.entrenador.universo.nodos) ** 0.5)
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
        universe_width = int(self.screenSize[0] * 0.8)
        config_width = self.screenSize[0] - universe_width
        self.universe_screen = pygame.Surface(
            (universe_width, self.screenSize[1]))
        self.config_screen = pygame.Surface((config_width, self.screenSize[1]))
        self.config_window.update_screen(self.config_screen)

    def run(self):
        running = True
        dragging = False
        while running:
            for event in pygame.event.get():
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
            self.screen.blit(self.universe_screen, (0, 0))
            self.screen.blit(self.config_screen,
                             (self.universe_screen.get_width(), 0))
            pygame.display.flip()
            # Reducir el delay para un movimiento más suave
            pygame.time.delay(50)

        pygame.quit()

    def update_grid(self):
        self.universe_screen.fill((0, 0, 0))   # Limpiar la pantalla

        cargas = cp.asnumpy(self.entrenador.universo.cargasMatriz)
        energias = cp.asnumpy(self.entrenador.universo.energiasMatriz)

        for index, (carga, energia) in enumerate(zip(cargas, energias)):
            cellSize = self.cellSize * self.zoom_level
            x = (index % self.gridSize) * cellSize - self.view_offset[0]
            y = (index // self.gridSize) * cellSize - self.view_offset[1]

            # Continuar con el siguiente nodo si está fuera de la ventana de visualización
            if x + self.cellSize < 0 or x > self.universe_screen.get_width() or y + self.cellSize < 0 or y > self.screenSize[1]:
                continue

            if energia > self.entrenador.universo.physics_rules.ENERGIA and carga > systemRules.LIMITE_RELACIONAL:
                color = (255, 255, 0)
            else:
                if carga > 0:
                    blueComponent = max(0, min(255, int(255 * carga)))
                    color = (0, 200, blueComponent)
                else:
                    cargas_value = abs(carga)
                    if cargas_value != float('inf'):
                        greyComponent = max(
                            0, min(255, 200 - int(255 * cargas_value)))
                        color = (greyComponent, greyComponent, greyComponent)
                    else:
                        color = (0, 0, 0)

            pygame.draw.rect(self.universe_screen, color,
                             (x, y, cellSize, cellSize))


if __name__ == '__main__':
    entrenador = Entrenador()
    entrenador.iniciarEntrenamiento()
    app = App(entrenador)
    app.run()
