import pygame
from entrenador import Entrenador
from types_universo import SystemRules, PhysicsRules

pygame.init()

class ConfigWindow:
    def __init__(self, entrenador, screen):
        self.entrenador = entrenador
        self.screen = screen
        self.font = pygame.font.Font(None, 24)

    def run(self):
        self.update_configurations()

    def refresh(self, entrenador):
        self.entrenador = entrenador
        self.screen.fill((255, 255, 255))  # Fondo blanco
        system_values = vars(self.entrenador.universo.valoresSistema).items()
        time_label = self.font.render(f"Tiempo: {self.entrenador.universo.tiempo}", True, (0, 0, 0))
        time_structure_label = self.font.render(f"Tiempo sin estructura: {self.entrenador.tiempoSinEstructuras}", True, (0, 0, 0))
        id_label = self.font.render(f"ID: {self.entrenador.universo.id}", True, (0, 0, 0))

        self.screen.blit(time_label, (10, 10))
        self.screen.blit(time_structure_label, (10, 35))
        self.screen.blit(id_label, (10, 55))

        for i, (attribute, value) in enumerate(system_values):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(label, (10, 80 + i * 15))

    def update_configurations(self):
        for i, (attribute, value) in enumerate(vars(self.entrenador.universo.valoresSistema).items()):
            label = self.font.render(f"{attribute}: {value}", True, (0, 0, 0))
            self.screen.blit(
                label, (self.screen.get_width() // 2 + 10, 10 + i * 20))



class App:
    def __init__(self):
        self.entrenador = Entrenador()
        self.entrenador.iniciarEntrenamiento()
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
        universe_width = int(self.screenSize[0] * 0.8)  # 80% de la pantalla
        config_width = self.screenSize[0] - \
            universe_width  # 20% de la pantalla
        self.universe_screen = pygame.Surface(
            (universe_width, self.screenSize[1]))
        self.config_screen = pygame.Surface((config_width, self.screenSize[1]))
        self.config_window = ConfigWindow(
            self.entrenador, self.config_screen)

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
        new_gridSize = int(len(self.entrenador.universo.nodos) ** 0.5)

        # Si la retícula ha crecido, actualizar la gridSize
        if new_gridSize != self.gridSize:
            self.gridSize = new_gridSize

        self.universe_screen.fill((0, 0, 0))   # Limpiar la pantalla

        for index, nodo in enumerate(self.entrenador.universo.nodos):
            cellSize = self.cellSize * self.zoom_level
            x = (index % self.gridSize) * cellSize - self.view_offset[0]
            y = (index // self.gridSize) * cellSize - self.view_offset[1]

            # Continuar con el siguiente nodo si está fuera de la ventana de visualización
            if x + self.cellSize < 0 or x > self.universe_screen.get_width() or y + self.cellSize < 0 or y > self.screenSize[1]:
                continue

            if nodo.memoria.energia > self.entrenador.universo.valoresSistema.ENERGIA and len(nodo.memoria.relaciones) > SystemRules.LIMITE_RELACIONAL:
                color = (255, 255, 0)
            else:
                if nodo.memoria.cargas > 0:
                    blueComponent = max(
                        0, min(255, int(255 * nodo.memoria.cargas)))
                    color = (0, 200, blueComponent)
                else:
                    greyComponent = max(
                        0, min(255, 200 - int(255 * abs(nodo.memoria.cargas))))
                    color = (greyComponent, greyComponent, greyComponent)

            # Cambia la siguiente línea para dibujar en self.universe_screen en lugar de self.screen
            pygame.draw.rect(self.universe_screen, color,
                             (x, y, cellSize, cellSize))


if __name__ == '__main__':
    app = App()
    app.run()
