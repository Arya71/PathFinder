import pygame
import time
import random
import math
from queue import PriorityQueue
from collections import deque

pygame.init()
pygame.font.init()

# --- Configuration and Constants ---

# Screen and Grid Dimensions
GRID_WIDTH, GRID_HEIGHT = 800, 800
UI_WIDTH = 300
ANALYSIS_PANEL_HEIGHT = 160  # Height for the new bottom panel
WIDTH, HEIGHT = GRID_WIDTH + UI_WIDTH, GRID_HEIGHT
ROWS = 50
GRID_SIZE = GRID_WIDTH // ROWS

# Window Setup
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced AI Pathfinder Simulation")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 80, 80)        # Closed Set
GREEN = (120, 220, 120)    # Open Set
BLUE = (100, 100, 255)     # Start Node
YELLOW = (255, 255, 100)   # End Node
PURPLE = (180, 120, 255)   # Path
GREY = (220, 220, 220)     # Grid Lines
UI_BG = (40, 40, 60)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (100, 150, 255)
OBSTACLE_COLOR = (80, 80, 90)
ANALYSIS_BG = (30, 30, 45) # Background for the analysis panel
PANEL_TEXT_HIGHLIGHT = (255, 255, 120) # Highlight for live stats

# Weighted terrain colors
GRASS_COLOR = (180, 245, 180)   # weight 2
WATER_COLOR = (170, 200, 255)   # weight 5
NORMAL_COLOR = WHITE            # weight 1

# --- Fonts ---
TITLE_FONT = pygame.font.SysFont('corbel', 30, bold=True)
TEXT_FONT = pygame.font.SysFont('corbel', 22)
STATS_FONT = pygame.font.SysFont('consolas', 20)
SMALL_FONT = pygame.font.SysFont('corbel', 18)
WEIGHT_FONT = pygame.font.SysFont('consolas', 14, bold=True)
ANALYSIS_FONT = pygame.font.SysFont('consolas', 18) # Font for analysis panel

# --- Obstacle Density Control (1–5 keys) ---
DENSITY_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
density_index = 2  # default 30%

# --- Brush Modes for editing tiles ---
BRUSH_WALL = "wall"
BRUSH_NORMAL = "normal"
BRUSH_GRASS = "grass"
BRUSH_WATER = "water"
brush_mode = BRUSH_WALL  # default: wall brush (matches your old L-click to place walls)


class Node:
    """
    Represents a single node in the grid. It handles its own state,
    color, position, neighbors, and traversal weight.
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * GRID_SIZE
        self.y = row * GRID_SIZE
        self.color = WHITE
        self.neighbors = []
        self.previous = None
        self.is_obstacle_flag = False
        self.weight = 1  # traversal cost to ENTER this node (1=normal)

    def get_pos(self):
        return self.row, self.col

    def is_obstacle(self):
        return self.is_obstacle_flag

    def set_weight(self, w, color):
        self.weight = w
        self.color = color
        self.is_obstacle_flag = False

    def make_normal(self):
        self.set_weight(1, NORMAL_COLOR)

    def make_grass(self):
        self.set_weight(2, GRASS_COLOR)

    def make_water(self):
        self.set_weight(5, WATER_COLOR)

    def reset(self):
        self.previous = None
        self.is_obstacle_flag = False
        self.make_normal()

    def make_start(self):
        self.color = BLUE
    
    def make_end(self):
        self.color = YELLOW

    def make_closed(self):
        if not self.is_obstacle():
            self.color = RED

    def make_open(self):
        if not self.is_obstacle():
            self.color = GREEN

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR
        self.is_obstacle_flag = True
        self.weight = float("inf")

    def make_path(self):
        if not self.is_obstacle():
            self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, GRID_SIZE, GRID_SIZE))
        if self.color != OBSTACLE_COLOR and GRID_SIZE > 10:
            pygame.draw.rect(win, GREY, (self.x, self.y, GRID_SIZE, GRID_SIZE), 1)

        if not self.is_obstacle_flag and self.weight > 1 and GRID_SIZE >= 14:
            label = WEIGHT_FONT.render(str(self.weight), True, BLACK)
            win.blit(label, (self.x + (GRID_SIZE - label.get_width()) // 2,
                             self.y + (GRID_SIZE - label.get_height()) // 2))

    def update_neighbors(self, grid, allow_diagonal):
        self.neighbors = []
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < ROWS - 1 and not grid[self.row][self.col + 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col - 1])

        if allow_diagonal:
            if self.row < ROWS - 1 and self.col < ROWS - 1 and not grid[self.row + 1][self.col + 1].is_obstacle():
                self.neighbors.append(grid[self.row + 1][self.col + 1])
            if self.row > 0 and self.col < ROWS - 1 and not grid[self.row - 1][self.col + 1].is_obstacle():
                self.neighbors.append(grid[self.row - 1][self.col + 1])
            if self.row < ROWS - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_obstacle():
                self.neighbors.append(grid[self.row + 1][self.col - 1])
            if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_obstacle():
                self.neighbors.append(grid[self.row - 1][self.col - 1])

# --- Heuristic Function ---
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# --- Path Reconstruction ---
def reconstruct_path(current, draw, path_length_ref):
    path_length = 0
    while current.previous:
        center1 = (current.x + GRID_SIZE // 2, current.y + GRID_SIZE // 2)
        center2 = (current.previous.x + GRID_SIZE // 2, current.previous.y + GRID_SIZE // 2)
        pygame.draw.line(win, (255, 255, 255), center1, center2, max(1, GRID_SIZE // 6))
        current.make_path()
        current = current.previous
        path_length += 1
    path_length_ref[0] = path_length
    draw()

# --- Random Obstacle Generation ---
def generate_random_obstacles(grid, start_node, end_node, density=0.3):
    for row in grid:
        for node in row:
            node.reset()
            if random.random() < density:
                node.make_obstacle()
    if start_node:
        start_node.reset()
        start_node.make_start()
    if end_node:
        end_node.reset()
        end_node.make_end()

# --- Pathfinding Algorithms (MODIFIED to accept live_stats) ---
def generic_search(draw, grid, start, end, algorithm_type, diagonal_allowed, live_stats):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    if algorithm_type == 'astar':
        f_score[start] = h(start.get_pos(), end.get_pos())
    elif algorithm_type == 'dijkstra':
        f_score[start] = 0
    elif algorithm_type == 'greedy':
        f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while not open_set.empty():
        pygame.time.delay(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = open_set.get()[2]
        if current in open_set_hash:
            open_set_hash.remove(current)
        nodes_explored += 1

        # Update live stats during the run
        live_stats['explored'] = nodes_explored
        live_stats['space'] = len(open_set_hash)

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}

        cr, cc = current.get_pos()
        for neighbor in current.neighbors:
            nr, nc = neighbor.get_pos()
            step_cost = neighbor.weight
            if diagonal_allowed and abs(cr - nr) == 1 and abs(cc - nc) == 1:
                step_cost *= math.sqrt(2)
            temp_g_score = g_score[current] + step_cost

            if temp_g_score < g_score[neighbor]:
                neighbor.previous = current
                g_score[neighbor] = temp_g_score
                heuristic = h(neighbor.get_pos(), end.get_pos())
                if algorithm_type == 'astar':
                    f_score[neighbor] = temp_g_score + heuristic
                elif algorithm_type == 'dijkstra':
                    f_score[neighbor] = temp_g_score
                elif algorithm_type == 'greedy':
                    f_score[neighbor] = heuristic
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        max_space = max(max_space, len(open_set_hash))
        draw()
        if current != start:
            current.make_closed()
    return False

def breadth_first_search(draw, grid, start, end, live_stats):
    queue = deque([start])
    visited = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while queue:
        pygame.time.delay(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return None
        current = queue.popleft()
        nodes_explored += 1

        # Update live stats
        live_stats['explored'] = nodes_explored
        live_stats['space'] = len(queue)

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                queue.append(neighbor)
                neighbor.make_open()
        max_space = max(max_space, len(queue))
        draw()
        if current != start: current.make_closed()
    return False

def depth_first_search(draw, grid, start, end, live_stats):
    stack = [start]
    visited = {start}
    nodes_explored = 0
    max_space = 0
    start_time = time.time()
    while stack:
        pygame.time.delay(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return None
        current = stack.pop()
        nodes_explored += 1

        # Update live stats
        live_stats['explored'] = nodes_explored
        live_stats['space'] = len(stack)

        if current != start: current.make_closed()
        draw()
        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {"time": time.time() - start_time, "explored": nodes_explored, "space": max_space, "path_len": path_length[0]}
        for neighbor in reversed(current.neighbors):
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                stack.append(neighbor)
                neighbor.make_open()
        max_space = max(max_space, len(stack))
    return False

# --- Maze Generation ---
def generate_maze(grid, start_node, end_node):
    for row in grid:
        for node in row:
            node.make_obstacle()
    
    def carve_passages(cx, cy, grid_ref):
        directions = [(0, 1), (1, 0), (0, -1), (0, -1)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx*2, cy + dy*2
            if 0 <= nx < ROWS and 0 <= ny < ROWS and grid_ref[nx][ny].is_obstacle():
                grid_ref[cx + dx][cy].reset()
                grid_ref[nx][ny].reset()
                carve_passages(nx, ny, grid_ref)

    start_x, start_y = random.randrange(1, ROWS, 2), random.randrange(1, ROWS, 2)
    grid[start_x][start_y].reset()
    carve_passages(start_x, start_y, grid)
    
    if start_node: start_node.reset(); start_node.make_start()
    if end_node: end_node.reset(); end_node.make_end()

# --- Drawing and UI ---
def create_grid():
    return [[Node(i, j) for j in range(ROWS)] for i in range(ROWS)]

def draw_grid_lines(win):
    for i in range(ROWS + 1):
        pygame.draw.line(win, GREY, (0, i * GRID_SIZE), (GRID_WIDTH, i * GRID_SIZE))
        pygame.draw.line(win, GREY, (i * GRID_SIZE, 0), (i * GRID_SIZE, GRID_HEIGHT))

def draw_ui(win, selected_algo, stats, algorithms, diagonal_allowed):
    ui_rect = pygame.Rect(GRID_WIDTH, 0, UI_WIDTH, HEIGHT)
    pygame.draw.rect(win, UI_BG, ui_rect)
    y_pos = 20
    title_text = TITLE_FONT.render("Pathfinding Visualizer", True, WHITE)
    win.blit(title_text, (GRID_WIDTH + (UI_WIDTH - title_text.get_width()) // 2, y_pos))
    y_pos += 50

    density_pct = int(DENSITY_LEVELS[density_index] * 100)
    # --- UI INSTRUCTIONS UPDATED ---
    instructions = [
        "CONTROLS:",
        "L-Click: Place Start, then End",
        "   ...then paint with Brush",
        "R-Click: Erase to Normal",
        "Space: Run Algorithm",
        "C: Clear Board",
        "M: Generate Maze",
        "R: Random Obstacles",
        f"1-5: Set Density ({density_pct}%)",
        "D: Toggle Diagonals",
        "A: Toggle Analysis Panel",
        "6-0: Select Algorithm",
        "",
        "BRUSH MODES:",
        "O: Wall (obstacle)",
        "N: Normal (cost 1)",
        "G: Grass (cost 2)",
        "W: Water (cost 5)",
    ]
    for line in instructions:
        line_text = SMALL_FONT.render(line, True, TEXT_COLOR)
        win.blit(line_text, (GRID_WIDTH + 20, y_pos))
        y_pos += 22
    y_pos += 4
    
    diag_status = "ON" if diagonal_allowed else "OFF"
    diag_color = GREEN if diagonal_allowed else RED
    diag_text = TEXT_FONT.render(f"Diagonal Movement: ", True, WHITE)
    diag_status_text = TEXT_FONT.render(diag_status, True, diag_color)
    win.blit(diag_text, (GRID_WIDTH + 20, y_pos))
    win.blit(diag_status_text, (GRID_WIDTH + 20 + diag_text.get_width(), y_pos))
    y_pos += 36

    dens_text = TEXT_FONT.render(f"Obstacle Density: {density_pct}%", True, WHITE)
    win.blit(dens_text, (GRID_WIDTH + 20, y_pos))
    y_pos += 28

    brush_label = {
        BRUSH_WALL: "Wall (Obstacle)",
        BRUSH_NORMAL: "Normal (1)",
        BRUSH_GRASS: "Grass (2)",
        BRUSH_WATER: "Water (5)"
    }[brush_mode]
    brush_text = TEXT_FONT.render(f"Brush: {brush_label}", True, WHITE)
    win.blit(brush_text, (GRID_WIDTH + 20, y_pos))
    y_pos += 36

    algo_title = TEXT_FONT.render("ALGORITHMS (6-0):", True, WHITE)
    win.blit(algo_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 30
    
    for i, (name, _) in enumerate(algorithms.items()):
        color = HIGHLIGHT_COLOR if name == selected_algo else TEXT_COLOR
        key_label = ["6", "7", "8", "9", "0"][i]
        key_text = f"{key_label}. {name}"
        line_text = TEXT_FONT.render(key_text, True, color)
        win.blit(line_text, (GRID_WIDTH + 25, y_pos))
        y_pos += 28
    y_pos += 16

    # --- STATS DISPLAY TITLE UPDATED ---
    stats_title = TEXT_FONT.render("LAST RUN STATS:", True, WHITE)
    win.blit(stats_title, (GRID_WIDTH + 20, y_pos))
    y_pos += 28
    
    if stats:
        if stats == "No Path Found":
            stat_text = STATS_FONT.render(stats, True, RED)
            win.blit(stat_text, (GRID_WIDTH + 25, y_pos))
        else:
            complexity = algorithms[selected_algo]['complexity']
            stat_lines = [
                f"Time Taken   : {stats['time']:.4f} s",
                f"Nodes Explored: {stats['explored']}",
                f"Path Length  : {stats['path_len']}",
                f"Space Used   : {stats['space']} nodes",
                "",
                f"Time Complexity : O({complexity[0]})",
                f"Space Complexity: O({complexity[1]})"
            ]
            for line in stat_lines:
                line_text = STATS_FONT.render(line, True, TEXT_COLOR)
                win.blit(line_text, (GRID_WIDTH + 25, y_pos))
                y_pos += 22

# --- NEW --- Function to draw the analysis panel
def draw_analysis_panel(win, algorithms, live_stats):
    panel_rect = pygame.Rect(0, HEIGHT - ANALYSIS_PANEL_HEIGHT, GRID_WIDTH, ANALYSIS_PANEL_HEIGHT)
    pygame.draw.rect(win, ANALYSIS_BG, panel_rect)
    pygame.draw.line(win, HIGHLIGHT_COLOR, (0, HEIGHT - ANALYSIS_PANEL_HEIGHT), (GRID_WIDTH, HEIGHT - ANALYSIS_PANEL_HEIGHT), 2)

    y_pos = HEIGHT - ANALYSIS_PANEL_HEIGHT + 10
    
    headers = ["Algorithm", "Time (Worst)", "Space (Worst)", "Explored", "Frontier"]
    col_widths = [160, 160, 160, 150, 150]
    x_pos = 20
    for i, header in enumerate(headers):
        header_text = ANALYSIS_FONT.render(header, True, WHITE)
        win.blit(header_text, (x_pos, y_pos))
        x_pos += col_widths[i]
    y_pos += 25

    for name, data in algorithms.items():
        is_running = live_stats and live_stats.get('algorithm_name') == name
        row_color = PANEL_TEXT_HIGHLIGHT if is_running else TEXT_COLOR
        
        x_pos = 20
        name_text = ANALYSIS_FONT.render(name, True, row_color)
        win.blit(name_text, (x_pos, y_pos))
        x_pos += col_widths[0]
        
        time_text = ANALYSIS_FONT.render(f"O({data['complexity'][0]})", True, row_color)
        win.blit(time_text, (x_pos, y_pos))
        x_pos += col_widths[1]

        space_text = ANALYSIS_FONT.render(f"O({data['complexity'][1]})", True, row_color)
        win.blit(space_text, (x_pos, y_pos))
        x_pos += col_widths[2]
        
        explored_val = str(live_stats.get('explored', '—')) if is_running else '—'
        space_val = str(live_stats.get('space', '—')) if is_running else '—'

        explored_text = ANALYSIS_FONT.render(explored_val, True, row_color)
        win.blit(explored_text, (x_pos, y_pos))
        x_pos += col_widths[3]
        
        space_text = ANALYSIS_FONT.render(space_val, True, row_color)
        win.blit(space_text, (x_pos, y_pos))

        y_pos += 22

# Main drawing function updated to handle the new panel
def draw_main(win, grid, selected_algo, stats, algorithms, diagonal_allowed, analysis_visible, live_stats):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid_lines(win)
    if analysis_visible:
        draw_analysis_panel(win, algorithms, live_stats)
    draw_ui(win, selected_algo, stats, algorithms, diagonal_allowed)
    pygame.display.update()

def get_clicked_pos(pos):
    x, y = pos
    if x >= GRID_WIDTH: return None, None
    return y // GRID_SIZE, x // GRID_SIZE

# --- Main Loop ---
def main(win):
    global density_index, brush_mode
    grid = create_grid()
    start, end = None, None
    
    algorithms = {
        "A* Search": {"func": generic_search, "complexity": ("b^d", "b^d")},
        "Dijkstra": {"func": generic_search, "complexity": ("V+E log V", "V")},
        "Greedy BFS": {"func": generic_search, "complexity": ("b^d", "b^d")},
        "BFS": {"func": breadth_first_search, "complexity": ("V+E", "V")},
        "DFS": {"func": depth_first_search, "complexity": ("V+E", "V")}
    }
    algo_keys = list(algorithms.keys())
    selected_algorithm_name = algo_keys[0]

    run = True
    started = False
    stats = {}
    diagonal_allowed = False
    analysis_panel_visible = False # State for panel visibility
    live_stats = {} # Dictionary for live algorithm data

    while run:
        draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed, analysis_panel_visible, live_stats)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False
            if started: 
                continue

            if pygame.mouse.get_pressed()[0]:
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: continue
                node = grid[row][col]
                if not start and node != end and not node.is_obstacle():
                    start = node
                    start.make_start()
                    start.weight = 1
                elif not end and node != start and not node.is_obstacle():
                    end = node
                    end.make_end()
                    end.weight = 1
                elif node != start and node != end:
                    if brush_mode == BRUSH_WALL: node.make_obstacle()
                    elif brush_mode == BRUSH_NORMAL: node.make_normal()
                    elif brush_mode == BRUSH_GRASS: node.make_grass()
                    elif brush_mode == BRUSH_WATER: node.make_water()

            elif pygame.mouse.get_pressed()[2]:
                row, col = get_clicked_pos(pygame.mouse.get_pos())
                if row is None: continue
                node = grid[row][col]
                if node == start: start = None
                elif node == end: end = None
                node.reset()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    stats = {}
                    live_stats = {'algorithm_name': selected_algorithm_name, 'explored': 0, 'space': 0}
                    
                    for row in grid:
                        for node in row: 
                            node.update_neighbors(grid, diagonal_allowed)
                    
                    draw_callback = lambda: draw_main(win, grid, selected_algorithm_name, stats, algorithms, diagonal_allowed, analysis_panel_visible, live_stats)
                    
                    started = True
                    func = algorithms[selected_algorithm_name]["func"]
                    
                    if selected_algorithm_name in ("A* Search", "Dijkstra", "Greedy BFS"):
                        algo_type = {'A* Search': 'astar', 'Dijkstra': 'dijkstra', 'Greedy BFS': 'greedy'}[selected_algorithm_name]
                        result = func(draw_callback, grid, start, end, algo_type, diagonal_allowed, live_stats)
                    else:
                        result = func(draw_callback, grid, start, end, live_stats)
                        
                    stats = "No Path Found" if not result else result
                    started = False
                    live_stats = {}

                if event.key == pygame.K_c:
                    start, end, stats = None, None, {}
                    grid = create_grid()
                
                if event.key == pygame.K_m:
                    generate_maze(grid, start, end)

                if event.key == pygame.K_r:
                    generate_random_obstacles(grid, start, end, DENSITY_LEVELS[density_index])

                if event.key == pygame.K_d:
                    diagonal_allowed = not diagonal_allowed
                
                if event.key == pygame.K_a:
                    analysis_panel_visible = not analysis_panel_visible

                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    density_index = event.key - pygame.K_1

                if event.key in (pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0):
                    key_to_index = {pygame.K_6: 0, pygame.K_7: 1, pygame.K_8: 2, pygame.K_9: 3, pygame.K_0: 4}
                    if key_to_index[event.key] < len(algo_keys):
                        selected_algorithm_name = algo_keys[key_to_index[event.key]]
                        stats = {}

                if event.key == pygame.K_o: brush_mode = BRUSH_WALL
                if event.key == pygame.K_n: brush_mode = BRUSH_NORMAL
                if event.key == pygame.K_g: brush_mode = BRUSH_GRASS
                if event.key == pygame.K_w: brush_mode = BRUSH_WATER

    pygame.quit()

if __name__ == "__main__":
    main(win)