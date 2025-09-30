import pygame
import time
from queue import PriorityQueue
from collections import deque

pygame.init()
pygame.font.init()

# --- Configuration and Constants ---

# Screen and Grid Dimensions
GRID_WIDTH, GRID_HEIGHT = 800, 800
UI_WIDTH = 300
WIDTH, HEIGHT = GRID_WIDTH + UI_WIDTH, GRID_HEIGHT
ROWS = 50
GRID_SIZE = GRID_WIDTH // ROWS

# Window Setup
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced AI Pathfinder Simulation")

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      # Closed Set
GREEN = (0, 255, 0)    # Open Set
BLUE = (0, 0, 255)     # Start Node
YELLOW = (255, 255, 0) # End Node
PURPLE = (128, 0, 128) # Path
GREY = (128, 128, 128) # Grid Lines
UI_BG = (40, 40, 60)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (100, 150, 255)

# --- Fonts ---
TITLE_FONT = pygame.font.SysFont('corbel', 30, bold=True)
TEXT_FONT = pygame.font.SysFont('corbel', 22)
STATS_FONT = pygame.font.SysFont('consolas', 20)

class Node:
    """
    Represents a single node in the grid. It handles its own state,
    color, position, and neighbors.
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * GRID_SIZE  # Note: x corresponds to col
        self.y = row * GRID_SIZE  # Note: y corresponds to row
        self.color = WHITE
        self.neighbors = []
        self.previous = None

    def get_pos(self):
        return self.row, self.col

    # State checking methods
    def is_obstacle(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == BLUE

    def is_end(self):
        return self.color == YELLOW

    # State setting methods
    def reset(self):
        self.color = WHITE
        self.previous = None

    def make_start(self):
        self.color = BLUE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_obstacle(self):
        self.color = BLACK

    def make_end(self):
        self.color = YELLOW

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, GRID_SIZE, GRID_SIZE))

    def update_neighbors(self, grid):
        """ Populates the neighbors list with valid adjacent nodes. """
        self.neighbors = []
        # Down
        if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row + 1][self.col])
        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row - 1][self.col])
        # Right
        if self.col < ROWS - 1 and not grid[self.row][self.col + 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col + 1])
        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col - 1])

# --- Heuristic Function ---
def h(p1, p2):
    """ Manhattan distance heuristic for A* """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# --- Path Reconstruction ---
def reconstruct_path(current, draw, path_length_ref):
    """ Draws the final path by backtracking from the end node. """
    path_length = 0
    while current.previous:
        current = current.previous
        current.make_path()
        path_length += 1
        draw()
    path_length_ref[0] = path_length


# --- Pathfinding Algorithms ---

def a_star_algorithm(draw, grid, start, end):
    """ A* Search Algorithm Implementation """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}
    
    nodes_explored = 0
    max_space = 0

    start_time = time.time()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = open_set.get()[2]
        open_set_hash.remove(current)
        nodes_explored += 1

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {
                "time": time.time() - start_time,
                "explored": nodes_explored,
                "space": max_space,
                "path_len": path_length[0]
            }

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                neighbor.previous = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        max_space = max(max_space, len(open_set_hash))
        draw()

        if current != start:
            current.make_closed()

    return False # Path not found

def dijkstra_algorithm(draw, grid, start, end):
    """ Dijkstra's Algorithm Implementation (A* with h=0) """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start)) # Priority is just g_score

    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    
    open_set_hash = {start}
    
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = open_set.get()[2]
        open_set_hash.remove(current)
        nodes_explored += 1

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {
                "time": time.time() - start_time,
                "explored": nodes_explored,
                "space": max_space,
                "path_len": path_length[0]
            }

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                neighbor.previous = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        max_space = max(max_space, len(open_set_hash))
        draw()

        if current != start:
            current.make_closed()

    return False

def breadth_first_search(draw, grid, start, end):
    """ Breadth-First Search (BFS) Implementation """
    queue = deque([start])
    visited = {start}
    
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = queue.popleft()
        nodes_explored += 1

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {
                "time": time.time() - start_time,
                "explored": nodes_explored,
                "space": max_space,
                "path_len": path_length[0]
            }

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                queue.append(neighbor)
                neighbor.make_open()
        
        max_space = max(max_space, len(queue))
        draw()

        if current != start:
            current.make_closed()

    return False

def depth_first_search(draw, grid, start, end):
    """ Depth-First Search (DFS) Implementation """
    stack = [start]
    visited = {start}
    
    nodes_explored = 0
    max_space = 0
    start_time = time.time()

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        current = stack.pop()
        nodes_explored += 1

        if current != start:
            current.make_closed()
        draw() # Draw after making closed to see the exploration

        if current == end:
            end.make_end()
            path_length = [0]
            reconstruct_path(end, draw, path_length)
            start.make_start()
            return {
                "time": time.time() - start_time,
                "explored": nodes_explored,
                "space": max_space,
                "path_len": path_length[0]
            }

        # We add neighbors in reverse to explore in a more standard order (e.g., L, R, U, D)
        for neighbor in reversed(current.neighbors):
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.previous = current
                stack.append(neighbor)
                neighbor.make_open()
        
        max_space = max(max_space, len(stack))

    return False

# --- Drawing and UI ---

def create_grid():
    """ Initializes the grid with Node objects. """
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(ROWS):
            node = Node(i, j)
            grid[i].append(node)
    return grid

def draw_grid_lines(win):
    """ Draws the lines for the grid. """
    for i in range(ROWS + 1):
        pygame.draw.line(win, GREY, (0, i * GRID_SIZE), (GRID_WIDTH, i * GRID_SIZE))
        pygame.draw.line(win, GREY, (i * GRID_SIZE, 0), (i * GRID_SIZE, GRID_HEIGHT))

def draw_ui(win, selected_algo, stats, algorithms):
    """ Draws the entire UI panel on the right side of the screen. """
    ui_rect = pygame.Rect(GRID_WIDTH, 0, UI_WIDTH, HEIGHT)
    pygame.draw.rect(win, UI_BG, ui_rect)

    # Title
    title_text = TITLE_FONT.render("Pathfinding Visualizer", True, WHITE)
    win.blit(title_text, (GRID_WIDTH + (UI_WIDTH - title_text.get_width()) // 2, 20))

    # Instructions
    instructions = [
        "CONTROLS:",
        "L-Click: Place Start/End/Wall",
        "R-Click: Erase Node",
        "Space: Run Algorithm",
        "C: Clear Board",
        "1-4: Select Algorithm"
    ]
    for i, line in enumerate(instructions):
        line_text = TEXT_FONT.render(line, True, TEXT_COLOR)
        win.blit(line_text, (GRID_WIDTH + 20, 80 + i * 25))

    # Algorithm Selection
    algo_title = TEXT_FONT.render("ALGORITHMS:", True, WHITE)
    win.blit(algo_title, (GRID_WIDTH + 20, 240))
    
    for i, (name, _) in enumerate(algorithms.items()):
        color = HIGHLIGHT_COLOR if name == selected_algo else TEXT_COLOR
        key_text = f"{i+1}. {name}"
        line_text = TEXT_FONT.render(key_text, True, color)
        win.blit(line_text, (GRID_WIDTH + 25, 270 + i * 28))

    # Stats Display
    stats_title = TEXT_FONT.render("STATISTICS:", True, WHITE)
    win.blit(stats_title, (GRID_WIDTH + 20, 420))
    
    if stats:
        if stats == "No Path Found":
            stat_text = STATS_FONT.render(stats, True, RED)
            win.blit(stat_text, (GRID_WIDTH + 25, 450))
        else:
            complexity = algorithms[selected_algo]['complexity']
            stat_lines = [
                f"Time Taken  : {stats['time']:.4f} s",
                f"Nodes Explored: {stats['explored']}",
                f"Path Length : {stats['path_len']}",
                f"Space Used  : {stats['space']} nodes",
                "",
                f"Time Complexity: O({complexity[0]})",
                f"Space Complexity: O({complexity[1]})"
            ]
            for i, line in enumerate(stat_lines):
                line_text = STATS_FONT.render(line, True, TEXT_COLOR)
                win.blit(line_text, (GRID_WIDTH + 25, 450 + i * 22))

def draw_main(win, grid, selected_algo, stats, algorithms):
    """ Main drawing function, called every frame. """
    win.fill(WHITE)
    # Draw grid nodes
    for row in grid:
        for node in row:
            node.draw(win)
    # Draw grid lines
    draw_grid_lines(win)
    # Draw UI
    draw_ui(win, selected_algo, stats, algorithms)
    pygame.display.update()

def get_clicked_pos(pos):
    """ Converts mouse coordinates to grid row/col. """
    x, y = pos
    if x >= GRID_WIDTH: # Click is in the UI panel
        return None, None
    row = y // GRID_SIZE
    col = x // GRID_SIZE
    return row, col

# --- Main Loop ---

def main(win):
    grid = create_grid()
    start = None
    end = None
    
    algorithms = {
        "A* Search": {"func": a_star_algorithm, "complexity": ("b^d", "b^d")},
        "Dijkstra": {"func": dijkstra_algorithm, "complexity": ("V+E log V", "V")},
        "BFS": {"func": breadth_first_search, "complexity": ("V+E", "V")},
        "DFS": {"func": depth_first_search, "complexity": ("V+E", "V")}
    }
    algo_keys = list(algorithms.keys())
    selected_algorithm_name = algo_keys[0]

    run = True
    started = False
    stats = {}

    while run:
        draw_main(win, grid, selected_algorithm_name, stats, algorithms)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started: # Don't allow input while algorithm is running
                continue

            # --- MOUSE CLICKS ---
            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                if row is None: continue

                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != end and node != start:
                    node.make_obstacle()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                if row is None: continue

                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
            
            # --- KEY PRESSES ---
            if event.type == pygame.KEYDOWN:
                # Start Algorithm
                if event.key == pygame.K_SPACE and start and end:
                    stats = {} # Clear old stats
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    started = True
                    algorithm_func = algorithms[selected_algorithm_name]["func"]
                    result = algorithm_func(lambda: draw_main(win, grid, selected_algorithm_name, stats, algorithms), grid, start, end)
                    if not result:
                        stats = "No Path Found"
                    else:
                        stats = result
                    started = False

                # Clear Board
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = create_grid()
                    stats = {}
                
                # Select Algorithm
                if event.key == pygame.K_1:
                    selected_algorithm_name = algo_keys[0]
                    stats = {}
                if event.key == pygame.K_2:
                    selected_algorithm_name = algo_keys[1]
                    stats = {}
                if event.key == pygame.K_3:
                    selected_algorithm_name = algo_keys[2]
                    stats = {}
                if event.key == pygame.K_4:
                    selected_algorithm_name = algo_keys[3]
                    stats = {}


    pygame.quit()

if __name__ == "__main__":
    main(win)
