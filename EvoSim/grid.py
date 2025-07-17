def create_grid_environment(size: tuple) -> dict:
    return {(x, y): {'individual': None, 'food': None, 'blocked': False}
            for x in range(size[0]) for y in range(size[1])}

def add_barrier(grid: dict, size: tuple, orientation: str = 'vertical', position: int = None) -> None:
    max_x, max_y = size
    if orientation == 'vertical':
        x_bar = position or max_x // 2
        for y in range(max_y):
            grid[(x_bar, y)]['blocked'] = True
    else:
        y_bar = position or max_y // 2
        for x in range(max_x):
            grid[(x, y_bar)]['blocked'] = True

