import numpy as np

def populate_grid(grid: dict, population_density: float, food_density: float) -> dict:
    for cell in grid:
        if grid[cell]['individual'] is None and np.random.rand() < population_density:
            grid[cell]['individual'] = {
                'chromosome_action': np.random.rand(3),
                'chromosome_reproduction': np.random.rand(5),
                'chromosome_traits': np.random.rand(5),
                'energy': 100,
                'age': 0
            }
        if np.random.rand() < food_density:
            grid[cell]['food'] = np.random.randint(30, 50)
    return grid

