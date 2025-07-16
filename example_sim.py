# Simulates allopatric speciation between two species in a grid environment.
# Intended to be a working, all-in-one, example that we then turn into a package.

# Imports

import numpy as np

# First we create a grid environment, in this case a 100x100 grid.
# We should eventually add capability to create different types of environments.

def create_grid_environment(size: tuple) -> dict:
    """
    Creates a grid environment of given size. Each cell can hold:
      - an 'individual' (or None)
      - a food value (or None)
      - a blocked flag.
    """
    return {
        (x, y): {'individual': None, 'food': None, 'blocked': False}
        for x in range(size[0]) for y in range(size[1])
    }

# We need a way to split the grid into two sections.
def add_barrier(grid: dict,
                size: tuple,
                orientation: str = 'vertical',
                position: int = None) -> None:
    """
    Mark a line of cells as blocked so no one can move or reproduce across it.
    orientation: 'vertical' splits into left/right; 'horizontal' splits top/bottom.
    position: index of x or y coordinate for the barrier; defaults to center.
    """
    max_x, max_y = size
    if orientation == 'vertical':
        x_bar = position or max_x // 2
        for y in range(max_y):
            grid[(x_bar, y)]['blocked'] = True
    else:
        y_bar = position or max_y // 2
        for x in range(max_x):
            grid[(x, y_bar)]['blocked'] = True

# Now we populate the grid with individuals and food sources.
# Individuals contain a chromosome that defines their traits and behavior.

def populate_grid(grid: dict, population_density: float, food_density: float) -> dict:
    """
    Populates the grid with individuals and food sources.
    The population density defines how many cells are occupied by an individual.
    The food density defines how many cells contain food.
    Each individual has 3 chromosomes:
    one for action priorities,
    one for reproductive preferences,
    and one for reproductive traits.
    Each food source has a value that can be consumed by individuals.
    """
    for cell in grid:
        # Place an individual in the cell based on population density
        if grid[cell]['individual'] is None and np.random.rand() < population_density:
            # Create an individual with random chromosomes
            grid[cell]['individual'] = {
                'chromosome_action': np.random.rand(3),  # Random action priorities
                'chromosome_reproduction': np.random.rand(5),  # Random reproductive preferences
                'chromosome_traits': np.random.rand(5),  # Random reproductive traits
                'energy': 100,  # Starting energy for the individual
                'age': 0 # start at age 0
            }
        # Place food in the cell based on food density
        if np.random.rand() < food_density:
            # Create a food source with a random value
            grid[cell]['food'] = np.random.randint(30, 50) # Random food value between 10 and 30
    return grid

# Next, we define a function to decide how individuals interact with their environment.
# Individuals may choose a direction to act in. They interact with adjacent cells.
# When an individuals acts, it may either move to the chosen square (and consume food if present),
# or it may reproduce if it has a willing mate in the chosen cell.
# additionally, idividuals may choose to stay in their current cell, consuming less energy but not moving.

def individual_controller(ind: dict, grid: dict, pos: tuple) -> tuple:
    """
    Controls the actions of an individual based on its chromosomes and the environment.
    The individual uses a priority system, it tries high priority actions first then lower priority actions.
    If the individual has enough energy, it may choose to reproduce, otherwise it cannot.
    The individual has three main actions:
    1. Move to an adjacent cell (up, down, left, right), direction is chosen based on avalible food in adjacent cells.
    2. Reproduce with a willing mate in an adjacent cell, if it has enough energy.
    3. Stay in the current cell, consuming less energy but not moving.
    """
    actions = ['move', 'reproduce', 'stay']
    priorities = ind['chromosome_action']
    # sort by descending priority
    for action, _ in sorted(zip(actions, priorities), key=lambda x: x[1], reverse=True):
        if action == 'move':
            # find adjacent cell with max food
            dirs = [(0,1),(1,0),(0,-1),(-1,0)]
            best = None; best_val = -1
            for dx,dy in dirs:
                npos = (pos[0]+dx, pos[1]+dy)
                if npos in grid and grid[npos]['food'] is not None and grid[npos]['individual'] is None and not grid[npos]['blocked']:
                    if grid[npos]['food'] > best_val:
                        best_val = grid[npos]['food']; best = npos
            if best is None:
                # random move if unblocked & empty
                dx,dy = dirs[np.random.randint(4)]
                npos = (pos[0]+dx, pos[1]+dy)
                if npos in grid and not grid[npos]['blocked'] and grid[npos]['individual'] is None:
                    best = npos
            if best:
                # move
                grid[pos]['individual'] = None
                grid[best]['individual'] = ind
                ind['energy'] -= 10
                if grid[best]['food'] is not None:
                    ind['energy'] += grid[best]['food']
                    grid[best]['food'] = None
                return best
            else:
                ind['energy'] -= 5
                return pos

        if action == 'reproduce':
            if ind['energy'] <= 100:
                continue
            dirs = [(0,1),(1,0),(0,-1),(-1,0)]
            for dx,dy in dirs:
                mate_pos = (pos[0]+dx, pos[1]+dy)
                if mate_pos in grid:
                    mate = grid[mate_pos]['individual']
                    if mate is not None and mate['energy'] > 100:
                        diff = np.abs(mate['chromosome_traits'] - ind['chromosome_traits'])
                        if np.all(diff <= ind['chromosome_reproduction']) and np.all(diff <= mate['chromosome_reproduction']):
                            # find an empty spot for child
                            for ddx,ddy in dirs + [(0,0)]:
                                cpos = (pos[0]+ddx, pos[1]+ddy)
                                if cpos in grid and grid[cpos]['individual'] is None and not grid[cpos]['blocked']:
                                    # crossover + mutation
                                    def cx(a,b):
                                        p = np.random.randint(1, len(a))
                                        c = np.concatenate([a[:p], b[p:]])
                                        return c + np.random.normal(0,0.1,len(a))
                                    child = {
                                        'chromosome_action': cx(ind['chromosome_action'], mate['chromosome_action']),
                                        'chromosome_reproduction': cx(ind['chromosome_reproduction'], mate['chromosome_reproduction']),
                                        'chromosome_traits': cx(ind['chromosome_traits'], mate['chromosome_traits']),
                                        'energy': 50,
                                        'age': 0 
                                    }
                                    grid[cpos]['individual'] = child
                                    ind['energy']  -= 80
                                    mate['energy'] -= 80
                                    return pos
            # no successful mate found
            continue

        if action == 'stay':
            ind['energy'] -= 5
            return pos

    return pos

# We need a method to check speciation.
def check_speciation(grid: dict,
                     size: tuple,
                     orientation: str = 'vertical',
                     position: int = None,
                     sample_n: int = 50) -> bool:
    """
    Return True if the two sub-populations are reproductively isolated.
    We draw up to `sample_n` individuals from each side; if no pair
    satisfies the reproduction-compatibility test, we call it speciation.
    """
    max_x, max_y = size
    # pick barrier line
    if orientation == 'vertical':
        x_bar = position or max_x // 2
        mask_left  = [(x <  x_bar) for (x,y),c in grid.items() if c['individual'] is not None]
    else:
        y_bar = position or max_y // 2
        mask_left  = [(y <  y_bar) for (x,y),c in grid.items() if c['individual'] is not None]

    # collect coords
    all_coords = np.array([coord for coord,c in grid.items() if c['individual'] is not None])
    # boolean mask of same length
    if orientation == 'vertical':
        left_mask  = all_coords[:,0] < (position or max_x // 2)
    else:
        left_mask  = all_coords[:,1] < (position or max_y // 2)
    right_mask = ~left_mask

    left_arr  = all_coords[left_mask]
    right_arr = all_coords[right_mask]

    # extinction check
    if left_arr.size == 0 or right_arr.size == 0:
        return False

    # sample indices
    n_left  = min(sample_n, left_arr.shape[0])
    n_right = min(sample_n, right_arr.shape[0])
    idx_left  = np.random.choice(left_arr.shape[0],  size=n_left,  replace=False)
    idx_right = np.random.choice(right_arr.shape[0], size=n_right, replace=False)

    left_sample  = left_arr[idx_left]
    right_sample = right_arr[idx_right]

    # test compatibility
    interbreed_count = 0
    for l in left_sample:
        L = grid[(int(l[0]), int(l[1]))]['individual']
        for r in right_sample:
            R = grid[(int(r[0]), int(r[1]))]['individual']
            diff = np.abs(L['chromosome_traits'] - R['chromosome_traits'])
            if (np.all(diff <= L['chromosome_reproduction'])
             and np.all(diff <= R['chromosome_reproduction'])):
                interbreed_count += 1
    interbreed_percentage = interbreed_count / (n_left * n_right)
    print(f"Interbreeding percentage: {interbreed_percentage:.2%} "
            f"({interbreed_count} out of {n_left * n_right} pairs)")
    if interbreed_percentage > 0.05:
        print("Interbreeding detected, no speciation.")
        return False
    print("Very low interbreeding detected, speciation likely.")
    return True  # no cross-barrier mating possible → speciation!


# Now we need a simualtion manager function and a function to run the simulation.

def simulate_environment(grid: dict, steps: int):
    for _ in range(steps):
        # Snapshot all individuals at start of tick
        snapshots = [
            (pos, grid[pos]['individual'])
            for pos in grid if grid[pos]['individual'] is not None
        ]
        for pos, ind in snapshots:
            # Skip if moved or died
            if grid[pos]['individual'] is not ind:
                continue

            # Age up and pay sqrt(age) energy cost
            ind['age'] += 1
            ind['energy'] -= np.sqrt(ind['age'])

            # Act
            new_pos = individual_controller(ind, grid, pos)

            # Death check
            if ind['energy'] <= 0 and grid[new_pos]['individual'] is ind:
                grid[new_pos]['individual'] = None
        # Add new food sources randomly
        for cell in grid:
            if grid[cell]['food'] is None and np.random.rand() < 0.1:
                grid[cell]['food'] = np.random.randint(30, 50)
    return grid

def run_speciation_simulation(size: tuple,
                              population_density: float,
                              food_density: float,
                              burn_in_steps: int,
                              split_orientation: str = 'vertical',
                              split_position: int = None,
                              check_interval: int = 50,
                              max_steps: int = 200):
    # 1) set up
    grid = create_grid_environment(size)
    grid = populate_grid(grid, population_density, food_density)

    # 2) burn-in
    grid = simulate_environment(grid, burn_in_steps)
    steps = burn_in_steps

    # 3) insert barrier
    add_barrier(grid, size, split_orientation, split_position)
    print(f"Barrier placed at {split_orientation}={split_position or 'center'}")

    # Check if speciation already happened
    if check_speciation(grid, size, split_orientation, split_position):
        print("Speciation already achieved before simulation started.")
        return grid

    # 4) loop: simulate chunk, then check
    while steps < max_steps:
        grid = simulate_environment(grid, check_interval)
        steps += check_interval

        # check extinction
        total_inds = [c for c in grid.values() if c['individual'] is not None]
        if len(total_inds) == 0:
            print(f"Extinction occurred at step {steps}")
            break

        # check speciation
        if check_speciation(grid, size, split_orientation, split_position):
            print(f"Speciation achieved at step {steps}")
            break

    return grid

# Run the simulation
final_grid = run_speciation_simulation(
        size=(100,100),
        population_density=0.5,
        food_density=0.75,
        burn_in_steps=20_000,
        split_orientation='vertical',
        split_position=None,    # defaults to center
        check_interval=1_000,
        max_steps=20_000_000
    )
