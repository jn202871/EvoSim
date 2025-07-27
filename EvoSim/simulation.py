import numpy as np
from .grid import create_grid_environment, add_barrier
from .population import populate_grid
from .speciation import check_speciation

class Simulation:
    def __init__(self, size, pop_density, food_density,
                 controller, logger=None,
                 barrier_orientation='vertical',
                 barrier_position=None):
        self.size = size
        self.grid = create_grid_environment(size)
        self.grid = populate_grid(self.grid, pop_density, food_density)
        self.controller = controller
        self.logger = logger
        self.barrier_orientation = barrier_orientation
        self.barrier_position = barrier_position
        self.steps = 0
        self.speciated = False
        self.extinct = False
        self.has_barrier = False
        
    def add_barrier(self):
        if not self.has_barrier:
            add_barrier(self.grid, self.size, self.barrier_orientation, self.barrier_position)
            self.has_barrier = True

    def step(self):
        """Advance the simulation by a single time-step."""
        # Current individuals in the simulation
        snapshots = [(pos, self.grid[pos]['individual'])
                     for pos in self.grid if self.grid[pos]['individual']]
        for pos, ind in snapshots:
            # Skip if individual moved or died
            if self.grid[pos]['individual'] is not ind:
                continue
            # Ageing and passive energy loss
            ind['age'] += 1
            ind['energy'] -= np.sqrt(ind['age'])
            # Let controller decide on an action
            new_pos = self.controller.act(ind, self.grid, pos)
            # Remove individual if it ran out of energy
            if ind['energy'] <= 0 and self.grid[new_pos]['individual'] is ind:
                self.grid[new_pos]['individual'] = None

        # Add new food randomly
        for cell in self.grid:
            if self.grid[cell]['food'] is None and np.random.rand() < 0.1:
                self.grid[cell]['food'] = np.random.randint(30, 50)

        self.steps += 1

        # Notify logger after each step so that external tools such as visualizers can update
        if self.logger:
            self.logger.on_step(self.steps, self)

    def run(self, num_steps):
        """
        Run the simulation for the specified number of steps
        
        Parameters:
            num_steps: The number of steps to run the simulation for

        Returns:
            A tuple containing:
                - A boolean indicating if the population is extinct
                - The speciation rate
        """
        for _ in range(num_steps):  # Running the simulation for the specified number of steps
            self.step()

        # After running, determine population status and speciation rate
        individuals = [c for c in self.grid.values() if c['individual']]
        if len(individuals) == 0:
            self.extinct = True
            if self.logger:
                self.logger.on_extinction(self.steps, self)
            return True, 0
        else:
            return False, check_speciation(self.grid, self.size, self.barrier_orientation, self.barrier_position)
