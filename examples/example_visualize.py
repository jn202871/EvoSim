"""
Example script that runs a simulation and visualizes it in real-time.
"""

from EvoSim import Simulation, DefaultController, GridVisualizer

def main():
    # Setup the simulation
    sim = Simulation(
        size=(25, 25),           # Grid size (x, y)
        pop_density=0.05,        # Initial probability that a cell contains an individual
        food_density=0.10,       # Initial probability that a cell contains food
        controller=DefaultController(),
        barrier_orientation="vertical",
        barrier_position=None,
    )

    # Introduce a physical barrier
    sim.add_barrier()

    # Start the visualizer
    vis = GridVisualizer(sim, interval=100, show_energy=True)
    vis.show()

    sim.run(100)


if __name__ == "__main__":
    main()
