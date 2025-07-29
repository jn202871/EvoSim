# EvoSim
EvoSim is a package developed to enable the simulation of various types of speciation in the simplest possible enviornment that would enable such a behavior. EvoSim allows for allopatric speciation to occur by implimenting spatial gentic transfer and reproduction between agents inside the simulation. As agents move around, eat food, and reproduce; the agents evolve over many generations to develop specific fitness-agnostic traits and preferences for those traits. EvoSim creates a single population of these agents then splits this population in half, allowing each half to evolve on their own further. After some time, random mutations cause the two groups to seperate to such a degree that they will not choose to reproduce across groups, thus achieving allopatric speciation.

## Example Usage
EvoSim first creates a simulation of some size, with some initial population and food density. Then, we burn in the inital population for a number of steps before adding a barrier. After the barrier is added, EvoSim periodically checks if speciation has occured, and to what degree. A full simulation loop and speciation check can be seen below:
'''python
from EvoSim import Simulation, DefaultController

sim = Simulation(size=(100,100),
                 pop_density=0.5,
                 food_density=0.75,
                 controller=DefaultController(),
                 logger=None,
                 barrier_orientation='vertical')

extinct, _ = sim.run(num_steps=20000)
sim.add_barrier()

speciated = False
while not speciated and not extinct:
    extinct, hybrid_rate = sim.run(num_steps=1000)
    if hybrid_rate < 0.05:
        speciated = True
    individuals = [c for c in sim.grid.values() if c['individual']]
    print(f"Hybrid Rate: {hybrid_rate} Population Size: {len(individuals)}")
'''

## Visualization
The internal simulation of EvoSim can be visualized via the following at any point:
'''python
vis = GridVisualizer(sim, interval=100, show_energy=True)
vis.show()
'''
