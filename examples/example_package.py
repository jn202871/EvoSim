import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
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
