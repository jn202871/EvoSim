import numpy as np

def crossover_mutate(a, b):
    p = np.random.randint(1, len(a))
    c = np.concatenate([a[:p], b[p:]])
    return c + np.random.normal(0, 0.1, len(a))

