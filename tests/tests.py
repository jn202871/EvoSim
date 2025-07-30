import unittest
import EvoSim
from EvoSim.grid import create_grid_environment
from EvoSim.speciation import check_speciation
from EvoSim.grid import add_barrier
import itertools
import numpy as np

class Tests(unittest.TestCase):
    def test_create_grid(self):
        grid = create_grid_environment((50,50))
        self.assertIsInstance(grid, dict)
        self.assertEqual(len(grid), 50*50)
        for loc in list(itertools.product(range(50), range(50))):
            self.assertIsInstance(grid[loc], dict)
            self.assertIsNone(grid[loc]['individual'])
            self.assertIsNone(grid[loc]['food'])
            self.assertFalse(grid[loc]['blocked'])

    def test_speciation(self):
        grid = create_grid_environment((2,2))
        grid[(0,0)]['individual'] = {
                'chromosome_action': np.zeros(3),
                'chromosome_reproduction': np.zeros(5),
                'chromosome_traits': np.zeros(5),
                'energy': 100,
                'age': 0
            }
        grid[(1,1)]['individual'] = {
                'chromosome_action': np.zeros(3),
                'chromosome_reproduction': np.zeros(5),
                'chromosome_traits': np.ones(5),
                'energy': 100,
                'age': 0
            }
        rate = check_speciation(grid, (2,2))
        self.assertEqual(rate, 0)

    def test_barrier(self):
        grid = create_grid_environment((3,3))
        add_barrier(grid, (3,3))
        for y in range(3):
            self.assertTrue(grid[(1,y)]['blocked'])

if __name__ == '__main__':
    unittest.main()
