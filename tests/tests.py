import itertools

import numpy as np
import pytest

from EvoSim.grid import create_grid_environment, add_barrier
from EvoSim.speciation import check_speciation


def test_create_grid():
    grid = create_grid_environment((50, 50))
    assert isinstance(grid, dict)
    assert len(grid) == 50 * 50

    for loc in itertools.product(range(50), range(50)):
        cell = grid[loc]
        assert isinstance(cell, dict)
        assert cell['individual'] is None
        assert cell['food'] is None
        assert cell['blocked'] is False


def test_speciation_no_divergence():
    grid = create_grid_environment((2, 2))
    # place two individuals with identical traits
    grid[(0, 0)]['individual'] = {
        'chromosome_action': np.zeros(3),
        'chromosome_reproduction': np.zeros(5),
        'chromosome_traits': np.zeros(5),
        'energy': 100,
        'age': 0
    }
    grid[(1, 1)]['individual'] = {
        'chromosome_action': np.zeros(3),
        'chromosome_reproduction': np.zeros(5),
        'chromosome_traits': np.ones(5),
        'energy': 100,
        'age': 0
    }

    rate = check_speciation(grid, (2, 2))
    assert rate == 0


def test_barrier_blocks_middle_column():
    grid = create_grid_environment((3, 3))
    add_barrier(grid, (3, 3))

    # barrier should block all cells in the middle column (x == 1)
    for y in range(3):
        assert grid[(1, y)]['blocked'], f"Cell (1,{y}) should be blocked"
