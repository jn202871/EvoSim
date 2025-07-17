import numpy as np
from .interfaces import Controller
from .utils import crossover_mutate

class DefaultController(Controller):
    def act(self, ind: dict, grid: dict, pos: tuple) -> tuple:
        actions = ['move', 'reproduce', 'stay']
        priorities = ind['chromosome_action']
        for action, _ in sorted(zip(actions, priorities), key=lambda x: x[1], reverse=True):
            if action == 'move':
                dirs = [(0,1),(1,0),(0,-1),(-1,0)]
                best, best_val = None, -1
                for dx,dy in dirs:
                    npos = (pos[0]+dx, pos[1]+dy)
                    if npos in grid and grid[npos]['food'] is not None and grid[npos]['individual'] is None and not grid[npos]['blocked']:
                        if grid[npos]['food'] > best_val:
                            best_val, best = grid[npos]['food'], npos
                if not best:
                    dx,dy = dirs[np.random.randint(4)]
                    npos = (pos[0]+dx, pos[1]+dy)
                    if npos in grid and not grid[npos]['blocked'] and grid[npos]['individual'] is None:
                        best = npos
                if best:
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
                if ind['energy'] <= 100: continue
                dirs = [(0,1),(1,0),(0,-1),(-1,0)]
                for dx,dy in dirs:
                    mate_pos = (pos[0]+dx, pos[1]+dy)
                    if mate_pos in grid:
                        mate = grid[mate_pos]['individual']
                        if mate and mate['energy'] > 100:
                            diff = np.abs(mate['chromosome_traits'] - ind['chromosome_traits'])
                            if np.all(diff <= ind['chromosome_reproduction']) and np.all(diff <= mate['chromosome_reproduction']):
                                for ddx,ddy in dirs + [(0,0)]:
                                    cpos = (pos[0]+ddx, pos[1]+ddy)
                                    if cpos in grid and grid[cpos]['individual'] is None and not grid[cpos]['blocked']:
                                        child = {
                                            'chromosome_action': crossover_mutate(ind['chromosome_action'], mate['chromosome_action']),
                                            'chromosome_reproduction': crossover_mutate(ind['chromosome_reproduction'], mate['chromosome_reproduction']),
                                            'chromosome_traits': crossover_mutate(ind['chromosome_traits'], mate['chromosome_traits']),
                                            'energy': 50,
                                            'age': 0
                                        }
                                        grid[cpos]['individual'] = child
                                        ind['energy']  -= 80
                                        mate['energy'] -= 80
                                        return pos
                continue

            if action == 'stay':
                ind['energy'] -= 5
                return pos
        return pos

