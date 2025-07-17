import numpy as np

def check_speciation(grid, size, orientation='vertical', position=None, sample_n=50):
    max_x, max_y = size
    pos = position or (max_x // 2 if orientation == 'vertical' else max_y // 2)
    all_coords = np.array([coord for coord, c in grid.items() if c['individual']])
    if all_coords.size == 0:
        return False
    left_mask = all_coords[:,0] < pos if orientation == 'vertical' else all_coords[:,1] < pos
    right_mask = ~left_mask
    left = all_coords[left_mask]
    right = all_coords[right_mask]
    if left.size == 0 or right.size == 0:
        return False
    l_idx = np.random.choice(len(left), min(sample_n, len(left)), replace=False)
    r_idx = np.random.choice(len(right), min(sample_n, len(right)), replace=False)
    left_s = left[l_idx]; right_s = right[r_idx]
    count = 0
    for l in left_s:
        L = grid[tuple(l)]['individual']
        for r in right_s:
            R = grid[tuple(r)]['individual']
            diff = np.abs(L['chromosome_traits'] - R['chromosome_traits'])
            if np.all(diff <= L['chromosome_reproduction']) and np.all(diff <= R['chromosome_reproduction']):
                count += 1
    rate = count / (len(left_s)*len(right_s))
    return rate

