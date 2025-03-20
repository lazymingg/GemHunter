
from typing import List, Tuple, Dict, Set

grid : List[str] = [['3', ' ', '2', ' '],
                    [' ', ' ', '2', ' '],
                    [' ', '3', ' ', ' ']]

def get_neigh(pos : Tuple) -> List[Tuple]:
    ajacent = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    neigh = []
    for i, j in ajacent:
        x, y = pos
        x += i
        y += j
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            neigh.append((x, y))
    
    return neigh

