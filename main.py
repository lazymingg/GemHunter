
from typing import List, Tuple, Dict, Set
from itertools import combinations
from pysat.solvers import Solver
from copy import deepcopy
grid : List[str] = [['3', ' ', '2', ' '],
                    [' ', ' ', '2', ' '],
                    [' ', '3', ' ', ' ']]

def get_neigh(pos : Tuple, grid : List[str]) -> List[Tuple]:
    ajacent = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    neigh = []
    for i, j in ajacent:
        x, y = pos
        x += i
        y += j
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == ' ':
            neigh.append((x, y))
    
    return neigh

def convert_pos_to_int(pos : Tuple, grid : List[str]) -> int:
    x, y = pos
    return x * len(grid[0]) + y + 1

def convert_int_to_pos(num : int, grid : List[str]) -> Tuple: 
    if num < 0:
        num = -num
    num -= 1

    
    x = num // len(grid[0])
    y = num % len(grid[0])
    return (x, y)

def convert_list_to_int(pos : List[Tuple], grid : List[str]) -> List[int]:
    return [convert_pos_to_int(p, grid) for p in pos]

def convert_int_to_list(num : List[int], grid : List[str]) -> List[Tuple]:
    return [convert_int_to_pos(n, grid) for n in num]

def generate_trap_combinations(array : List[Tuple], n : int) -> List[List[Tuple]]:
    temp = list(combinations(array, n))
    return temp


def generate_cnf(grid : List[str]) -> List[List[int]]:
    cnf = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == ' ':
                continue
            pos = (i, j)
            neigh = get_neigh(pos, grid)
            neigh = convert_list_to_int(neigh, grid)
            length = len(neigh) - int(grid[i][j]) + 1
            print(length)   
            for comb in generate_trap_combinations(neigh, length):
                temp = list(comb)
                cnf.append(temp)
    return cnf

def get_grid_result(grid : List[str], result : List[int]) -> List[str]:
    new_grid = deepcopy(grid)
    for i in range(len(result)):
        temp = None
        if result[i] < 0:
            temp = 'G'
        else:
            temp = 'T'
        x, y = convert_int_to_pos(result[i], grid)
        if new_grid[x][y] == ' ':
            new_grid[x][y] = temp
    return new_grid

# data = [1, 2, 3, 4]
# r = 3

# combs = list(combinations(data, r))

# print(type(combs[1]))
# print(list(combs[1]))


# neigh = get_neigh((0, 0))
# neigh = convert_list_to_int(neigh)
# print(neigh)

solver = Solver(name='glucose3')

cnf = generate_cnf(grid)
cnf.pop(0)

for row in grid:
    print(row)

for clause in cnf:
    solver.add_clause(clause)

if solver.solve():
    print("SAT (có nghiệm)")
    print("Nghiệm:", solver.get_model())
else:
    print("UNSAT (không có nghiệm)")

result = solver.get_model()
grid_result = get_grid_result(grid, result)
for row in grid_result:
    print(row)
