from typing import List, Tuple
from itertools import combinations
from pysat.solvers import Solver
from copy import deepcopy

grid: List[List[str]] = [['3', ' ', '2', ' '],
                         [' ', ' ', '2', ' '],
                         [' ', '3', '1', ' ']]

def get_neigh(pos: Tuple[int, int], grid: List[List[str]]) -> List[Tuple[int, int]]:
    adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    neigh = []
    for di, dj in adjacent:
        x, y = pos
        x += di
        y += dj
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == ' ':
            neigh.append((x, y))
    return neigh

def convert_pos_to_int(pos: Tuple[int, int], grid: List[List[str]]) -> int:
    x, y = pos
    return x * len(grid[0]) + y + 1

def convert_int_to_pos(num: int, grid: List[List[str]]) -> Tuple[int, int]: 
    if num < 0:
        num = -num
    num -= 1
    x = num // len(grid[0])
    y = num % len(grid[0])
    return (x, y)

def generate_trap_combinations(array: List[int], n: int) -> List[List[int]]:
    clauses = []
    for combo in combinations(array, n):
        clause = list(combo) + [-v for v in array if v not in combo]
        clauses.append(clause)
    return clauses

def generate_cnf(grid: List[List[str]]) -> List[List[int]]:
    cnf = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == ' ':
                continue
            pos = (i, j)
            neigh = get_neigh(pos, grid)
            neigh_vars = [convert_pos_to_int(p, grid) for p in neigh]
            leng = len(neigh_vars) - int(grid[i][j]) + 1
            trap_clauses = generate_trap_combinations(neigh_vars, leng)
            cnf.extend(trap_clauses)
    return cnf

def get_grid_result(grid: List[List[str]], result: List[int]) -> List[List[str]]:
    new_grid = deepcopy(grid)
    for num in result:
        temp = 'T' if num > 0 else 'G'
        x, y = convert_int_to_pos(num, grid)
        if new_grid[x][y] == ' ':
            new_grid[x][y] = temp
    return new_grid

solver = Solver(name='glucose3')
cnf = generate_cnf(grid)
print ("CNF: ", cnf)

for row in grid:
    print(row)

for clause in cnf:
    solver.add_clause(clause)

if solver.solve():
    print("SAT (có nghiệm)")
    model = solver.get_model()
    print("Nghiệm:", model)
    grid_result = get_grid_result(grid, model)
    print("Lưới kết quả:")
    for row in grid_result:
        print(row)
else:
    print("UNSAT (không có nghiệm)")