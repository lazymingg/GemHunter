from typing import List, Tuple
from itertools import combinations
from pysat.solvers import Solver
from copy import deepcopy
from itertools import product

grid: List[List[str]] = [['3', '_', '2', '_'],
                         ['_', '_', '2', '_'],
                         ['_', '3', '1', '_']]

def read_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        array_2d = [line.strip().split(", ") for line in file]
    return array_2d

# grid = read_input('input_3.txt')
# for row in grid:
#     print(row)


def get_neigh(pos: Tuple[int, int], grid: List[List[str]]) -> List[Tuple[int, int]]:
    adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    neigh = []
    for di, dj in adjacent:
        x, y = pos
        x += di
        y += dj
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '_':
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
    # ít hơn n trap
    leng = len(array) - n + 1
    for combo in combinations(array, leng):
        clause = list(combo)
        clauses.append(clause)
    k = n + 1
    #không quá n trap
    if k <= len(array):
        for combo in combinations(array, k):
            clause = [-var for var in combo]
            clauses.append(clause)
    return clauses

def generate_cnf(grid: List[List[str]]) -> List[List[int]]:
    cnf = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '_':
                continue
            pos = (i, j)
            neigh = get_neigh(pos, grid)
            neigh_vars = [convert_pos_to_int(p, grid) for p in neigh]
            trap_clauses = generate_trap_combinations(neigh_vars, int(grid[i][j]))
            cnf.extend(trap_clauses)
    return cnf



# solving cnf with brute force 
def checking_clause(clause: List[int], model: List[int]) -> bool:
    for var in clause:
        if var > 0 and var in model:
            return True
        if var < 0 and -var not in model:
            return True
    return False

def checking_cnf(cnf: List[List[int]], model: List[int]) -> bool:
    for clause in cnf:
        if not checking_clause(clause, model):
            return False
    return True

def backtracking_cnf(cnf: List[List[int]]) -> List[int]:
    all_vars = set()
    for clause in cnf:
        all_vars.update(clause)
    all_vars = list(all_vars)
    all_vars.sort()
    all_vars = [abs(var) for var in all_vars]
    return backtracking_util(cnf, [], all_vars, 0)

def backtracking_util(cnf: List[List[int]], model: List[int], all_vars: List[int], index: int) -> List[int]:
    if index == len(all_vars):
        if checking_cnf(cnf, model):
            return deepcopy(model)
        return []
    var = all_vars[index]
    model.append(var)
    result = backtracking_util(cnf, model, all_vars, index + 1)
    if result:
        return result
    model.pop()
    model.append(-var)
    result = backtracking_util(cnf, model, all_vars, index + 1)
    if result:
        return result
    model.pop()
    return []



# solve by brute force
def generate_all_combinations(all_vars: List[int]) -> List[List[int]]:
    n = len(all_vars)
    binary_combinations = product([0, 1], repeat=n)
    all_models = []
    for binary in binary_combinations:
        model = []
        for i, value in enumerate(binary):
            if value == 1:
                model.append(all_vars[i])
            else:
                model.append(-all_vars[i])
        all_models.append(model)
    return all_models

def solve_by_brute_force(cnf: List[List[int]]) -> List[int]:
    all_vars = sorted(set(abs(v) for clause in cnf for v in clause))
    all_models = generate_all_combinations(all_vars)
    
    for model in all_models:
        if checking_cnf(cnf, model):
            return model
    return None




def get_grid_result(grid: List[List[str]], result: List[int]) -> List[List[str]]:
    new_grid = deepcopy(grid)
    for num in result:
        temp = 'T' if num > 0 else 'G'
        x, y = convert_int_to_pos(num, grid)
        if new_grid[x][y] == '_':
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

# backtracking test
result = backtracking_cnf(cnf)

if result:
    print("SAT (có nghiệm)")
    print("Nghiệm:", result)
    grid_result = get_grid_result(grid, result)
    print("Lưới kết quả:")
    for row in grid_result:
        print(row)


# brute force test
result = solve_by_brute_force(cnf)

if result:
    print("SAT (có nghiệm)")
    print("Nghiệm:", result)
    grid_result = get_grid_result(grid, result)
    print("Lưới kết quả:")
    for row in grid_result:
        print(row)