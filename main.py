from typing import List, Tuple
from itertools import combinations
from pysat.solvers import Solver
from copy import deepcopy
import argparse
import sys

def read_input(file_path: str) -> List[List[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            array_2d = [line.strip().split(", ") for line in file]
        return array_2d
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)

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
    num = abs(num) - 1
    x = num // len(grid[0])
    y = num % len(grid[0])
    return (x, y)

def generate_trap_combinations(array: List[int], n: int) -> List[List[int]]:
    clauses = []
    leng = len(array) - n + 1
    
    for combo in combinations(array, leng):
        clauses.append(list(combo))
    k = n + 1
    if k <= len(array):
        for combo in combinations(array, k):
            clauses.append([-var for var in combo])
    return clauses

def generate_cnf(grid: List[List[str]]) -> List[List[int]]:
    cnf = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != '_':
                pos = (i, j)
                neigh = get_neigh(pos, grid)
                neigh_vars = [convert_pos_to_int(p, grid) for p in neigh]
                if len(neigh_vars) - int(grid[i][j]) < 0:
                    cnf.append([]) #cant solve
                else:
                    trap_clauses = generate_trap_combinations(neigh_vars, int(grid[i][j]))
                    cnf.extend(trap_clauses)
    return cnf

def checking_clause(clause: List[int], model: List[int]) -> bool:
    for var in clause:
        if var > 0 and var in model:
            return True
        if var < 0 and -var not in model:
            return True
    return False

def checking_cnf(cnf: List[List[int]], model: List[int]) -> bool:
    return all(checking_clause(clause, model) for clause in cnf)

def backtracking_cnf(cnf: List[List[int]]) -> List[int]:
    all_vars = sorted(set(abs(var) for clause in cnf for var in clause))
    return backtracking_util(cnf, [], all_vars, 0)

def backtracking_util(cnf: List[List[int]], model: List[int], all_vars: List[int], index: int) -> List[int]:
    if index == len(all_vars):
        return deepcopy(model) if checking_cnf(cnf, model) else []
    var = all_vars[index]
    for value in [var, -var]:
        model.append(value)
        result = backtracking_util(cnf, model, all_vars, index + 1)
        if result:
            return result
        model.pop()
    return []

def solve_by_brute_force(cnf: List[List[int]]) -> List[int]:
    all_vars = sorted(set(abs(v) for clause in cnf for v in clause))
    n = len(all_vars)
    for num in range(1 << n):
        model = [all_vars[i] if (num >> i) & 1 else -all_vars[i] for i in range(n)]
        if checking_cnf(cnf, model):
            return model
    return None

def solve_by_sat(cnf: List[List[int]]) -> List[int]:
    solver = Solver(name='glucose3')
    for clause in cnf:
        solver.add_clause(clause)
    return solver.get_model() if solver.solve() else None

def get_grid_result(grid: List[List[str]], result: List[int]) -> List[List[str]]:
    new_grid = deepcopy(grid)
    for num in result:
        temp = 'T' if num > 0 else 'G'
        x, y = convert_int_to_pos(num, grid)
        if new_grid[x][y] == '_':
            new_grid[x][y] = temp
    return new_grid

def save_results(filename: str, method: str, grid: List[List[str]], result: List[int], original_grid: List[List[str]]):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Solving method: {method}\n\n")
        f.write("Original Grid:\n")
        for row in original_grid:
            f.write(' '.join(row) + '\n')
        f.write("\nCNF Result:\n")
        if result:
            f.write("SAT (satisfiable)\n")
            f.write(f"Model: {result}\n\n")
            f.write("Result Grid:\n")
            for row in grid:
                f.write(' '.join(row) + '\n')
        else:
            f.write("UNSAT (unsatisfiable)\n")

def main():
    parser = argparse.ArgumentParser(description="Solve grid puzzle with different methods")
    parser.add_argument('input_file', help="Input file path")
    parser.add_argument('--method', choices=['sat', 'backtracking', 'brute'], 
                       default='sat', help="Solving method (default: sat)")
    parser.add_argument('--output', default='result.txt', 
                       help="Output file path (default: result.txt)")
    
    args = parser.parse_args()

    # Read input
    grid = read_input(args.input_file)
    
    # Generate CNF
    cnf = generate_cnf(grid)
    
    # Select solving method
    solvers = {
        'sat': solve_by_sat,
        'backtracking': backtracking_cnf,
        'brute': solve_by_brute_force
    }
    
    solver = solvers[args.method]
    result = solver(cnf)
    result_grid = get_grid_result(grid, result) if result else grid
    
    # Print to console
    print(f"\nSolving with {args.method} method:")
    print("Original Grid:")
    for row in grid:
        print(' '.join(row))
    print("\nResult:")
    if result:
        print("SAT (satisfiable)")
        print(f"Model: {result}")
        print("Result Grid:")
        for row in result_grid:
            print(' '.join(row))
    else:
        print("UNSAT (unsatisfiable)")
    
    # Save to file
    save_results(args.output, args.method, result_grid, result, grid)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()