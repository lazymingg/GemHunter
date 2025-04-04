import time
from typing import List, Tuple, Set
from itertools import combinations
from pysat.solvers import Solver
from copy import deepcopy
import argparse
import random
from collections import Counter



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

def is_conflict(cnf: List[List[int]], model: List[int]) -> bool:
    for clause in cnf:
        satisfied = False
        for var in clause:
            if (var > 0 and var in model) or (var < 0 and -var not in model):
                satisfied = True
                break
        if not satisfied and all(abs(v) in [abs(m) for m in model] for v in clause):
            return True
    return False

def checking_cnf(cnf: List[List[int]], model: List[int]) -> bool:
    return all(checking_clause(clause, model) for clause in cnf)

# unit clause heuristic and unit propagation
def unit_propagation(cnf: List[List[int]], model: List[int]) -> Tuple[List[List[int]], List[int]]:
    
    changed = True
    while changed:
        changed = False
        unit_clauses = [clause[0] for clause in cnf if len(clause) == 1] #fiding unit clauses

        for unit in unit_clauses:
            #if unit is already in model
            if unit in model or -unit in model:
                continue  
            
            #assigning the unit clause
            model.append(unit)  
            cnf = [clause for clause in cnf if unit not in clause]

            # delete -unit from remaining clauses
            for clause in cnf:
                if -unit in clause:
                    clause.remove(-unit)

            #change flag to True
            changed = True  

    return cnf, model

def pure_literal_elimination(cnf: List[List[int]], model: List[int]) -> Tuple[List[List[int]], List[int]]:
    all_vars = set(abs(var) for clause in cnf for var in clause)
    pure_literals = set()
    all_literals = {lit for clause in cnf for lit in clause}

    for var in all_vars:
        if var in all_literals and -var not in all_literals:
            pure_literals.add(var)
        elif -var in all_literals and var not in all_literals:
            pure_literals.add(-var)

    for pure in pure_literals:
        model.append(pure) 
        cnf = [clause for clause in cnf if pure not in clause]

    return cnf, model

# def choose_variable(cnf: List[List[int]]) -> int:
#     return abs(cnf[0][0])
def choose_variable(cnf):
    flat = [abs(var) for clause in cnf for var in clause]
    counter = Counter(flat)
    return counter.most_common(1)[0][0]

def dpll(cnf: List[List[int]], model: List[int]) -> List[int]:
    cnf, model = unit_propagation(cnf, model) 
    cnf, model = pure_literal_elimination(cnf, model) 

    #if ALL cnf is empty mean all clauses are satisfied
    if not cnf:
        return model

    #if clause is empty mean there is conflict
    if any(len(clause) == 0 for clause in cnf):
        return []

    #choose variable
    var = choose_variable(cnf)

    
    new_model = model + [var]
    new_cnf = [clause for clause in cnf if var not in clause]
    new_cnf = [[x for x in clause if x != -var] for clause in new_cnf]
    result = dpll(new_cnf, new_model)
    if result:
        return result

    new_model = model + [-var]
    new_cnf = [clause for clause in cnf if -var not in clause]
    new_cnf = [[x for x in clause if x != var] for clause in new_cnf]
    return dpll(new_cnf, new_model)


def solve_by_dpll(cnf: List[List[int]]) -> List[int]:
    all_vars = set(abs(var) for clause in cnf for var in clause)
    
    max_var = max(all_vars) if all_vars else 0

    assignment = dpll(cnf, [])

    if assignment is None:
        return []

    assigned_vars = set(abs(var) for var in assignment)

    for var in range(1, max_var + 1):
        if var not in assigned_vars:
            assignment.append(-var)

    return assignment


def solve_by_brute_force(cnf: List[List[int]]) -> List[int]:
    all_vars = set()
    for clause in cnf:
        for var in clause:
            all_vars.add(abs(var))
    
    sorted_vars = sorted(all_vars)
    num_vars = len(sorted_vars)
    
    total_assignments = 1 << num_vars
    for assignment in range(total_assignments):
        model = []
        for i in range(num_vars):
            if (assignment >> i) & 1: 
                model.append(sorted_vars[i])  
            else:  
                model.append(-sorted_vars[i])  
        
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

#made by chatgpt
def is_valid_filled_grid(grid: List[List[str]]) -> bool:
    n_rows, n_cols = len(grid), len(grid[0])

    def count_traps_around(i: int, j: int) -> int:

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        trap_count = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n_rows and 0 <= nj < n_cols:
                if grid[ni][nj] == 'T':
                    trap_count += 1
        return trap_count

    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j].isdigit():
                expected_traps = int(grid[i][j])
                actual_traps = count_traps_around(i, j)
                
                if actual_traps != expected_traps:
                    print(f"errors at cell ({i}, {j}): "
                          f"Need {expected_traps} trap but found {actual_traps}")
                    return False
            elif grid[i][j] not in ['_', 'T', 'G']:
                print(f"unvalid value in cell ({i}, {j}): {grid[i][j]}")
                return False

    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         if grid[i][j] == '_':
    #             print(f" cell ({i}, {j}) not filled!")
    #             return False

    print("All cells are filled correctly!")
    return True

def generate_random_grid(rows: int, cols: int) -> List[List[str]]:
    grid = [['_' for _ in range(cols)] for _ in range(rows)]
    
    num_cells = rows * cols
    #place traps (T and G)
    trap_positions = random.sample(range(num_cells), num_cells)
    for pos in trap_positions:
        x = pos // cols
        y = pos % cols
        grid[x][y] = random.choice(['G', 'T', 'N', 'N'])
    
    # print(grid)

    # Update numbers based on traps around
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                  (1, 1), (-1, -1), (1, -1), (-1, 1)]
    

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'N':
                grid[i][j] = '0'
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if grid[ni][nj] == 'T':
                            grid[i][j] = str(int(grid[i][j]) + 1)

    #remove all t g
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'T' or grid[i][j] == 'G' or grid[i][j] == '0':
                grid[i][j] = '_'

    return grid

def run_experiment():
    sizes = [(5, 5), (11, 11), (20, 20), (30, 30)]
    trials = 10
    method = solve_by_sat
    # method = solve_by_dpll
    # method = solve_by_brute_force
    output_file = "experiment_results.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        for rows, cols in sizes:
            total_time = 0.0
            success_count = 0

            header = f"\n=== Running {trials} trials for grid size {rows}x{cols} ===\nUsing {method.__name__} method\n"
            print(header)
            f.write(header)

            for i in range(trials):
                grid = generate_random_grid(rows, cols)
                cnf = generate_cnf(grid)

                grid_str = "\n".join(', '.join(row) for row in grid)
                print("Solving grid:")
                print(grid_str)
                f.write(f"\nGrid {i+1}:\n{grid_str}\n")

                start_time = time.perf_counter()
                result = method(cnf)
                solve_time = time.perf_counter() - start_time
                total_time += solve_time

                result_grid = get_grid_result(grid, result) if result else grid
                is_valid = is_valid_filled_grid(result_grid)
                if is_valid:
                    success_count += 1

                trial_log = f"Trial {i+1}: {'SAT' if result else 'UNSAT'} | Time: {solve_time:.5f}s | Valid: {is_valid}\n"
                print(trial_log)
                f.write(trial_log)

            avg_time = total_time / trials
            summary = f"\n✅ Average solving time for {rows}x{cols}: {avg_time:.5f} seconds\n Valid solutions: {success_count}/{trials}\n"
            print(summary)
            f.write(summary)

#print results to screen
def print_results(method, cnf_time, solve_time, grid, result, result_grid, output_path):
    print(f"\nSolving with {method} method:")
    print(f"Time to generate CNF: {cnf_time:.10f} seconds")
    print(f"Time to solve CNF: {solve_time:.10f} seconds")
    print("\nOriginal Grid:")
    for row in grid:
        print(','.join(row))
    print("\nResult:")
    if result:
        print("SAT (satisfiable)")
        print(f"Model: {result}")
        print("Result Grid:")
        for row in result_grid:
            print(' '.join(row))
    else:
        print("UNSAT (unsatisfiable)")
    print(f"\nResults and performance metrics saved to {output_path}")

#write results to file
def write_results_to_file(output_path, method, cnf_time, solve_time, grid, result, result_grid):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Solving method: {method}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"Time to generate CNF: {cnf_time:.10f} seconds\n")
        f.write(f"Time to solve CNF: {solve_time:.10f} seconds\n\n")
        f.write("Original Grid:\n")
        for row in grid:
            f.write(','.join(row) + '\n')
        f.write("\nCNF Result:\n")
        if result:
            f.write("SAT (satisfiable)\n")
            f.write(f"Model: {result}\n\n")
            f.write("Result Grid:\n")
            for row in result_grid:
                f.write(','.join(row) + '\n')
        else:
            f.write("UNSAT (unsatisfiable)\n")


def main():
    parser = argparse.ArgumentParser(description="Solve grid puzzle with different methods")
    parser.add_argument('input_file', help="Input file path")
    parser.add_argument('--method', choices=['sat', 'backtracking', 'brute'], 
                       default='sat', help="Solving method (default: sat)")
    parser.add_argument('--output', default='result.txt', 
                       help="Output file path (default: result.txt)")
    parser.add_argument('--random', action='store_true', help="Generate random grid")

    args = parser.parse_args()

    grid = read_input(args.input_file)

    start_time = time.perf_counter()
    cnf = generate_cnf(grid)
    cnf_time = time.perf_counter() - start_time

    solvers = {
        'sat': solve_by_sat,
        'backtracking': solve_by_dpll,
        'brute': solve_by_brute_force
    }

    solver = solvers[args.method]

    start_time = time.perf_counter()
    result = solver(cnf)
    solve_time = time.perf_counter() - start_time

    result_grid = get_grid_result(grid, result) if result else grid

    if is_valid_filled_grid(result_grid):
        print("valid grid!!")

    print_results(args.method, cnf_time, solve_time, grid, result, result_grid, args.output)
    write_results_to_file(args.output, args.method, cnf_time, solve_time, grid, result, result_grid)
    
    
    #comment all code and uncomment this to run experiment
    # run_experiment()

if __name__ == "__main__":
    main()