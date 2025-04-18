# CLI Usage Guide for Trap Grid Puzzle Solver

## Overview
This program solves the trap grid puzzle using a SAT solver (Glucose3). Users provide an input grid via a text file, and the program determines the placement of traps ('T') and empty spaces ('G') to satisfy the grid's conditions. Results are displayed on the console and saved to an output file.

## Requirements
- **Python 3.x**: Installed on your machine.
- **PySAT Library**: Install with:
  ```bash
  pip install python-sat
  ```
- **Input File**: A text file containing the puzzle grid.

## Installation
1. Download the program code (e.g., `trap_solver.py`).
2. Ensure the PySAT library is installed.
3. Prepare an input file as described below.

## Usage
Run the program via the command line with required and optional arguments. Basic syntax:
```bash
python trap_solver.py <input_file> [--method METHOD] [--output OUTPUT_FILE]
```

### Arguments
1. **`<input_file>`** (Required)
   - Path to the text file containing the input grid.
   - Example: `input.txt`

2. **`--method`** (Optional)
   - Solving method to use.
   - Currently supported: `sat` (SAT solver Glucose3).
   - Default: `sat`
   - Example: `--method sat`

3. **`--output`** (Optional)
   - Path to the output result file.
   - Default: `result.txt`
   - Example: `--output my_result.txt`

### Input File Format
- Plain text file (`.txt`).
- Each line represents a row in the grid.
- Cells in each row are separated by commas (`,`).
- Cell meanings:
  - Numbers (0, 1, 2, ...): Number of traps required in adjacent cells (8 directions).
  - `_`: Empty cell, can be assigned a trap ('T') or empty ('G').
- Example `input.txt`:
  ```
  1, _, _
  _, 1, _
  ```

### Example Commands
1. **Basic Usage**:
   ```bash
   python trap_solver.py input.txt
   ```
   - Uses SAT solver.
   - Results saved to `result.txt`.

2. **Custom Output File**:
   ```bash
   python trap_solver.py input.txt --output solution.txt
   ```
   - Results saved to `solution.txt`.

3. **Specify Method (currently only SAT)**:
   ```bash
   python trap_solver.py input.txt --method sat --output result.txt
   ```

## Output
### Console Output
The program prints:
1. **Original Grid**:
   - Displays the grid read from the input file.
2. **Result**:
   - `SAT (satisfiable)`: If a solution exists, includes:
     - `Model`: List of variables (positive: trap 'T', negative: empty 'G').
     - `Result Grid`: Grid with empty cells filled as 'T' or 'G'.
   - `UNSAT (invalid grid - more traps required than empty cells)`: If total required traps exceed empty cells.
   - `UNSAT (no solution exists)`: If no trap placement satisfies the conditions.
3. **File Save Notification**:
   - Location of the output file.

Example output:
```
Solving with sat method:
Original Grid:
1 _ _
_ 1 _

Result:
SAT (satisfiable)
Model: [2]
Result Grid:
1 T _
_ 1 _

Results saved to result.txt
```

### Output File
The output file (e.g., `result.txt`) contains similar information:
```
Solving method: sat

Original Grid:
1 _ _
_ 1 _

Result:
SAT (satisfiable)
Model: [2]

Result Grid:
1 T _
_ 1 _
```

## Debugging and Error Handling
- **File Not Found**: If `<input_file>` doesn’t exist, the program exits with:
  ```
  Error: File input.txt not found
  ```
- **Incorrect Format**: Ensure the input file uses commas (`,`) as separators, no extra spaces.
- **Invalid Grid Check**: If total required traps exceed empty cells, it reports UNSAT with the reason.

## Tips
1. **Verify Input File**: Check the format using a text editor (e.g., Notepad, VSCode).
2. **Testing**: Create multiple input files with different cases (solvable, unsolvable, invalid) to test behavior.
3. **Output Management**: Use unique `--output` filenames to avoid overwriting results.

## Limitations
- Only supports the SAT solver method. Other methods (`backtracking`, `brute`) are not included in this version.
- Input format is strict (comma-separated values only).

## Support
If you encounter issues:
1. Verify the input file format.
2. Run with debug output (add print statements in the code if needed).
3. Contact the author with: input file content, command used, and observed output.