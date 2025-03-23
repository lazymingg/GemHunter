from itertools import product

n = 3

binary_combinations = product([0, 1], repeat=n)

for binary in binary_combinations:
    print(binary)