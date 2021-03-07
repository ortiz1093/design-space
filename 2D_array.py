import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(arr):
    max_sum = 0
    
    for i in range(4):
        for j in range(4):
            print(f'{arr[i][j]}\t{arr[i][j+1]}\t{arr[i][j+2]}')
            print(f' \t{arr[i+1][j+1]}\t ')
            print(f'{arr[i+2][j]}\t{arr[i+2][j+1]}\t{arr[i+2][j+2]}')
            print()
            hg_sum = sum(arr[i][j:j+3] + [arr[i+1][j+1]] + arr[i+2][j:j+3])
            max_sum = hg_sum if hg_sum > max_sum else max_sum
    
    return max_sum

if __name__ == '__main__':
    arr = [[1, 1, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0],
           [0, 0, 2, 4, 4, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 1, 2, 4, 0]]

    result = hourglassSum(arr)

    print(result)