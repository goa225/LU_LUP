# Comparison of LU and LUP Decomposition Algorithms to Solve Linear Systems

# Code adapted from: 
# https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
# https://stackoverflow.com/questions/29358167/matlab-lu-decomposition-partial-pivoting (originally in Matlab)
# with minor modifications to the LUP algorithm.

# Amanda Gin
# Colorado State University, Fort Collins, CO
# MATH 450 - Introduction to Numerical Analysis I, Fall 2023
# Final Project

# Import necessary modules
import numpy as np


# # LU Algorithm 
# Create LU decomposition (A = LU)
def lu(A):
    #Get the number of rows
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    #Loop over rows
    for i in range(n):
        # Eliminate entries below i with row operations on U 
        # Reverse the row operations to manipulate L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
    return L, U

# Define forward substitution to solve Ly = b
def forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]
    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);
    #Here we perform the forward-substitution.  
    #Initializing  with the first row.
    y[0] = b[0] / L[0, 0]
    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the 
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
    return y

# Define forward substitution to solve Ux =y
def back_substitution(U, y):
    # Number of rows
    n = U.shape[0]
    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double);
    # Here we perform the back-substitution.  
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]
    # Looping over rows in reverse (from the bottom up), 
    # starting with the second to last row, because the 
    # last row solve was completed in the last step.
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]
    return x

# Solve the linear system using LU decomposition
def lu_solve(A, b):
    L, U = lu(A)
    y = forward_substitution(L, b)
    return back_substitution(U, y)


# # LUP Decomposition
# Create LUP decomposition 
def lup(A):
    # Get the number of rows
    n = A.shape[0]
    # Allocate space for P, L, and U
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)
    for k in range(n):
        # find the entry in the left column with the largest abs value (pivot)
        r = np.argmax(np.abs(U[k:, k])) + k
        # Permute rows
        U[[k, r], :] = U[[r, k], :]
        P[[k, r], :] = P[[r, k], :]
        L[[k, r], :] = L[[r, k], :]
        # from the pivot down divide by the pivot
        L[k:n, k] = U[k:n, k] / U[k, k]
        U[k+1:n, :] = U[k+1:n, :] - L[k+1:n, k][:, np.newaxis] * U[k, :]
    return P, L, U

# Solve the linear system using LUP decomposition
def lup_solve(A, b):
    P, L, U = lup(A)
    y = forward_substitution(L, np.dot(P, b))
    return back_substitution(U, y)


# # Input matrix A and vector b
# Function to input a matrix from the user
def input_matrix():
    size = int(input("Enter the number of rows or columns of A (must be square):"))
    print("Enter the matrix entries row-wise:")
    matrix = []
    for i in range(size):
        row = [float(x) for x in input().split()]
        matrix.append(row)
    return np.array(matrix)

# Function to input a vector from the user
def input_vector():
    print("Enter the vector entries of b:")
    vector = [float(x) for x in input().split()]
    return np.array(vector)

# Ask the user for input
A = input_matrix()
b = input_vector()

# Print the input matrix and vector
print("\nInput Matrix A:")
print(A)
print("\nInput Vector b:")
print(b)

# LU Decomposition and Solution
print("\nLU Decomposition Method:")
x_lu = lu_solve(A, b)
print("Solution:", x_lu)
print("NumPy Solve Result:", np.linalg.solve(A, b))
L, U = lu(A)
print("\nLU Decomposition:")
print("L Matrix:")
print(L)
print("\nU Matrix:")
print(U)

# LUP Decomposition and Solution
print("\nLUP Decomposition Method:")
x_lup = lup_solve(A, b)
print("Solution:", x_lup)
print("NumPy Solve Result:", np.linalg.solve(A, b))
P, L, U = lup(A)
print("\nLUP Decomposition:")
print("P Matrix:")
print(P)
print("\nL Matrix:")
print(L)
print("\nU Matrix:")
print(U)
