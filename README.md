# CUDA Jacobi Implementation

## Overview
This is my course assignment of COL380: Introduction to Parallel Programming and Distributed Computing (Sem-II, 2018-19) (Instructor: Prof. Subodh V. Sharma) at Indian Institute of Technology (IIT), Delhi. The assignment asked for an implementation of Principal Component Analysis (PCA) through Singular Value Decomposition (SVD) using a parallel Jacobi eigenvalue algorithm. The problem statement could be found in lab3.pdf.

It contains a highly optimised parallel GPU implementation of Jacobi method to calculate eigenvalues and eigenvectors of a symmetric matrix. This Jacobi function could be found in lab3_cuda.cu and could be used anywhere else as well.

**Primary Reference: Novel GPU Implementation of Jacobi Algorithm for Karhunen-Loeve Transform of Dense Matrices (Mustafa U. Tamn, Onur Yilmaz, and Ali N. Akansu) [IEEE 2012]**

## Extent of parallelisation and Input limitation
This code works on Input matrices of dimensions M (#samples) x N (#features) and uses N/2 blocks of GPU with N threads in each block. So, N could not exceed maximum number of threads in a block of GPU.

## Directories and files
- `testcase/`: contains python script `gen_testcase.py` for sample testcase generation  
- `lab3_io.h` and `lab3_io.cu`: functions to read matrix from file and check the correctness of the result. This file also contains a dummy function to check the dimensions of the returned matrix
- `main_cuda.cu`: function `main()`  
- `lab3_cuda.h`: header file for the functions to be implemented  
- `lab3_cuda.cu`: implement the function in this file  
Refer to respective files for furthur details.  

## Building and Executing
```
nvcc -lm -std=c++11 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
```
#### Command Line Arguments
The program takes two command line arguments:
- arg1: input filename (consist M, N and D)  
- arg2: retention (percentage of information to be retained by PCA) 

Note that the retention percentage is integer.  Please refer to `main_cuda.cu` for more details.  
To run the program:
```
./pca <input filename> <retention>
```
Example:
```
./pca testcase/testcase_1000_1000 90
```

## Generating testcases
Script `gen_testcase.py` generates testcases as per the parameters and output the generated testcase in file `testcase_<M>_<N>` in the desired format.
```
python3 gen_testcase.py M N
```

## Input-Output Specifications
#### Input dataset specifications
- M : number of rows (samples) in input matrix D
- N : number of columns (features) in input matrix D
- D : input matrix, #elements in D is (M * N)

The first line of the input file contains `M` followed by `N`. The second line contains elements of matrix `D`. All the values in one line are space separated.

#### Input file specifications
- First line contain two integer (`M` followed by `N`)
- It then contains `M*N` doubles in matrix `D`.

A sample input file could be found in testcases folder.

#### Output file specifications
- First line contain value of `K`
- Next M lines contains K doubles space seperated representing the modified D matrix. 

