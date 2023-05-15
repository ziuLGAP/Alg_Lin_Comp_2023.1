"""
Task 1 - Algrebra Linear Computacional
Luiz Guilherme de Andrade Pires - ECI
DRE -> 121070338
"""
import numpy as np
import math as math

def lu_decomp(A):
    """Na decomposição LU, para utilizar melhor a memória do computador, a matriz U é a matriz com somente zeros"""
    n = len(A)
    L = np.identity(n) #iniciamos L como uma matriz identidade
    U = np.zeros((n, n)) #iniciamos U como uma matriz zerada
    for j in range(n):
        U[0][j] = A[0][j] #A primeira linha da matriz U é sempre igual a primeira linha da matriz original
    for i in range(1, n):
        L[i][0] = A[i][0] / U[0][0]
    for i in range(1, n):
        for j in range(1, n):
            if j >= i: #acima da diagonal principal
                U[i][j] = A[i][j] - sum([L[i][k] * U[k][j] for k in range(i)]) #atualizamos o valor de U
            else:
                L[i][j] = (A[i][j] - sum([L[i][k] * U[k][j] for k in range(j)])) / U[j][j]
    

    return L, U

def solve_LU_cholesky(A, B, tipo):
    n = len(A)
    x = [0.0] * n
    
    #inicio tanto o vetor x  como sendo vetor nulo, e utilizo esse mesmo vetor, inicialmente como o y do Ly = B e depois como o x de Ux = y, para melhor utilizar a memória
    if tipo == 1:
        L, U = lu_decomp(A)
        
        # Resolvendo Ly = B
        for i in range(n):
            temp_sum = sum([L[i][j] * x[j] for j in range(i)])
            x[i] = B[i] - temp_sum / L[i][i] # essa conta foi feita com base na fórmula mostrada no slide 3
    
        #print("Y:\n", x) utilizado para verificar se o vetor y está correto

    # Resolvendo Ux = y
        for i in reversed(range(n)):
            temp_sum = sum([U[i][j] * x[j] for j in range(i+1, n)])
            x[i] = (x[i] - temp_sum) / U[i][i] # essa conta foi feita com base na fórmula mostrada no slide 3
    
    elif tipo == 2:
        L = cholesky_decomp(A)
        
        # Resolvendo Ly = b
        for i in range(n):
            temp_sum = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (B[i] - temp_sum) / L[i][i]

        #print("Y:\n", x) utilizado para verificar se o vetor y está correto

        # Resolvendo L^Tx = y
        for i in reversed(range(n)):
            temp_sum = sum(L[j][i] * x[j] for j in range(i+1, n))
            x[i] = (x[i] - temp_sum) / L[i][i]
    else:
        raise ValueError("O tipo de operação não pode ser escolhido, por favor escolha entre os valores:\n"
                         + "1: Decomposição LU;\n" + "2: Decomposição Cholesky;\n")
    
    return x

def cholesky_decomp(A):
    n = len(A)
    L = np.zeros((n,n)) #iniciamos L como uma matriz zerada

    for i in range(n):
        for j in range(i+1):
            temp_sum = sum(L[i][k] * L[j][k] for k in range(j)) #somatório que aparece para calcular o valor de L[i][j]
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - temp_sum) #se for um valor da diagonal principal
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - temp_sum))
    return L
