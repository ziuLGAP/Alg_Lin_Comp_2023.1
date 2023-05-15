import numpy as np
import math


def power_method(A, tol =(10**-10), max_iter=1000):
    n = len(A)
    x0 = np.ones(n)
    x = 1
    for k in range(max_iter):
        x_old = x
        Ax = np.dot(A, x0)
        x = max(Ax)
        x0 = np.dot((1/x), Ax)
        err = abs(x - x_old) / abs(x)
        if err < tol:
            print(f"O metódo da pontência achou o autovalor em {k} iterações.")
            return x, x0
    print(f"O metódo da pontência não encontrou o autovalor em {max_iter} iterações. Retornando o valor encontrado.")
    return x, x0

def max_offdiag(A):
    """Acha o valor de maior módulo fora da diagonal principal"""
    n = len(A)
    max_val = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_i, max_j = i, j
    return  max_i, max_j


def Phi_Value(A, i, j):
    """Retorna o Valor de Phi"""
    Phi = 0
    if A[i][i] != A[j][j]:
        Phi = 0.5*math.atan((2*A[i][j])/(A[i][i]-A[j][j]))
    else:
        Phi = math.pi/4
    
    return Phi

def tol_check(A, tol):
    """checa a se os valores fora da diagonal principal são menores que a tolerância"""
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) < tol:
                pass
            else:
                return False
    return True

def get_autovalores(A):
    """retorna uma lista com os valores da diagonal principal de uma matriz"""
    n = len(A)
    autovalues = []
    for i in range(n):
        autovalues.append(A[i][i])

    return autovalues

def jacobi(A, tol=(10**-10), max_iter=1000):
        n = len(A)
        x = np.identity(n)

        for k in range(max_iter):
            P = np.identity(n)
            max_i, max_j = max_offdiag(A)
            Phi = Phi_Value(A, max_i, max_j)
            P[max_i][max_i] = math.cos(Phi)
            P[max_j][max_j] = math.cos(Phi)
            P[max_i][max_j] = -math.sin(Phi)
            P[max_j][max_i] = math.sin(Phi)
            aux = np.dot(np.transpose(P), A)
            A = np.dot(aux, P)
            x = np.dot(x, P)

            if tol_check(A, tol):
                autovalores = get_autovalores(A)
                print(f"O metódo de Jacobi achou o autovalor em {k} iterações.")
                return autovalores, x
        
        print(f"O metódo de Jacobi não encontrou o autovalor em {max_iter} iterações. Retornando o valor encontrado.")
        autovalores = get_autovalores(A)
        return autovalores, x

A = np.array([[1,0.2,0],[0.2,1,0.5],[0,0.5,1]])

x, x0 = jacobi(A)

print(x)
print(x0)


