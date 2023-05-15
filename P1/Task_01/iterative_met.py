import numpy as np
import math

def jacobi(A, b, x, tol=(10**-10), max_iter=1000):
    n = len(A)

    for k in range(max_iter):
        x_new = np.zeros(n) #vetor x(n+1)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i] #somatório usado no slide 4
        s1 = sum(((x_new[i] - x[i])**2) for i in range(n))
        s2 = sum(((x_new[i])**2) for i in range(n))
        err = math.sqrt(s1) / math.sqrt(s2) #norma euclidiana
        x = x_new
        if err < tol:
            print(f"O método de Jacobi convergiu após {k} iterações")
            return x
    print(f"O método de Jacobi não convergiu após {max_iter} iterações. Retornando os valores encontrados depois de {max_iter} iterações")
    return x
def gauss_seidel(A, b, x, tol=(10**-10), max_iter=1000):
    n = len(A)

    for k in range(max_iter):
        x_new = np.zeros(n)
        for j in range(n):
            s1 = np.dot(A[j, :j], x_new[:j]) #valores atualizados
            s2 = np.dot(A[j, j + 1:], x[j + 1:]) #valores ainda não atualizados
            x_new[j] = (b[j] - s1 - s2) / A[j, j]
        if np.allclose(x, x_new, rtol=tol): #Manhattan norm
            print(f"O método de Jacobi convergiu após {k} iterações")
            return x_new
        x = x_new
    print(f"O método de Gauss-Seidel não convergiu após {max_iter} iterações. Retornando os valores encontrados depois de {max_iter} iterações")
    return x
