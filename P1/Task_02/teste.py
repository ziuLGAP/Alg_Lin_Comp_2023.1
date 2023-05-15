import numpy as np

def jacobi_method(A, eps=1e-10, max_iterations=1000):
    n = A.shape[0]
    D = np.diag(A) # Vetor de elementos diagonais
    R = A - np.diag(D) # Matriz de elementos não diagonais
    
    # Matriz de autovetores (inicialmente é a matriz identidade)
    Q = np.eye(n)
    
    # Iterações
    for i in range(max_iterations):
        # Encontra o maior elemento fora da diagonal
        off_diag = np.abs(R - np.diag(np.diag(R)))
        idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        row, col = idx
        
        # Se o maior elemento fora da diagonal é menor que o limite de tolerância,
        # então a matriz já está aproximadamente diagonal
        if off_diag[row, col] < eps:
            break
        
        # Calcula o ângulo de rotação
        theta = 0.5 * np.arctan2(2*R[row, col], D[col]-D[row])
        
        # Matriz de rotação
        G = np.eye(n)
        G[row, row] = np.cos(theta)
        G[col, col] = np.cos(theta)
        G[row, col] = np.sin(theta)
        G[col, row] = -np.sin(theta)
        
        # Atualiza a matriz de autovetores e a matriz A
        Q = Q @ G
        A = Q.T @ R @ Q
        D = np.diag(A)
        R = A - np.diag(D)
        
    # Ordena os autovalores e autovetores em ordem decrescente
    idx = np.argsort(D)[::-1]
    D = D[idx]
    Q = Q[:, idx]
    
    return D, Q

A = np.array([[1,0.2,0],[0.2,1,0.5],[0,0.5,1]])
D, Q = jacobi_method(A)
print("Autovalores:", D)
print("Autovetores:\n", Q)