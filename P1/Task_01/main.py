import cholesky_LU
import iterative_met
import numpy as np

def read_matrix(b):
    A = []
    f = open("Matriz_A.dat", "r")
    for line in f:
        linha = (line.split())
        A.append(linha)

    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] = float(A[i][j])
    
    A = np.array(A)

    B = []
    f1 = open("Vetor_B_0"+ b +".dat", "r")
    for line in f1:
        linha = line.split()
        B.append(linha)

    for i in range(len(B)):
        B[i][0] = float(B[i][0])
    
    B = np.array(B)

    return A, B

def valida_entrada_B(b):
    if b.isnumeric():
        if (int(b) != 1 and int(b) != 2 and int(b) != 3):
            raise ValueError("Selecione um valor entre 1, 2 ou 3!")
    else:
        raise ValueError("Selecione um valor entre 1, 2 ou 3!")

def valida_entrada_tipo(tipo):
    if tipo.isnumeric():
        if (int(tipo) != 1 and int(tipo) != 2 and int(tipo) != 3 and int(tipo) != 4):
            raise ValueError("Selecione um valor entre 1, 2, 3 ou 4!")
    else:
        raise ValueError("Selecione um valor entre 1, 2, 3 ou 4!")


if __name__ == "__main__":
    b = input("Selecione o Vetor B para o sistema de equações:\n" + "- 1: para o Vetor B_01\n" + "- 2: para o Vetor B_02\n" + "- 3: para o Vetor B_03\n" )
    
    valida_entrada_B(b)
    
    A, B = read_matrix(b)

    tipo = input("Selecione o método para a solução do sistema: \n" + "- 1: para decomposição LU\n" + "- 2: para decomposição cholesky\n" + "- 3: para método iterativo de Jacobi\n" + "- 4: para o método iterativo de Gauss-Seidel\n")
    
    valida_entrada_tipo(tipo)
    
    if tipo == "1":
        x = cholesky_LU.solve_LU_cholesky(A, B, int(tipo))
    
    elif tipo == "2":
        x = cholesky_LU.solve_LU_cholesky(A, B, int(tipo))
    
    elif tipo == "3":
        x0 = np.ones(len(B))
        x = iterative_met.jacobi(A, B, x0)

    else:
        x0 = np.ones(len(B))
        x = iterative_met.gauss_seidel(A, B, x0)
    

    x = np.array(x)
    print("Solução: \n" , x)






