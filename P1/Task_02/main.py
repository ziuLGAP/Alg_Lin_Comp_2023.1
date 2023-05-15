import numpy as np
import autos_solver as As


def read_matrix():
    A = []
    f = open("Matriz_A.dat", "r")
    for line in f:
        linha = (line.split())
        A.append(linha)

    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] = float(A[i][j])
        
    A = np.array(A)
    return A

def valida_entrada_tipo(tipo):
    if tipo.isnumeric():
        if (int(tipo) != 1 and int(tipo) != 2 ):
            raise ValueError("Selecione um valor entre 1 ou 2!")
    else:
        raise ValueError("Selecione um valor entre 1 ou 2")

if __name__ == "__main__":
    A = read_matrix()
    tipo = input("Selecione o método para a solução do sistema: \n" + "- 1: para Power Method\n" + "- 2: para Metódo de Jacobi\n")
    valida_entrada_tipo(tipo)

    if tipo == "1":
        x, x0 = As.power_method(A)
    elif tipo == "2":
        x, x0 = As.jacobi(A)
    
    print("Autovalor(s): \n", x, "\n")
    print("Autovetor(s): \n", x0, "\n")
    