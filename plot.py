import matplotlib.pyplot as plt
import numpy as np

# recebe a função do usuário
funcao = input("Digite a função que deseja plotar: ")


# define a função a ser plotada

def f(x):
    return eval(funcao)


# cria o vetor x
x = np.linspace(-5, 5, 100)

# calcula os valores de y
y = f(x)

# plota o gráfico
plt.plot(x, y)
plt.show()
