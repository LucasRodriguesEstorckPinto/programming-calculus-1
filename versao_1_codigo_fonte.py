import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Lendo a função do usuário
func_str = input("Digite a função: ")
x = sympy.Symbol('x')
func = sympy.sympify(func_str)

# Calculando a derivada
deriv = sympy.diff(func, x)
print(f'A derivada é: {deriv}')

# Obtendo o ponto para calcular a reta tangente
x0 = float(input("Digite o valor de x para calcular a reta tangente: "))

# Convertendo a função e a derivada em funções lambda para cálculos numéricos
func_lambda = sympy.lambdify(x, func)
deriv_lambda = sympy.lambdify(x, deriv)

# Calculando a inclinação da reta tangente
coef_angular = deriv_lambda(x0)

# Calculando a equação da reta tangente
reta_tangente = lambda x: coef_angular * (x - x0) + func_lambda(x0)

# Mostrando a equação da reta tangente na tela
print(f"A equação da reta tangente é: y = {coef_angular:.2f} * (x - {x0}) + {func_lambda(x0):.2f}")

# Plotando a função e a reta tangente
x_vals = np.linspace(x0 - 10, x0 + 10, 100)
y_vals = func_lambda(x_vals)
y_tangente = reta_tangente(x_vals)

plt.plot(x_vals, y_vals, label="Função")
plt.plot(x_vals, y_tangente, label="Reta tangente")

# Adicionando a linha dos eixos x e y
plt.axhline(0, color='red', linestyle='-', linewidth=0.5)
plt.axvline(0, color='red', linestyle='-', linewidth=0.5)

# Ajustando a escala do gráfico para focar no ponto escolhido
plt.xlim(x0 - 10, x0 + 10)
plt.ylim(func_lambda(x0) - 10, func_lambda(x0) + 10)

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico da função e da reta tangente')
plt.grid(True)
plt.show()
