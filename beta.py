import sympy
import matplotlib.pyplot as plt

# Lendo a função do usuário
func_str = input("Digite a função: ")
x = sympy.Symbol('x')
func = sympy.sympify(func_str)

# Calculando a derivada
deriv = sympy.diff(func, x)
print(f'A derivada é: {deriv}')

# Obtendo o coeficiente angular da reta tangente
x0 = float(input("Digite o valor de x para calcular a reta tangente: "))
y0 = func.subs(x, x0)
deriv_func = deriv.subs(x, x0)
coef_angular = deriv_func.evalf()

# Obtendo a equação da reta tangente
reta_tangente = coef_angular * (x - x0) + y0
print(f"A equação da reta tangente é: {reta_tangente}")

# Plotando a função e a reta tangente
x_vals = range(-10, 11)
y_vals = [func.subs(x, i) for i in x_vals]
plt.plot(x_vals, y_vals, label="Função")

y_tangente = [reta_tangente.subs(x, i) for i in x_vals]
plt.plot(x_vals, y_tangente, label="Reta tangente")

plt.legend()
plt.show()
