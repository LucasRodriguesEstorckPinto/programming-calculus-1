import sympy as sp

# Define a variável simbólica x
x = sp.symbols('x')

# Define a função f(x) (altere esta função conforme necessário)
f_x_str = input("Digite a função: ").strip()
f_x = sp.sympify(f_x_str)


# Define o valor de dx
dx = sp.symbols('dx')

# Calcule f(x+dx)
f_x_plus_dx = f_x.subs(x, x + dx)

# Calcule a diferença f(x+h) - f(x)
difference = f_x_plus_dx - f_x

# Calcule a derivada f'(x) usando a definição
derivative = difference / dx

# Simplifique a expressão da derivada
derivative_simplified = sp.simplify(derivative)

# Limite conforme dx se aproxima de 0
derivative_limit = sp.limit(derivative_simplified, dx, 0)

# Imprima o cálculo passo a passo
print("_" *100)
print("Função original: f(x) =", f_x)

print("\nPasso 1: Calculando f(x+dx)")

print("f(x+dx) =", f_x_plus_dx)

print("\nPasso 2: Calculando a diferença f(x+dx) - f(x)")

print("f(x+dx) - f(x) =", difference)

print("\nPasso 3: Calculando a derivada f'(x) usando a definição")

print("f'(x) =", derivative)

print("\nPasso 4: Simplificando a expressão da derivada")

print("f'(x) =", derivative_simplified)

print("\nPasso 5: Calculando o limite conforme dx se aproxima de 0")

print("f'(x) =", derivative_limit)
