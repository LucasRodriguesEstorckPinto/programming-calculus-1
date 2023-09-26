import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
matplotlib.use('TkAgg')


def calcular_tangente():
    # Lendo a função do usuário
    func_str = entry_func.get()
    x = sympy.Symbol('x')
    func = sympy.sympify(func_str)

    try:
        # Calculando a derivada
        deriv = sympy.diff(func, x)
        deriv_lambda = sympy.lambdify(x, deriv)

        # Obtendo o ponto para calcular a reta tangente
        x0 = float(entry_x.get())

        # Calculando a inclinação da reta tangente
        coef_angular = deriv_lambda(x0)

        # Convertendo a função em função lambda para cálculos numéricos
        func_lambda = sympy.lambdify(x, func)

        # Calculando a equação da reta tangente
        reta_tangente = lambda x: coef_angular * (x - x0) + func_lambda(x0)

        # Mostrando a equação da reta tangente na tela
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, f"A função é: {func}\n")
        resultado_text.insert(tk.END, f"A derivada é: {deriv}\n")
        resultado_text.insert(tk.END, f"A equação da reta tangente é: y = {coef_angular:.2f} * (x - {x0}) + {func_lambda(x0):.2f}\n")

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
    except:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Ocorreu um erro ao calcular a reta tangente. Verifique se a função e o valor de x estão corretos.")


# Criação da janela principal
root = tk.Tk()
root.title("Cálculo de Reta Tangente")
root.geometry("400x400")

# Função Label e Entry
label_func = tk.Label(root, text="Função:")
label_func.pack()
entry_func = tk.Entry(root)
entry_func.pack()

# Valor de x Label e Entry
label_x = tk.Label(root, text="Valor de x:")
label_x.pack()
entry_x = tk.Entry(root)
entry_x.pack()

# Botão de Calcular
button_calcular = tk.Button(root, text="Calcular", command=calcular_tangente)
button_calcular.pack()

# Resultado
label_resultado = tk.Label(root, text="Resultado:")
label_resultado.pack()
resultado_text = tk.Text(root, height=10, width=40)
resultado_text.pack()

root.mainloop()
