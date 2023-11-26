import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import ttk
import math
matplotlib.use('TkAgg')

# FUNÇÕES

def inputstr(pai):
    entry = tk.Entry(pai, width=40 , bd=1 , relief=tk.SOLID)
    entry.pack(pady=10)
    return entry

def botao(pai, func):
    tk.Button(pai, text="Calcular", command=func, pady=2, padx=2 ,  bd=1 , relief=tk.SOLID).pack()

def calculo_derivada():
    try: 
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)  # Converte a string da função em uma expressão simbólica
        derivada = sp.diff(func, x)
        
        point = float(entradaponto.get())
        coef_angular = derivada.subs(x , point)
        reta = (func.subs(x , point) + coef_angular * (x - point))
        
        resultado_text_deriv.delete(1.0, tk.END)
        resultado_text_deriv.insert(tk.END, f"A derivada da função é: {derivada}\nA equação da reta tangente é: {reta}\n\n")
    except:
        resultado_text_deriv.delete(1.0, tk.END)
        resultado_text_deriv.insert(tk.END, f"OCORREU ALGO ERRADO, TENTE NOVAMENTE!OU CONTATE lucas.pinto@grad.iprj.uerj.br")

def calculo_limite():
    try: 
        func_str = entradalimit.get()
        func = sp.sympify(func_str)  # Converte a string da função em uma expressão simbólica
        variavel = sp.symbols(entradavar.get())
        valor_tendencia = float(entradatend.get())
        limite = sp.limit(func, variavel, valor_tendencia)
        resultado_text_limite.delete(1.0, tk.END)
        resultado_text_limite.insert(tk.END, f"O limite da função é: {limite}")
    except:
        resultado_text_limite.delete(1.0, tk.END)
        resultado_text_limite.insert(tk.END, f"OCORREU ALGO ERRADO, TENTE NOVAMENTE!OU CONTATE lucas.pinto@grad.iprj.uerj.br")

def raiz():
    try:
        numero = float(entradaraiz.get())
        m = 0
        n = 0
        if math.sqrt(numero).is_integer():
            while True:
                m = m + n + (n+1)
                n+=1
                if m==numero:
                    return n, numero
        else: 
            n = math.sqrt(numero)
            return n, numero
    except: 
        resultado_text_raiz.delete(1.0, tk.END)
        resultado_text_raiz.insert(tk.END, f"OCORREU ALGO ERRADO, TENTE NOVAMENTE!OU CONTATE lucas.pinto@grad.iprj.uerj.br")

def calculo_raizes():
    try:
        result, numero = raiz()
        resultado_text_raiz.delete(1.0, tk.END)
        resultado_text_raiz.insert(tk.END, f"A raiz de {numero} é: {result}")
    except:
        resultado_text_raiz.delete(1.0, tk.END)
        resultado_text_raiz.insert(tk.END, f"OCORREU ALGO ERRADO, TENTE NOVAMENTE!OU CONTATE lucas.pinto@grad.iprj.uerj.br")

def textresult(pai):
    return tk.Label(pai, text="Resultado:", pady=3)

def plot_grafico():
    try:
        x = sp.Symbol('x')
        func_str = entrada_grafico.get()
        func_list = [sp.sympify(func.strip()) for func in func_str.split(',')]

        # Convertendo as expressões simbólicas para funções numéricas
        func_numeric_list = [sp.lambdify(x, func, 'numpy') for func in func_list]

        # Gerando valores para x
        x_vals = np.linspace(-10, 10, 100)

        # Plotando o gráfico
        plt.figure()
        for i, func_numeric in enumerate(func_numeric_list):
            y_vals = func_numeric(x_vals)
            plt.plot(x_vals, y_vals, label=f'Função {i + 1}')

        plt.title('Gráfico das Funções')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

        resultado_text_grafico.delete(1.0, tk.END)
        resultado_text_grafico.insert(tk.END, "Gráfico plotado com sucesso!")
    except Exception as e:
        resultado_text_grafico.delete(1.0, tk.END)
        resultado_text_grafico.insert(tk.END, f"OCORREU ALGO ERRADO, TENTE NOVAMENTE!OU CONTATE lucas.pinto@grad.iprj.uerj.br")

# criando janela principal

app = tk.Tk()
app.title('DDX')
app.geometry("800x800")

notebook = ttk.Notebook(app)
notebook.place(x=0, y=0, width=800, height=800)

aba_derivada = ttk.Frame(notebook)
notebook.add(aba_derivada, text='Derivadas')

aba_limite = ttk.Frame(notebook)
notebook.add(aba_limite, text='Limites')

aba_raizes = ttk.Frame(notebook)
notebook.add(aba_raizes, text="Raízes quadradas")

aba_graficos = ttk.Frame(notebook)
notebook.add(aba_graficos , text='Gráficos de funções')

# ITENS ABA DERIVADA

lb1 = tk.Label(aba_derivada, text='Insira abaixo a função:')
lb1.pack()
entradaderiv = inputstr(aba_derivada)
lb6 = tk.Label(aba_derivada, text='Insira o ponto:')
lb6.pack()
entradaponto = inputstr(aba_derivada)
botao(aba_derivada, calculo_derivada)
textresult(aba_derivada).pack()
resultado_text_deriv = tk.Text(aba_derivada, height=14, width=50 , padx=10 , pady=10 ,  bd=1 , relief=tk.SOLID)
resultado_text_deriv.pack()
resultado_text_deriv.insert(tk.END, f"\n\n A derivada, em termos simples , descreve a taxa de variação instantânea da função em relação à sua variável independente. Se você tem uma função f(x), a derivada f'(x) representa a taxa na qual f(x) está mudando em relação a x\n\n fonte:  Munem, M.A..; Foulis, D.J. Cálculo - Rio de Janeiro - Guanabara Dois , 1982. v1.")

# adicionando imagem
caminho = 'deriva.png'
imagem = tk.PhotoImage(file=caminho)
lb_i = tk.Label(aba_derivada)
lb_i.pack(padx=10)
lb_i.config(image=imagem, width=445 , height=101)
lb_i.image = imagem


# ITENS ABA LIMITES

lb2 = tk.Label(aba_limite, text='Insira abaixo o limite:')
lb2.pack()
entradalimit = inputstr(aba_limite)
lb3 = tk.Label(aba_limite, text='Insira a variável:')
lb3.pack()
entradavar = inputstr(aba_limite)
lb4 = tk.Label(aba_limite, text='variavel tende para que numero?')
lb4.pack()
entradatend = inputstr(aba_limite)
botao(aba_limite, calculo_limite)
textresult(aba_limite).pack()
resultado_text_limite = tk.Text(aba_limite, height=14, width=50 , padx=10 , pady=10 , bd=1 , relief=tk.SOLID)
resultado_text_limite.pack()
resultado_text_limite.insert(tk.END, f"\n\n O limite de uma função descreve o comportamento da função à medida que a variável independente se aproxima de um determinado valor. Em termos simples, estamos interessados em saber para qual valor a função se aproxima à medida que a variável de entrada se aproxima de um ponto específico.\n\n fonte:  Munem, M.A..; Foulis, D.J. Cálculo - Rio de Janeiro - Guanabara Dois , 1982. v1.")

#adicionando imagem
caminho_lim = 'limit.png'
imagem_lim = tk.PhotoImage(file=caminho_lim)
imagem_lim = imagem_lim.subsample(2,2)
lb_ii = tk.Label(aba_limite)
lb_ii.pack(padx=10)
lb_ii.config(image=imagem_lim, width=461 , height=113)
lb_ii.image = imagem_lim


# ITENS ABA RAIZ

lb5 = tk.Label(aba_raizes, text='insira o número: ')
lb5.pack()
entradaraiz = inputstr(aba_raizes)
entradaraiz.pack()
botao(aba_raizes, calculo_raizes)
textresult(aba_raizes).pack()
resultado_text_raiz = tk.Text(aba_raizes, height=14, width=50 , padx=10 , pady=10 , bd=1 , relief=tk.SOLID)
resultado_text_raiz.pack()
resultado_text_raiz.insert(tk.END, f"\n\n A raiz n-ésima de um número é definida como um número que, quando elevado a n, é igual a esse número. A radiciação é a operação inversa da potenciação, e todas as propriedades da radiciação são derivadas da potenciação. Aqui, usamos o método (Regressão de Júlia) da aluna brasileira Júlia Pimenta Ferreira de 11 anos para resolver raízes exatas\n\nfonte:  https://mathworld.wolfram.com/ e http://www.sbem.com.br/revista/index.php/emr/index.")

#adicionando imagem
caminho_raiz = 'raiz.png'
imagem_raiz = tk.PhotoImage(file=caminho_raiz)
imagem_raiz = imagem_raiz.subsample(2,2)
lb_iii = tk.Label(aba_raizes)
lb_iii.pack(padx=10)
lb_iii.config(image=imagem_raiz, width=461 , height=113)
lb_iii.image = imagem_raiz

# SEÇÃO PARA PLOT DE GRÁFICO

lb7 = tk.Label(aba_graficos, text='Insira a função (use "x" como variável):')
lb7.pack()
entrada_grafico = inputstr(aba_graficos)
botao(aba_graficos, plot_grafico)
textresult(aba_graficos).pack()
resultado_text_grafico = tk.Text(aba_graficos, height=14, width=50, padx=10, pady=10, bd=1, relief=tk.SOLID)
resultado_text_grafico.pack()


app.mainloop()