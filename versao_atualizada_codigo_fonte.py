import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def show_menu():
    menu_frame.pack()

def hide_menu():
    menu_frame.pack_forget()

def show_section(section_frame):
    hide_menu()
    section_frame.pack()

def show_domain_image_section():
    show_section(aba_dominio)

def show_roots_section():
    show_section(aba_raizes)

def show_limits_section():
    show_section(aba_limite)

def show_derivatives_section():
    show_section(aba_derivada)

def show_graphs_section():
    show_section(aba_graficos)

def show_integrals_section():
    show_section(aba_integrais)

def return_to_menu():
    for frame in [aba_dominio, aba_raizes, aba_limite, aba_derivada, aba_graficos, aba_integrais]:
        frame.pack_forget()
    show_menu()

def inputstr(pai):
    entry = tk.Entry(pai, width=40, bd=1, relief=tk.SOLID)
    entry.pack(pady=10)
    return entry

def botao(pai, func, texto):
    tk.Button(pai, text=texto, command=func, pady=2, padx=2, bd=1, relief=tk.SOLID).pack()

def calculo_derivada():
    global resultado_text_deriv
    try:
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        derivada = sp.diff(func, x)
        point = float(entradaponto.get())
        coef_angular = derivada.subs(x, point)
        reta = (func.subs(x, point) + coef_angular * (x - point))
        resultado_text_deriv.delete(1.0, tk.END)
        resultado_text_deriv.insert(tk.END, f"A derivada da função é: {derivada}\nA equação da reta tangente é: {reta}\n\n")
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular a derivada. Verifique sua entrada.")

def calculo_limite():
    global resultado_text_limite
    try:
        func_str = entradalimit.get()
        func = sp.sympify(func_str)
        variavel = sp.symbols(entradavar.get())
        valor_tendencia = float(entradatend.get())
        
        limite_esquerda = sp.limit(func, variavel, valor_tendencia, dir='-')
        limite_direita = sp.limit(func, variavel, valor_tendencia, dir='+')
        
        resultado_text_limite.delete(1.0, tk.END)

        if limite_esquerda == limite_direita:
            resultado_text_limite.insert(tk.END, f"O limite da função é: {limite_esquerda}")
        else:
            resultado_text_limite.insert(tk.END, f"O limite da função não existe.")
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular o limite. Verifique sua entrada.")

def raiz():
    try:
        numero = float(entradaraiz.get())
        indice_input = entradaindice.get()  # Capturando a entrada do índice
        if not indice_input:                # Verificando se o campo de entrada está vazio
            raise ValueError("Índice não fornecido")
        indice = int(indice_input)
        raiz_value = pow(numero, 1/indice)
        resultado_text_raiz.delete(1.0, tk.END)
        resultado_text_raiz.insert(tk.END, f"A raíz {indice}-ésima de {numero} é: {raiz_value:.4}")
    except ValueError:
        messagebox.showerror("Erro", "Por favor, forneça um índice e/ou número válido para calcular a raiz.")

def plot_grafico():
    global resultado_text_grafico
    try:
        x = sp.Symbol('x')
        func_str = entrada_grafico.get()
        func_list = [sp.sympify(func.strip()) for func in func_str.split(',')]
        func_numeric_list = [sp.lambdify(x, func, 'numpy') for func in func_list]
        x_vals = np.linspace(-10, 10, 100)
        plt.figure()
        for func, func_numeric in zip(func_list, func_numeric_list):
            y_vals = func_numeric(x_vals)
            plt.plot(x_vals, y_vals, label=f'${sp.latex(func)}$')
        
        if show_points_var.get():
            for func in func_list:
                crit_points = sp.solve(sp.diff(func, x), x)
                inflex_points = sp.solve(sp.diff(func, x, x), x)
                
                for cp in crit_points:
                    if cp.is_real:
                        plt.plot(cp, func.subs(x, cp), 'ro', label='Ponto Crítico')
                
                for ip in inflex_points:
                    if ip.is_real:
                        plt.plot(ip, func.subs(x, ip), 'go', label='Ponto de Inflexão')

        plt.axhline(0, color='black', lw=0.6)  # Linha no eixo x
        plt.axvline(0, color='black', lw=0.6)  # Linha no eixo y
        plt.title('Gráfico das Funções')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
        resultado_text_grafico.delete(1.0, tk.END)
        resultado_text_grafico.insert(tk.END, "Gráfico plotado com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao plotar o gráfico. Verifique sua entrada.")

def calculo_dominio_imagem():
    global resultado_text_dom
    try:
        func_str = entradadom.get()
        x = sp.symbols('x')
        func = sp.sympify(func_str)

        # Calculando o domínio
        domain = sp.S.Reals
        singularities = sp.solve(sp.denom(func), x)
        for singularity in singularities:
            domain = domain - sp.FiniteSet(singularity)

        # Calculando a imagem
        critical_points = sp.solve(sp.diff(func, x), x)
        y_values = []

        # Avaliando a função nos pontos críticos e nos extremos do domínio
        for cp in critical_points:
            if cp.is_real:
                y_values.append(func.subs(x, cp))

        # Considerando os limites nos extremos do domínio
        try:
            y_values.append(sp.limit(func, x, sp.oo))
        except:
            pass
        try:
            y_values.append(sp.limit(func, x, -sp.oo))
        except:
            pass

        # Usando alguns valores numéricos se não houver pontos críticos reais
        if not y_values:
            x_values = np.linspace(-100, 100, 1000)
            y_values = [func.subs(x, val) for val in x_values]

        # Calculando os valores mínimo e máximo
        min_value = min(y_values)
        max_value = max(y_values)
        image = sp.Interval(min_value, max_value)

        # Convertendo os resultados para strings
        domain_str = str(domain)
        image_str = str(image)

        resultado_text_dom.delete(1.0, tk.END)
        resultado_text_dom.insert(tk.END, f"O domínio da função é: {domain_str}\nA imagem da função é: {image_str}\n")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular o domínio e a imagem: {e}")

def calculo_integral():
    global resultado_text_int
    try:
        func_str = entrada_integrais.get()
        x = sp.symbols('x')
        func = sp.sympify(func_str)
        limite_inf = entrada_limite_inf.get().strip()
        limite_sup = entrada_limite_sup.get().strip()

        if limite_inf and limite_sup:
            limite_inf = float(limite_inf)
            limite_sup = float(limite_sup)
            integral_def = sp.integrate(func, (x, limite_inf, limite_sup))
            integral_def_str = str(integral_def)
            resultado_text_integral.delete(1.0, tk.END)
            resultado_text_integral.insert(tk.END, f"A integral definida da função de {limite_inf} a {limite_sup} é: {integral_def_str:.3}\n")
        else:
            # Calculando a integral indefinida
            integral = sp.integrate(func, x)
            integral_str = str(integral)
            resultado_text_integral.delete(1.0, tk.END)
            resultado_text_integral.insert(tk.END, f"A integral indefinida da função é: {integral_str} + C\n")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a integral: {e}")

def plot_func_tangente():
    try:
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        derivada = sp.diff(func, x)
        point = float(entradaponto.get())
        coef_angular = derivada.subs(x, point)
        reta = func.subs(x, point) + coef_angular * (x - point)
        
        entrada_grafico.delete(0, tk.END)
        entrada_grafico.insert(0, f"{func_str}, {reta}")
        
        plot_grafico()
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao plotar o gráfico. Verifique sua entrada.")


app = tk.Tk()
app.title('DDX')
app.geometry("800x800")

menu_frame = tk.Frame(app)
menu_frame.pack()

label = tk.Label(menu_frame, text="Selecione uma opção:", font=("Helvetica", 16))
label.pack(pady=20)

options = ["Domínio e Imagem de Funções", "Raiz", "Limites", "Derivadas", "Gráficos", "Integrais"]
commands = [show_domain_image_section, show_roots_section, show_limits_section, show_derivatives_section, show_graphs_section, show_integrals_section]

for i, option in enumerate(options):
    button = tk.Button(menu_frame, text=option, width=40, command=commands[i], relief="raised", bg="#f0f0f0", bd=0.5)
    button.pack(pady=10, ipadx=10, ipady=5)

# Abas e funcionalidades de cada seção
aba_dominio = tk.Frame(app)
aba_raizes = tk.Frame(app)
aba_limite = tk.Frame(app)
aba_derivada = tk.Frame(app)
aba_graficos = tk.Frame(app)
aba_integrais = tk.Frame(app)

# Aba Domínio e Imagem de Funções
lb1 = tk.Label(aba_dominio, text='Insira abaixo a função:', font=("Helvetica", 12))
lb1.pack()
entradadom = inputstr(aba_dominio)
botao(aba_dominio, calculo_dominio_imagem, 'Calcular')
botao(aba_dominio, return_to_menu,'Voltar para o menu')
resultado_text_dom = tk.Text(aba_dominio, height=12, width=52)
resultado_text_dom.pack(padx=10 , pady=10)
resultado_text_dom.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


# Aba Raiz 
lb2 = tk.Label(aba_raizes, text='insira o número:', font=("Helvetica", 12))
lb2.pack()
entradaraiz = inputstr(aba_raizes)
lb3 = tk.Label(aba_raizes, text='insira o índice:', font=("Helvetica", 12))  # Novo label para o índice
lb3.pack()
entradaindice = inputstr(aba_raizes)  # Novo campo para o índice
botao(aba_raizes, raiz , 'Calcular')
botao(aba_raizes, return_to_menu , 'Voltar para o menu')
resultado_text_raiz = tk.Text(aba_raizes, height=12, width=55)
resultado_text_raiz.pack(padx=10 , pady=10)
resultado_text_raiz.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


resultado_text_raiz.insert(tk.END,
                           f"\n\n A raiz n-ésima de um número é definida como um número que, quando elevado a n, é igual a esse número. A radiciação é a operação inversa da potenciação, e todas as propriedades da radiciação são derivadas da potenciação. Aqui, usamos o método (Regressão de Júlia) da aluna brasileira Júlia Pimenta Ferreira de 11 anos para resolver raízes exatas\n\nfonte:  https://mathworld.wolfram.com/ e http://www.sbem.com.br/revista/index.php/emr/index.")



# adicionando imagem
caminho_raiz = 'raiz.png'
imagem_raiz = tk.PhotoImage(file=caminho_raiz)
imagem_raiz = imagem_raiz.subsample(2, 2)
lb_iii = tk.Label(aba_raizes)
lb_iii.pack(padx=10)
lb_iii.config(image=imagem_raiz, width=461, height=113)
lb_iii.image = imagem_raiz

# Aba Limites
lb4 = tk.Label(aba_limite, text='Insira abaixo o limite:', font=("Helvetica", 12))
lb4.pack()
entradalimit = inputstr(aba_limite)
lb5 = tk.Label(aba_limite, text='Insira a variável:', font=("Helvetica", 12))
lb5.pack()
entradavar = inputstr(aba_limite)
lb6 = tk.Label(aba_limite, text='variável tende para que número?', font=("Helvetica", 12))
lb6.pack()
entradatend = inputstr(aba_limite)
botao(aba_limite, calculo_limite , 'Calcular')
botao(aba_limite, return_to_menu, 'Voltar para o menu')
resultado_text_limite = tk.Text(aba_limite, height=12, width=55)
resultado_text_limite.pack(padx=10 , pady=10)
resultado_text_limite.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


resultado_text_limite.insert(tk.END,
                             f"\n\n O limite de uma função descreve o comportamento da função à medida que a variável independente se aproxima de um determinado valor. Em termos simples, estamos interessados em saber para qual valor a função se aproxima à medida que a variável de entrada se aproxima de um ponto específico.\n\n fonte:  Munem, M.A..; Foulis, D.J. Cálculo - Rio de Janeiro - Guanabara Dois , 1982. v1.")


#adicionando imagem
caminho_lim = 'limit.png'
imagem_lim = tk.PhotoImage(file=caminho_lim)
imagem_lim = imagem_lim.subsample(2, 2)
lb_ii = tk.Label(aba_limite)
lb_ii.pack(padx=10)
lb_ii.config(image=imagem_lim, width=461, height=113)
lb_ii.image = imagem_lim

#ADICIONAR AQUI EXEMPLO


# Aba Derivadas
lb7 = tk.Label(aba_derivada, text='Insira abaixo a função:', font=("Helvetica", 12))
lb7.pack()
entradaderiv = inputstr(aba_derivada)
lb8 = tk.Label(aba_derivada, text='Insira o ponto:', font=("Helvetica", 12))
lb8.pack()
entradaponto = inputstr(aba_derivada)
botao(aba_derivada, calculo_derivada , 'Calcular')
botao(aba_derivada, return_to_menu , 'Voltar para o menu')
resultado_text_deriv = tk.Text(aba_derivada, height=12, width=55)
resultado_text_deriv.pack(padx=10 , pady=10)
resultado_text_deriv.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)

resultado_text_deriv.insert(tk.END,
                            f"\n\n A derivada, em termos simples , descreve a taxa de variação instantânea da função em relação à sua variável independente. Se você tem uma função f(x), a derivada f'(x) representa a taxa na qual f(x) está mudando em relação a x\n\n fonte:  Munem, M.A..; Foulis, D.J. Cálculo - Rio de Janeiro - Guanabara Dois , 1982. v1.")

botao(aba_derivada, plot_func_tangente, 'Plotar Função e Reta Tangente')


# adicionando imagem
caminho = 'deriva.png'
imagem = tk.PhotoImage(file=caminho)
lb_i = tk.Label(aba_derivada)
lb_i.pack(padx=10)
lb_i.config(image=imagem, width=445, height=101)
lb_i.image = imagem

# Aba Gráficos
lb9 = tk.Label(aba_graficos, text='Insira a função (use "x" como variável):', font=("Helvetica", 12))
lb9.pack()
entrada_grafico = inputstr(aba_graficos)
show_points_var = tk.IntVar()
show_points_checkbox = tk.Checkbutton(aba_graficos, text="Mostrar pontos de inflexão, mínimos e máximos", variable=show_points_var)
show_points_checkbox.pack()
botao(aba_graficos, plot_grafico , 'Calcular')
botao(aba_graficos, return_to_menu, 'Voltar para o menu')
resultado_text_grafico = tk.Text(aba_graficos, height=12, width=55)
resultado_text_grafico.pack(padx=10 , pady=10)
resultado_text_grafico.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


# Aba Integrais 
lb10 = tk.Label(aba_integrais , text="Insira a função:" , font=("Helvetica", 12))
lb10.pack()
entrada_integrais = inputstr(aba_integrais)
lb11 = tk.Label(aba_integrais , text="Limite inferior (opcional):" , font=("Helvetica", 12))
lb11.pack()
entrada_limite_inf = inputstr(aba_integrais)
lb12 = tk.Label(aba_integrais , text="Limite superior (opcional):" , font=("Helvetica", 12))
lb12.pack()
entrada_limite_sup = inputstr(aba_integrais)
botao(aba_integrais, calculo_integral , 'Calcular')
botao(aba_integrais, return_to_menu, 'Voltar para o menu')
resultado_text_integral = tk.Text(aba_integrais, height=12, width=55)
resultado_text_integral.pack(padx=10 , pady=10)
resultado_text_integral.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


# adicionando imagem
caminho_integral = 'integral.png'
imagem_integral = tk.PhotoImage(file=caminho_integral)
imagem_integral = imagem_integral.subsample(2, 2)
lb_iiii = tk.Label(aba_integrais)
lb_iiii.pack(padx=10)
lb_iiii.config(image=imagem_integral, width=461, height=213)
lb_iiii.image = imagem_integral

app.mainloop()
