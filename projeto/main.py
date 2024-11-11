import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import matplotlib

matplotlib.use('TkAgg')

font = ('Arial' , 13)

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
    entry = tk.Entry(pai, width=40, bd=1, relief=tk.SOLID , font=font)
    entry.pack(pady=10)
    return entry

def botao(pai, func, texto):
    tk.Button(pai, text=texto, command=func, pady=3, padx=4, bd=1, relief=tk.SOLID, width=25 , font=font).pack()

def calculo_derivada():
    global resultado_text_deriv
    try:
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        derivada = sp.diff(func, x)
        
        resultado_text_deriv.delete(1.0, tk.END)
        resultado_text_deriv.insert(tk.END, f"A derivada da função é: {derivada}\n")
        
        # Verifica se o ponto foi inserido
        point_str = entradaponto.get()
        if point_str:
            point = float(point_str)
            coef_angular = derivada.subs(x, point)
            reta = func.subs(x, point) + coef_angular * (x - point)
            resultado_text_deriv.insert(tk.END, f"A equação da reta tangente é: {reta}\n\n")
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular a derivada. Verifique sua entrada.")

def calculo_limite():
    global resultado_text_limite
    try:
        func_str = entradalimit.get()
        func = sp.sympify(func_str)
        variavel = sp.symbols(entradavar.get())
        valor_tendencia = float(entradatend.get())
        direcao = direcao_var.get()  # Obtém a direção selecionada

        if direcao == "Ambos":
            limite_esquerda = sp.limit(func, variavel, valor_tendencia, dir='-')
            limite_direita = sp.limit(func, variavel, valor_tendencia, dir='+')
            
            resultado_text_limite.delete(1.0, tk.END)
            if limite_esquerda == limite_direita:
                resultado_text_limite.insert(tk.END, f"O limite da função é: {limite_esquerda}")
            else:
                resultado_text_limite.insert(tk.END, f"O limite da função não existe.")
        else:
            if direcao == "Esquerda":
                limite = sp.limit(func, variavel, valor_tendencia, dir='-')
            elif direcao == "Direita":
                limite = sp.limit(func, variavel, valor_tendencia, dir='+')
            
            resultado_text_limite.delete(1.0, tk.END)
            resultado_text_limite.insert(tk.END, f"O limite da função pela {direcao.lower()} é: {limite}")

    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular o limite. Verifique sua entrada.")

def raiz():
    try:
        numero = float(entradaraiz.get())
        indice_input = entradaindice.get()  # Capturando a entrada do índice
        if not indice_input:                # Verificando se o campo de entrada está vazio
            raise ValueError("Índice não fornecido")
        indice = int(indice_input)
        
        if indice == 2:
            tolerancia = 1e-10
            x = numero /2 # estimativa inicial 
            
            while True:
                raiz_value = 0.5 * (x + numero / x)
                if abs(raiz_value - x) < tolerancia:
                    break
                x = raiz_value
        else: 
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
        interval = (intervalo.get().split(','))
        interval_int = [float(val) for val in interval]
        x_vals = np.linspace(interval_int[0], interval_int[1], 100)
        plt.figure()
        
        result_text = ""
        critical_points = []  # Lista para armazenar os pontos críticos e de inflexão
        inflection_points = []
        vertical_asymptotes = []
        horizontal_asymptotes = []

        for func, func_numeric in zip(func_list, func_numeric_list):
            y_vals = func_numeric(x_vals)
            plt.plot(x_vals, y_vals, label=f'${sp.latex(func)}$')
            
            # Calcular a primeira e segunda derivada
            first_derivative = sp.diff(func, x)
            second_derivative = sp.diff(first_derivative, x)
            
            # Pontos críticos e de inflexão
            crit_points = sp.solve(first_derivative, x, domain=sp.S.Reals)
            inflex_points = sp.solve(second_derivative, x, domain=sp.S.Reals)
            
            # Filtrar os pontos críticos e de inflexão dentro do intervalo de plotagem
            crit_points = [p.evalf() for p in crit_points if p.is_real and interval_int[0] <= p.evalf() <= interval_int[1]]
            inflex_points = [p.evalf() for p in inflex_points if p.is_real and interval_int[0] <= p.evalf() <= interval_int[1]]
            
            # Detectar assíntotas verticais: onde o denominador é zero
            if func.is_rational_function(x):
                denom = sp.denom(func)
                vertical_asymptotes = sp.solve(denom, x, domain=sp.S.Reals)
                vertical_asymptotes = [a.evalf() for a in vertical_asymptotes if interval_int[0] <= a.evalf() <= interval_int[1]]

            # Detectar assíntotas horizontais: limite de x para infinito e -infinito
            lim_pos_inf = sp.limit(func, x, sp.oo)
            lim_neg_inf = sp.limit(func, x, -sp.oo)
            
            if lim_pos_inf.is_finite:
                horizontal_asymptotes.append(lim_pos_inf)
            if lim_neg_inf.is_finite and lim_neg_inf != lim_pos_inf:
                horizontal_asymptotes.append(lim_neg_inf)
            
            # Adicionar pontos críticos e de inflexão à lista
            for p in crit_points:
                y_p = func.subs(x, p).evalf()
                # Verificar o tipo de ponto crítico usando a primeira derivada
                left_val = p - 0.01
                right_val = p + 0.01
                left_slope = first_derivative.subs(x, left_val).evalf()
                right_slope = first_derivative.subs(x, right_val).evalf()
                
                if left_slope > 0 and right_slope < 0:
                    point_type = "Máximo"
                elif left_slope < 0 and right_slope > 0:
                    point_type = "Mínimo"
                else:
                    point_type = "Ponto de Sela"
                
                critical_points.append((p, y_p, point_type))
            
            for p in inflex_points:
                y_p = func.subs(x, p).evalf()
                inflection_points.append((p, y_p))
            
            # Determinar intervalos de crescimento e decrescimento
            growth_intervals = []
            decay_intervals = []
            crit_points_sorted = sorted(crit_points)
            
            test_points = [interval_int[0]] + crit_points_sorted + [interval_int[1]]
            
            for i in range(len(test_points) - 1):
                test_val = (test_points[i] + test_points[i + 1]) / 2
                if first_derivative.subs(x, test_val) > 0:
                    growth_intervals.append((test_points[i], test_points[i + 1]))
                else:
                    decay_intervals.append((test_points[i], test_points[i + 1]))
                    
            # Determinar intervalos de concavidade
            concave_up_intervals = []
            concave_down_intervals = []
            inflex_points_sorted = sorted(inflex_points)
            
            test_points = [interval_int[0]] + inflex_points_sorted + [interval_int[1]]
            
            for i in range(len(test_points) - 1):
                test_val = (test_points[i] + test_points[i + 1]) / 2
                if second_derivative.subs(x, test_val) > 0:
                    concave_up_intervals.append((test_points[i], test_points[i + 1]))
                else:
                    concave_down_intervals.append((test_points[i], test_points[i + 1]))
                    
            result_text += f"Função: {func}\n"
            result_text += "Intervalos de Crescimento: " + ", ".join([f"({a}, {b})" for a, b in growth_intervals]) + "\n"
            result_text += "Intervalos de Decrescimento: " + ", ".join([f"({a}, {b})" for a, b in decay_intervals]) + "\n"
            result_text += "Intervalos de Concavidade para Cima: " + ", ".join([f"({a}, {b})" for a, b in concave_up_intervals]) + "\n"
            result_text += "Intervalos de Concavidade para Baixo: " + ", ".join([f"({a}, {b})" for a, b in concave_down_intervals]) + "\n"
            if vertical_asymptotes:
                result_text += "Assíntotas Verticais em: " + ", ".join([f"x = {a}" for a in vertical_asymptotes]) + "\n"
            if horizontal_asymptotes:
                result_text += "Assíntotas Horizontais em: " + ", ".join([f"y = {a}" for a in horizontal_asymptotes]) + "\n"
            result_text += f"\n"

        if show_points_var.get():
            for cp, y_cp, point_type in critical_points:
                if point_type == "Mínimo":
                    plt.plot(cp, y_cp, 'bo', label=f'{point_type} ({cp:.2f}, {y_cp:.2f})')
                elif point_type == "Máximo":
                    plt.plot(cp, y_cp, 'ro', label=f'{point_type} ({cp:.2f}, {y_cp:.2f})')
                else:
                    plt.plot(cp, y_cp, 'yo', label=f'{point_type} ({cp:.2f}, {y_cp:.2f})')
                
            for ip, y_ip in inflection_points:
                plt.plot(ip, y_ip, 'go', label=f'Ponto de Inflexão ({ip:.2f}, {y_ip:.2f})')

        plt.axhline(0, color='red', lw=0.8)  # Linha no eixo x
        plt.axvline(0, color='red', lw=0.8)  # Linha no eixo y
        plt.title('Gráfico das Funções')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        resultado_text_grafico.delete(1.0, tk.END)
        resultado_text_grafico.insert(tk.END, result_text + "Gráfico plotado com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico. Verifique sua entrada.\n{e}")

def calculo_dominio_imagem():
    global resultado_text_dom
    try:
        func_str = entradadom.get()
        x = sp.symbols('x')
        func = sp.sympify(func_str)

        # Calculando o domínio diretamente
        domain = sp.calculus.util.continuous_domain(func, x, sp.S.Reals)

        # Calculando a imagem simplificada
        y_values = []
        
        # Pontos críticos
        critical_points = sp.solve(sp.diff(func, x), x)

        for cp in critical_points:
            if cp.is_real:
                y_values.append(func.subs(x, cp))

        # Considerando os limites nos extremos do domínio
        try:
            limit_pos_inf = sp.limit(func, x, sp.oo)
            if limit_pos_inf.is_real:
                y_values.append(limit_pos_inf)
        except:
            pass
        
        try:
            limit_neg_inf = sp.limit(func, x, -sp.oo)
            if limit_neg_inf.is_real:
                y_values.append(limit_neg_inf)
        except:
            pass

        # Usando valores numéricos se não houver pontos críticos reais
        if not y_values:
            x_values = np.linspace(-100, 100, 1000)
            y_values = [func.subs(x, val) for val in x_values]

        # Calculando os valores mínimo e máximo da imagem
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
    global resultado_text_integral  # Certifique-se de que o nome da variável está correto
    try:
        func_str = entrada_integrais.get()
        x = sp.symbols('x')
        func = sp.sympify(func_str)
        
        # Obtendo os limites inferiores e superiores, se houver
        limite_inf = entrada_limite_inf.get().strip()
        limite_sup = entrada_limite_sup.get().strip()

        if limite_inf and limite_sup:
            # Calculando a integral definida
            limite_inf = float(limite_inf)
            limite_sup = float(limite_sup)
            integral_def = sp.integrate(func, (x, limite_inf, limite_sup))
            resultado_text_integral.delete(1.0, tk.END)
            resultado_text_integral.insert(tk.END, f"A integral definida da função de {limite_inf} a {limite_sup} é: {integral_def}\n")
        else:
            # Calculando a integral indefinida
            integral = sp.integrate(func, x)
            resultado_text_integral.delete(1.0, tk.END)
            resultado_text_integral.insert(tk.END, f"A integral indefinida da função é: {integral} + C\n")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a integral: {e}")

def plot_func_tangente():
    try:
        # Define a variável simbólica
        x = sp.Symbol('x')
        
        # Obtém a função inserida pelo usuário e a ponto
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        point = float(entradaponto.get())
        
        # Calcula a derivada da função
        derivada = sp.diff(func, x)
        
        # Calcula o coeficiente angular da reta tangente
        coef_angular = derivada.subs(x, point)
        
        # Calcula a equação da reta tangente
        reta = func.subs(x, point) + coef_angular * (x - point)
        
        # Converte as funções simbólicas para funções numéricas
        func_num = sp.lambdify(x, func, "numpy")
        reta_num = sp.lambdify(x, reta, "numpy")
        
        # Gera os valores de x
        x_vals = np.linspace(-10, 10, 400)
        
        # Calcula os valores de y para a função original e a reta tangente
        y_vals_func = func_num(x_vals)
        y_vals_reta = reta_num(x_vals)
        
        # Plota as funções
        plt.figure()
        plt.plot(x_vals, y_vals_func, label=f"f(x) = {func_str}")
        plt.plot(x_vals, y_vals_reta, label=f"Tangente em x = {point}")
        
        # Configurações adicionais do gráfico
        plt.axhline(0, color='red', lw=0.8)  # Linha no eixo x
        plt.axvline(0, color='red', lw=0.8)  # Linha no eixo y
        plt.title('Gráfico das Funções')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico: {e}")



def exemplo_raiz():
    example_text = ("Exemplo de Raiz Quadrada:\n"
        "Número: 256\n"
        "Definição: A raiz quadrada de um número é um valor que, quando multiplicado por si mesmo, "
        "resulta no número original.\n"
        "Cálculo: A raiz quadrada de 256 é 16, pois 16 * 16 = 256.\n"
        "Propriedades: A raiz quadrada de um número positivo é sempre um número positivo. "
        "Neste caso, a raiz quadrada de 256 é um valor exato e inteiro, 16.")
    resultado_text_raiz.delete(1.0 , tk.END)
    resultado_text_raiz.insert(tk.END , example_text)


def exemplo_dominio_imagem():
    example_text = (
        "Exemplo de Domínio e Imagem:\n"
        "Função: f(x) = 1/(x-2)\n"
        "Domínio: Todos os valores de x, exceto x=2. Isso porque a função se torna indefinida quando x=2, "
        "pois resultaria em uma divisão por zero.\n"
        "Imagem: Todos os valores reais, exceto f(x)=0. A função nunca toca o eixo x, "
        "pois não há valor de x que faça a função igual a zero."
    )
    resultado_text_dom.delete(1.0, tk.END)
    resultado_text_dom.insert(tk.END, example_text)

def exemplo_limite():
    example_text = (
        "Exemplo de Limite:\n"
        "Função: f(x) = (x^2 - 1)/(x - 1)\n"
        "Para calcular o limite de f(x) quando x tende a 1, simplificamos a função:\n"
        "f(x) = (x + 1) para x ≠ 1.\n"
        "Então, o limite de f(x) quando x tende a 1 é 2.\n"
        "Lembre-se de que o limite se refere ao valor que a função se aproxima à medida que x se aproxima de 1."
    )
    resultado_text_limite.delete(1.0, tk.END)
    resultado_text_limite.insert(tk.END, example_text)

def exemplo_derivada():
    example_text = (
        "Exemplo de Derivada e Tangente:\n"
        "Função: f(x) = x^2\n"
        "Derivada: f'(x) = 2x. Isso representa a inclinação da função em qualquer ponto x.\n"
        "No ponto x=3, f'(3) = 6. Isso significa que a inclinação da tangente à curva no ponto (3, f(3)) é 6.\n"
        "A equação da reta tangente é dada por: y = f(3) + f'(3)*(x - 3)\n"
        "Neste caso, a reta tangente é y = 9 + 6(x - 3), simplificando: y = 6x - 9."
    )
    resultado_text_deriv.delete(1.0, tk.END)
    resultado_text_deriv.insert(tk.END, example_text)

def exemplo_integral():
    example_text = (
        "Exemplo de Integral:\n"
        "Função: f(x) = x^2\n"
        "Integral Indefinida: ∫x^2 dx = (1/3)x^3 + C, onde C é a constante de integração.\n"
        "Integral Definida de 0 a 2: ∫(de 0 a 2) x^2 dx = [(1/3)x^3] de 0 a 2 = (8/3) - 0 = 8/3.\n"
        "Isso representa a área sob a curva de f(x) entre x=0 e x=2."
    )
    resultado_text_integral.delete(1.0, tk.END)
    resultado_text_integral.insert(tk.END, example_text)


app = tk.Tk()
app.title('DDX')
app.geometry("700x800")

menu_frame = tk.Frame(app)
menu_frame.pack()

label = tk.Label(menu_frame, text="Selecione uma opção:", font=("Helvetica", 16))
label.pack(pady=20)

options = ["Domínio e Imagem de Funções", "Raiz", "Limites", "Derivadas", "Gráficos", "Integrais"]
commands = [show_domain_image_section, show_roots_section, show_limits_section, show_derivatives_section, show_graphs_section, show_integrals_section]

for i, option in enumerate(options):
    button = tk.Button(menu_frame, text=option, width=40, command=commands[i], relief="raised", bg="#f0f0f0", bd=0.5 , font=font)
    button.pack(pady=10, ipadx=10, ipady=5)

button = tk.Button(menu_frame , text="Manual do DDX" , width=40 , command=lambda: webbrowser.open('https://github.com/LucasRodriguesEstorckPinto/programming-calculus-1/blob/main/DDX%20MANUAL.pdf') , relief="raised" ,bg="#f0f0f0" , bd=0.5 , font=font)
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
botao(aba_dominio, exemplo_dominio_imagem, "Exemplo")
botao(aba_dominio, return_to_menu,'Voltar para o menu')
resultado_text_dom = tk.Text(aba_dominio, height=12, width=52, font = font)
resultado_text_dom.pack(padx=10 , pady=10)
resultado_text_dom.tag_configure("margin", lmargin1=10, lmargin2=10, rmargin=10)


# Aba Raiz 
lb2 = tk.Label(aba_raizes, text='insira o número:', font=("Helvetica", 12))
lb2.pack()
entradaraiz = inputstr(aba_raizes)
lb3 = tk.Label(aba_raizes, text='insira o índice:', font=("Helvetica", 12))  # Novo label para o índice
lb3.pack()
entradaindice = inputstr(aba_raizes)  # Novo campo para o índice
botao(aba_raizes, raiz , 'Calcular')
botao(aba_raizes, exemplo_raiz, "Exemplo")
botao(aba_raizes, return_to_menu , 'Voltar para o menu')
resultado_text_raiz = tk.Text(aba_raizes, height=12, width=55 , font=font)
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

direcao_var = tk.StringVar(value="Ambos")  # Valor padrão é "Ambos"
direcao_menu = tk.OptionMenu(aba_limite, direcao_var, "Esquerda", "Direita", "Ambos")
direcao_menu.pack()

botao(aba_limite, calculo_limite , 'Calcular')
botao(aba_limite, exemplo_limite, "Exemplo")
botao(aba_limite, return_to_menu, 'Voltar para o menu')
resultado_text_limite = tk.Text(aba_limite, height=12, width=55 , font=font)
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
botao(aba_derivada, exemplo_derivada, "Exemplo")
botao(aba_derivada, return_to_menu , 'Voltar para o menu')
resultado_text_deriv = tk.Text(aba_derivada, height=12, width=55 , font=font)
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
lb9 = tk.Label(aba_graficos, text='Insira a função (use "x" como variável):', font=("Arial", 12))
lb9.pack()
entrada_grafico = inputstr(aba_graficos)
lb13 = tk.Label(aba_graficos, text='Insira o intervalo:', font=("Arial", 12))
lb13.pack()
intervalo = inputstr(aba_graficos)
show_points_var = tk.IntVar()
show_points_checkbox = tk.Checkbutton(aba_graficos, text="Mostrar pontos de inflexão, mínimos e máximos", variable=show_points_var)
show_points_checkbox.pack()
botao(aba_graficos, plot_grafico , 'Calcular')
botao(aba_graficos, return_to_menu, 'Voltar para o menu')
resultado_text_grafico = tk.Text(aba_graficos, height=12, width=55, font=font)
resultado_text_grafico.pack(padx=10 , pady=10)
resultado_text_grafico.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)


# Aba Integrais 
lb10 = tk.Label(aba_integrais , text="Insira a função:" , font=("Arial", 12))
lb10.pack()
entrada_integrais = inputstr(aba_integrais)
lb11 = tk.Label(aba_integrais , text="Limite inferior (opcional):" , font=("Arial", 12))
lb11.pack()
entrada_limite_inf = inputstr(aba_integrais)
lb12 = tk.Label(aba_integrais , text="Limite superior (opcional):" , font=("Arial", 12))
lb12.pack()
entrada_limite_sup = inputstr(aba_integrais)
botao(aba_integrais, calculo_integral , 'Calcular')
botao(aba_integrais, exemplo_integral, "Exemplo")
botao(aba_integrais, return_to_menu, 'Voltar para o menu')
resultado_text_integral = tk.Text(aba_integrais, height=12, width=55,font=font) 
resultado_text_integral.pack(padx=10 , pady=10)
resultado_text_integral.tag_configure("padding", lmargin1=10, lmargin2=10, rmargin=10)
resultado_text_integral.insert(tk.END , f"\n\nA integral de uma função é uma medida da área sob a curva dessa função, em um intervalo específico. Se a função é contínua em um intervalo [a, b], a integral definida dessa função, denotada por ∫(de a a b) f(x) dx, representa a soma das áreas de infinitos retângulos infinitamente estreitos que se encaixam sob a curva de f(x) de x=a até x=b.\n\n fonte:  Munem, M.A..; Foulis, D.J. Cálculo - Rio de Janeiro - Guanabara Dois , 1982. v1.")


# adicionando imagem
caminho_integral = 'integral.png'
imagem_integral = tk.PhotoImage(file=caminho_integral)
imagem_integral = imagem_integral.subsample(2, 2)
lb_iiii = tk.Label(aba_integrais)
lb_iiii.pack(padx=10)
lb_iiii.config(image=imagem_integral, width=461, height=213)
lb_iiii.image = imagem_integral

app.mainloop()