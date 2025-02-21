import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import matplotlib
import customtkinter as ctk
import re

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

class ModernEntry(ctk.CTkEntry):
    def get(self):
        text = super().get()
        # Se houver um dígito imediatamente seguido de "pi" ou "e", insere o sinal de multiplicação.
        text = re.sub(r'(?<=\d)(?=pi\b)', '*', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=\d)(?=e\b)', '*', text, flags=re.IGNORECASE)
        # Substitui ocorrências isoladas (ou após o *) de "pi" e "e" por "pi" e "E"
        # (Sympy já reconhece "pi" e "E" como as constantes π e e)
        text = re.sub(r'\bpi\b', 'pi', text, flags=re.IGNORECASE)
        text = re.sub(r'\be\b', 'E', text, flags=re.IGNORECASE)
        return text

# Cria um rótulo e uma entrada logo abaixo
def labeled_input(parent, label_text):
    label = ctk.CTkLabel(parent, text=label_text, font=font)
    label.pack(padx=10, pady=(10, 0), anchor="w")
    entry = ModernEntry(parent, width=400, height=30, corner_radius=5, font=font)
    entry.pack(padx=10, pady=(0,10))
    return entry


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


def numerical_roots(sym_expr, var, lower, upper, num_points=500):
    """
    Procura raízes de sym_expr no intervalo [lower, upper] usando
    uma busca por mudança de sinal e sp.nsolve.
    Retorna uma lista de raízes numéricas.
    """
    func_num = sp.lambdify(var, sym_expr, 'numpy')
    sample_points = np.linspace(lower, upper, num_points)
    roots = []
    for i in range(len(sample_points) - 1):
        a = sample_points[i]
        b = sample_points[i + 1]
        fa = func_num(a)
        fb = func_num(b)
        # Se o valor for exatamente zero, adiciona a raiz
        if fa == 0:
            if lower <= a <= upper and not any(abs(a - r) < 1e-5 for r in roots):
                roots.append(a)
        # Se houver mudança de sinal, tenta encontrar a raiz
        elif fa * fb < 0:
            try:
                r = sp.nsolve(sym_expr, a)
                r_val = float(r.evalf())
                if lower <= r_val <= upper and not any(abs(r_val - rr) < 1e-5 for rr in roots):
                    roots.append(r_val)
            except Exception:
                pass
    return roots

def plot_grafico():
    global resultado_text_grafico
    try:
        # Utiliza o estilo "ggplot" nativo do matplotlib para uma aparência mais moderna
        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'legend.fontsize': 12
        })
        
        # Define a variável simbólica e lê as entradas da interface
        x = sp.symbols('x')
        func_str = entrada_grafico.get()  # Exemplo: "sin(x)*x**2, cos(x)"
        func_list = [sp.sympify(f.strip()) for f in func_str.split(',')]
        func_numeric_list = [sp.lambdify(x, func, 'numpy') for func in func_list]

        # Obtém o intervalo de análise (ex.: "-10,10")
        intervalo_str = intervalo.get()
        lower, upper = map(float, intervalo_str.split(','))
        x_vals = np.linspace(lower, upper, 800)
        
        # Cria a figura e o eixo
        fig, ax = plt.subplots(figsize=(10, 6))
        result_text = ""
        
        for func, func_numeric in zip(func_list, func_numeric_list):
            # Plota a função
            y_vals = func_numeric(x_vals)
            ax.plot(x_vals, y_vals, label=f'${sp.latex(func)}$', linewidth=2.5)
            
            # Calcula as derivadas
            fprime = sp.diff(func, x)
            fsecond = sp.diff(fprime, x)
            
            # --- Pontos Críticos (f'(x)=0) ---
            cp_candidates = sp.solveset(fprime, x, domain=sp.Interval(lower, upper))
            cp = []
            if isinstance(cp_candidates, sp.ConditionSet) or not cp_candidates:
                cp = numerical_roots(fprime, x, lower, upper, num_points=1000)
            else:
                cp_candidates = list(cp_candidates)
                for candidate in cp_candidates:
                    try:
                        candidate_val = float(candidate.evalf())
                        if lower <= candidate_val <= upper:
                            cp.append(candidate_val)
                    except Exception:
                        continue
                numeric_cp = numerical_roots(fprime, x, lower, upper, num_points=1000)
                for r in numeric_cp:
                    if not any(abs(r - cr) < 1e-5 for cr in cp):
                        cp.append(r)
            cp = sorted(cp)
            
            # --- Pontos de Inflexão (f''(x)=0) ---
            ip_candidates = sp.solveset(fsecond, x, domain=sp.Interval(lower, upper))
            ip = []
            if isinstance(ip_candidates, sp.ConditionSet) or not ip_candidates:
                ip = numerical_roots(fsecond, x, lower, upper, num_points=1000)
            else:
                ip_candidates = list(ip_candidates)
                for candidate in ip_candidates:
                    try:
                        candidate_val = float(candidate.evalf())
                        if lower <= candidate_val <= upper:
                            ip.append(candidate_val)
                    except Exception:
                        continue
                numeric_ip = numerical_roots(fsecond, x, lower, upper, num_points=1000)
                for r in numeric_ip:
                    if not any(abs(r - rp) < 1e-5 for rp in ip):
                        ip.append(r)
            ip = sorted(ip)
            
            # Se o checkbox estiver marcado, mostra os pontos explicitamente
            if show_points_var.get():
                # --- Marcação dos Pontos Críticos com flags coloridas ---
                for p in cp:
                    y_p = float(func.subs(x, p).evalf())
                    try:
                        fsecond_val = float(fsecond.subs(x, p).evalf())
                    except Exception:
                        fsecond_val = None
                    
                    if fsecond_val is not None:
                        if fsecond_val < 0:
                            point_type = "Máximo"
                            color = "#e41a1c"  # vermelho vibrante
                            marker = "^"       # seta para cima
                            offset = (0.4, 0.4)
                        elif fsecond_val > 0:
                            point_type = "Mínimo"
                            color = "#4daf4a"  # verde
                            marker = "v"       # seta para baixo
                            offset = (0.4, -0.4)
                        else:
                            point_type = "Sela"
                            color = "#ff7f00"  # laranja
                            marker = "D"       # diamante
                            offset = (0.4, 0.4)
                    else:
                        point_type = "Crítico"
                        color = "#984ea3"  # roxo
                        marker = "o"
                        offset = (0.4, 0.4)
                    
                    ax.scatter(p, y_p, color=color, marker=marker, s=100, edgecolors='black', zorder=6)
                    ax.annotate(
                        f'{point_type}\n({p:.2f}, {y_p:.2f})',
                        xy=(p, y_p),
                        xytext=(p + offset[0], y_p + offset[1]),
                        textcoords='data',
                        fontsize=10,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none'),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5),
                        zorder=7
                    )
                    result_text += f'{point_type} em ({p:.2f}, {y_p:.2f})\n'
                
                # --- Marcação dos Pontos de Inflexão com flag quadrada azul ---
                for p in ip:
                    y_p = float(func.subs(x, p).evalf())
                    ax.scatter(p, y_p, color="#377eb8", marker='s', s=100, edgecolors='black', zorder=6)
                    ax.annotate(
                        f'Inflexão\n({p:.2f}, {y_p:.2f})',
                        xy=(p, y_p),
                        xytext=(p + 0.4, y_p + 0.4),
                        textcoords='data',
                        fontsize=10,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3', fc="#377eb8", ec='none'),
                        arrowprops=dict(arrowstyle='-|>', color="#377eb8", lw=1.5),
                        zorder=7
                    )
                    result_text += f'Inflexão em ({p:.2f}, {y_p:.2f})\n'
            else:
                result_text += "Pontos não explicitados (checkbox desativado).\n"
            
            result_text += "\n"
        
        # Configurações finais do gráfico
        ax.axhline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.axvline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title('Gráfico das Funções', fontsize=18, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        # Atualiza a área de texto com os resultados
        resultado_text_grafico.delete('1.0', tk.END)
        resultado_text_grafico.insert(tk.END, result_text + "\nGráfico plotado com sucesso!")
        
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico. Verifique sua entrada.\n{e}")

def calcular_dominio(func, x):
    """Calcula o domínio analítico da função de forma robusta."""
    try:
        # Tenta calcular o domínio contínuo na reta real
        dominio = sp.calculus.util.continuous_domain(func, x, sp.S.Reals)
        return dominio
    except Exception as e:
        # Fallback: amostra numérica para identificar onde a função está definida
        try:
            sample_points = np.linspace(-100, 100, 1000)
            valid_points = []
            for val in sample_points:
                try:
                    result = func.subs(x, val)
                    # Verifica se o resultado é um número real e não contém infinitos ou indeterminados
                    if result.is_real and not result.has(sp.oo, sp.zoo, sp.nan):
                        valid_points.append(val)
                except Exception:
                    continue
            if valid_points:
                return sp.Interval(min(valid_points), max(valid_points))
            else:
                return "Domínio não determinado"
        except Exception as e2:
            return f"Erro ao calcular o domínio: {e} | {e2}"

def calcular_imagem(func, x, dominio):
    """Calcula a imagem (faixa de valores) da função de forma robusta."""
    try:
        func_str = str(func)
        # Tratamentos especiais para funções trigonométricas comuns
        if func_str.strip() in ['sin(x)', 'cos(x)']:
            return "[-1, 1]"
        if any(trig in func_str for trig in ['tan', 'cot']):
            return "Todos os reais (exceto singularidades)"
        if any(trig in func_str for trig in ['sec', 'csc']):
            return "(-∞, -1] ∪ [1, ∞)"
        
        y_values = []

        # 1. Pontos críticos: onde a derivada é zero
        try:
            deriv = sp.diff(func, x)
            critical_points = sp.solve(deriv, x)
            for cp in critical_points:
                try:
                    cp_val = float(cp.evalf())
                    # Se o domínio for um intervalo, verifica se o ponto está contido nele
                    if isinstance(dominio, sp.Interval):
                        if dominio.contains(cp_val):
                            y_val = func.subs(x, cp)
                            y_values.append(y_val)
                    else:
                        y_values.append(func.subs(x, cp))
                except Exception:
                    continue
        except Exception:
            pass

        # 2. Limites nos extremos do domínio (sejam finitos ou infinitos)
        try:
            limit_pos = sp.limit(func, x, sp.oo)
            if limit_pos.is_real and not limit_pos.has(sp.oo, sp.zoo, sp.nan):
                y_values.append(limit_pos)
        except Exception:
            pass
        try:
            limit_neg = sp.limit(func, x, -sp.oo)
            if limit_neg.is_real and not limit_neg.has(sp.oo, sp.zoo, sp.nan):
                y_values.append(limit_neg)
        except Exception:
            pass

        # 3. Amostragem numérica dentro do domínio para captar variações
        try:
            if isinstance(dominio, sp.Interval):
                a = float(dominio.start) if dominio.start != -sp.oo else -100
                b = float(dominio.end) if dominio.end != sp.oo else 100
            else:
                a, b = -100, 100
            sample_points = np.linspace(a, b, 1000)
            for val in sample_points:
                try:
                    y_val = func.subs(x, sp.Float(val))
                    if y_val.is_real and not y_val.has(sp.oo, sp.zoo, sp.nan):
                        y_values.append(float(y_val))
                except Exception:
                    continue
        except Exception:
            pass

        # Remove duplicatas (comparando valores numéricos)
        y_values_clean = []
        for y in y_values:
            try:
                y_num = float(y)
                if not any(abs(y_num - float(existing)) < 1e-5 for existing in y_values_clean):
                    y_values_clean.append(y_num)
            except Exception:
                # Caso não seja conversível para float, adiciona de qualquer forma
                y_values_clean.append(y)
        
        if y_values_clean:
            min_val = min(y_values_clean)
            max_val = max(y_values_clean)
            return f"[{min_val:.4f}, {max_val:.4f}]"
        else:
            return "Imagem indefinida (não foram encontrados valores válidos)"
    except Exception as e:
        return f"Erro ao calcular a imagem: {e}"

def calculo_dominio_imagem():
    global resultado_text_dom
    try:
        func_str = entradadom.get()
        x = sp.symbols('x')
        try:
            func = sp.sympify(func_str)
        except Exception as e:
            resultado_text_dom.delete(1.0, tk.END)
            resultado_text_dom.insert(tk.END, f"Erro ao interpretar a função: {e}")
            return
        
        # Calcula o domínio usando o método analítico com fallback
        dominio = calcular_dominio(func, x)
        # Se houver erro na determinação do domínio, define imagem de forma apropriada
        if isinstance(dominio, str) and "Erro" in dominio:
            imagem = "Não foi possível calcular a imagem devido ao domínio inválido."
        else:
            imagem = calcular_imagem(func, x, dominio)
        
        # Formata a saída para a interface
        resultado = f"""Resultados:
========================
Função: {func_str}
Domínio: {dominio}
Imagem: {imagem}
========================"""
        resultado_text_dom.delete('1.0', tk.END)
        resultado_text_dom.insert(tk.END, resultado)
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular domínio e imagem: {e}")

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

button = tk.Button(menu_frame , text="Manual do DDX" , width=40 , command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing') , relief="raised" ,bg="#f0f0f0" , bd=0.5 , font=font)
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
entradadom = labeled_input(aba_dominio, "Expressão:")
botao(aba_dominio, calculo_dominio_imagem, 'Calcular')
botao(aba_dominio, exemplo_dominio_imagem, "Exemplo")
botao(aba_dominio, return_to_menu,'Voltar para o menu')
resultado_text_dom = tk.Text(aba_dominio, height=12, width=52, font = font)
resultado_text_dom.pack(padx=10 , pady=10)
resultado_text_dom.tag_configure("margin", lmargin1=10, lmargin2=10, rmargin=10)

resultado_text_dom.insert(tk.END,
                           f"\n\n O domínio de uma função é o conjunto de valores de entrada em que ela está definida, enquanto a imagem é o conjunto de saídas resultantes da aplicação da função. Por exemplo, em f(x) = 1/(x – 2), x = 2 é excluído do domínio para evitar divisão por zero, e em f(x) = x², a imagem são os números reais não negativos. Esses conceitos são essenciais para garantir a validade das operações e construir modelos matemáticos.\n\nfonte: https://mathworld.wolfram.com/ e http://www.sbem.com.br/revista/index.php/emr/index")


# Aba Raiz 
lb2 = tk.Label(aba_raizes, text='insira o número:', font=("Helvetica", 12))
lb2.pack()
entradaraiz = labeled_input(aba_raizes, "Expressão:")
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
entradalimit = labeled_input(aba_limite, "Expressão:")
lb5 = tk.Label(aba_limite, text='Insira a variável:', font=("Helvetica", 12))
lb5.pack()
entradavar = labeled_input(aba_limite, "variavel")
lb6 = tk.Label(aba_limite, text='variável tende para que número?', font=("Helvetica", 12))
lb6.pack()
entradatend = labeled_input(aba_limite, "tendencia")

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
entradaderiv = labeled_input(aba_derivada, "")
lb8 = tk.Label(aba_derivada, text='Insira o ponto:', font=("Helvetica", 12))
lb8.pack()
entradaponto = labeled_input(aba_derivada, "")
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
entrada_grafico = labeled_input(aba_graficos, "")
lb13 = tk.Label(aba_graficos, text='Insira o intervalo:', font=("Arial", 12))
lb13.pack()
intervalo = labeled_input(aba_graficos, "")
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
entrada_integrais = labeled_input(aba_integrais, "")
lb11 = tk.Label(aba_integrais , text="Limite inferior (opcional):" , font=("Arial", 12))
lb11.pack()
entrada_limite_inf = labeled_input(aba_integrais, "")
lb12 = tk.Label(aba_integrais , text="Limite superior (opcional):" , font=("Arial", 12))
lb12.pack()
entrada_limite_sup = labeled_input(aba_integrais, "")
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