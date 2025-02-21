import customtkinter as ctk
import tkinter.messagebox as messagebox
import webbrowser
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re

# Configuração do tema (dark, light ou system)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Fonte padrão para os widgets
font = ("Arial", 14)



def calculo_derivada():
    global resultado_text_deriv, entradaderiv, entradaponto
    try:
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        derivada = sp.diff(func, x)
        
        resultado_text_deriv.delete("1.0", ctk.END)
        resultado_text_deriv.insert(ctk.END, f"A derivada da função é: {derivada}\n")
        
        # Verifica se o ponto foi inserido
        point_str = entradaponto.get()
        if point_str:
            point = float(sp.sympify(point_str))
            coef_angular = derivada.subs(x, point)
            reta = func.subs(x, point) + coef_angular * (x - point)
            resultado_text_deriv.insert(ctk.END, f"A equação da reta tangente é: {reta}\n\n")
    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular a derivada. Verifique sua entrada.")

def calculo_limite():
    global resultado_text_limite, entradalimit, entradavar, entradatend, direcao_var
    try:
        func_str = entradalimit.get()
        func = sp.sympify(func_str)
        variavel = sp.symbols(entradavar.get())
        valor_tendencia = float(sp.sympify(entradatend.get()))
        direcao = direcao_var.get()  # Obtém a direção selecionada

        if direcao == "Ambos":
            limite_esquerda = sp.limit(func, variavel, valor_tendencia, dir='-')
            limite_direita = sp.limit(func, variavel, valor_tendencia, dir='+')
            
            resultado_text_limite.delete("1.0", ctk.END)
            if limite_esquerda == limite_direita:
                resultado_text_limite.insert(ctk.END, f"O limite da função é: {limite_esquerda}")
            else:
                resultado_text_limite.insert(ctk.END, f"O limite da função não existe.")
        else:
            if direcao == "Esquerda":
                limite = sp.limit(func, variavel, valor_tendencia, dir='-')
            elif direcao == "Direita":
                limite = sp.limit(func, variavel, valor_tendencia, dir='+')
            
            resultado_text_limite.delete("1.0", ctk.END)
            resultado_text_limite.insert(ctk.END, f"O limite da função pela {direcao.lower()} é: {limite}")

    except Exception as e:
        messagebox.showerror("Erro", "Ocorreu um erro ao calcular o limite. Verifique sua entrada.")

def raiz():
    global entradaraiz, entradaindice, resultado_text_raiz
    try:
        numero = float(entradaraiz.get())
        indice_input = entradaindice.get()  # Capturando a entrada do índice
        if not indice_input:
            raise ValueError("Índice não fornecido")
        indice = int(indice_input)
        
        if indice == 2:
            tolerancia = 1e-10
            x_val = numero / 2  # estimativa inicial 
            
            while True:
                raiz_value = 0.5 * (x_val + numero / x_val)
                if abs(raiz_value - x_val) < tolerancia:
                    break
                x_val = raiz_value
        else: 
            raiz_value = pow(numero, 1/indice)
        
        resultado_text_raiz.delete("1.0", ctk.END)
        resultado_text_raiz.insert(ctk.END, f"A raíz {indice}-ésima de {numero} é: {raiz_value:.4}")
    except ValueError:
        messagebox.showerror("Erro", "Por favor, forneça um índice e/ou número válido para calcular a raiz.")

def numerical_roots(sym_expr, var, lower, upper, num_points=500):
    func_num = sp.lambdify(var, sym_expr, 'numpy')
    sample_points = np.linspace(lower, upper, num_points)
    roots = []
    for i in range(len(sample_points) - 1):
        a = sample_points[i]
        b = sample_points[i + 1]
        fa = func_num(a)
        fb = func_num(b)
        if fa == 0:
            if lower <= a <= upper and not any(abs(a - r) < 1e-5 for r in roots):
                roots.append(a)
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
    global resultado_text_grafico, entrada_grafico, intervalo, show_points_var
    try:
        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'legend.fontsize': 12
        })
        
        x = sp.symbols('x')
        func_str = entrada_grafico.get()  # Exemplo: "sin(x)*x**2, cos(x)"
        func_list = [sp.sympify(f.strip()) for f in func_str.split(',')]
        func_numeric_list = [sp.lambdify(x, func, 'numpy') for func in func_list]

        intervalo_str = intervalo.get()
        lower, upper = map(float, intervalo_str.split(','))
        x_vals = np.linspace(lower, upper, 800)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        result_text = ""
        
        for func, func_numeric in zip(func_list, func_numeric_list):
            y_vals = func_numeric(x_vals)
            ax.plot(x_vals, y_vals, label=f'${sp.latex(func)}$', linewidth=2.5)
            
            # --- NOVO: Cálculo de Assíntotas ---
            # Assíntotas verticais
            try:
                vertical_asymptotes = sp.singularities(func, x)
                vertical_asymptotes = [asy for asy in vertical_asymptotes if asy.is_real]
                for asy in vertical_asymptotes:
                    asy_val = float(asy.evalf())
                    if lower < asy_val < upper:
                        ax.axvline(asy_val, color='magenta', linestyle='--', linewidth=1.5, label=f'Assíntota x={asy_val:.2f}')
                        result_text += f'Assíntota vertical em x = {asy_val:.2f}\n'
            except Exception:
                pass
            
            # Assíntotas horizontais
            try:
                lim_neg = sp.limit(func, x, -sp.oo)
                lim_pos = sp.limit(func, x, sp.oo)
                if lim_neg.is_real and not sp.oo == lim_neg:
                    lim_neg_val = float(lim_neg.evalf())
                    ax.axhline(lim_neg_val, color='cyan', linestyle='--', linewidth=1.5, label=f'Assíntota y={lim_neg_val:.2f}')
                    result_text += f'Assíntota horizontal em y = {lim_neg_val:.2f} (limite em -∞)\n'
                if lim_pos.is_real and not sp.oo == lim_pos:
                    lim_pos_val = float(lim_pos.evalf())
                    ax.axhline(lim_pos_val, color='cyan', linestyle='--', linewidth=1.5, label=f'Assíntota y={lim_pos_val:.2f}')
                    result_text += f'Assíntota horizontal em y = {lim_pos_val:.2f} (limite em +∞)\n'
            except Exception:
                pass
            # --- Fim das Assíntotas ---
            
            fprime = sp.diff(func, x)
            fsecond = sp.diff(fprime, x)
            
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
            
            if show_points_var.get():
                for p in cp:
                    y_p = float(func.subs(x, p).evalf())
                    try:
                        fsecond_val = float(fsecond.subs(x, p).evalf())
                    except Exception:
                        fsecond_val = None
                    
                    if fsecond_val is not None:
                        if fsecond_val < 0:
                            point_type = "Máximo"
                            color = "#e41a1c"
                            marker = "^"
                            offset = (0.4, 0.4)
                        elif fsecond_val > 0:
                            point_type = "Mínimo"
                            color = "#4daf4a"
                            marker = "v"
                            offset = (0.4, -0.4)
                        else:
                            point_type = "Sela"
                            color = "#ff7f00"
                            marker = "D"
                            offset = (0.4, 0.4)
                    else:
                        point_type = "Crítico"
                        color = "#984ea3"
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
            
            # --- NOVO: Intervalos de Crescimento e Decrescimento ---
            growth_points = [lower] + cp + [upper]
            growth_points = sorted(growth_points)
            for i in range(len(growth_points) - 1):
                mid = (growth_points[i] + growth_points[i+1]) / 2
                try:
                    derivative_mid = float(fprime.subs(x, mid).evalf())
                except Exception:
                    derivative_mid = None
                if derivative_mid is not None:
                    if derivative_mid > 0:
                        result_text += f'Função crescente em ({growth_points[i]:.2f}, {growth_points[i+1]:.2f})\n'
                    elif derivative_mid < 0:
                        result_text += f'Função decrescente em ({growth_points[i]:.2f}, {growth_points[i+1]:.2f})\n'
                    else:
                        result_text += f'Função constante em ({growth_points[i]:.2f}, {growth_points[i+1]:.2f})\n'
            # --- Fim dos intervalos ---
            
            result_text += "\n"
        
        ax.axhline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.axvline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title('Gráfico das Funções', fontsize=18, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        resultado_text_grafico.delete("1.0", ctk.END)
        resultado_text_grafico.insert(ctk.END, result_text + "\nGráfico plotado com sucesso!")
        
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico. Verifique sua entrada.\n")
def calcular_dominio(func, x):
    try:
        dominio = sp.calculus.util.continuous_domain(func, x, sp.S.Reals)
        return dominio
    except Exception as e:
        try:
            sample_points = np.linspace(-100, 100, 1000)
            valid_points = []
            for val in sample_points:
                try:
                    result = func.subs(x, val)
                    if result.is_real and not result.has(sp.oo, sp.zoo, sp.nan):
                        valid_points.append(val)
                except Exception:
                    continue
            if valid_points:
                return sp.Interval(min(valid_points), max(valid_points))
            else:
                return "Domínio não determinado"
        except Exception as e2:
<<<<<<< HEAD
            return f"Erro ao calcular o domínio:"
=======
            return f"Erro ao calcular o domínio"
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4

def calcular_imagem(func, x, dominio):
    try:
        func_str = str(func)
        if func_str.strip() in ['sin(x)', 'cos(x)']:
            return "[-1, 1]"
        if any(trig in func_str for trig in ['tan', 'cot']):
            return "Todos os reais (exceto singularidades)"
        if any(trig in func_str for trig in ['sec', 'csc']):
            return "(-∞, -1] ∪ [1, ∞)"
        
        y_values = []
        try:
            deriv = sp.diff(func, x)
            critical_points = sp.solve(deriv, x)
            for cp in critical_points:
                try:
                    cp_val = float(cp.evalf())
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

        y_values_clean = []
        for y in y_values:
            try:
                y_num = float(y)
                if not any(abs(y_num - float(existing)) < 1e-5 for existing in y_values_clean):
                    y_values_clean.append(y_num)
            except Exception:
                y_values_clean.append(y)
        
        if y_values_clean:
            min_val = min(y_values_clean)
            max_val = max(y_values_clean)
            return f"[{min_val:.4f}, {max_val:.4f}]"
        else:
            return "Imagem indefinida (não foram encontrados valores válidos)"
    except Exception as e:
<<<<<<< HEAD
        return f"Erro ao calcular a imagem:"
=======
        return f"Erro ao calcular a imagem"
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4

def calculo_dominio_imagem():
    global resultado_text_dom, entradadom
    try:
        func_str = entradadom.get()
        x = sp.symbols('x')
        try:
            func = sp.sympify(func_str)
        except Exception as e:
            resultado_text_dom.delete("1.0", ctk.END)
<<<<<<< HEAD
            resultado_text_dom.insert(ctk.END, f"Erro ao interpretar a função:")
=======
            resultado_text_dom.insert(ctk.END, f"Erro ao interpretar a função")
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4
            return
        
        dominio = calcular_dominio(func, x)
        if isinstance(dominio, str) and "Erro" in dominio:
            imagem = "Não foi possível calcular a imagem devido ao domínio inválido."
        else:
            imagem = calcular_imagem(func, x, dominio)
        
        resultado = f"""Resultados:
========================
Função: {func_str}
Domínio: {dominio}
Imagem: {imagem}
========================"""
        resultado_text_dom.delete("1.0", ctk.END)
        resultado_text_dom.insert(ctk.END, resultado)
    except Exception as e:
<<<<<<< HEAD
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular domínio e imagem:")
=======
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular domínio e imagem")
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4

def calculo_integral():
    global resultado_text_integral, entrada_integrais, entrada_limite_inf, entrada_limite_sup
    try:
        func_str = entrada_integrais.get()
        x = sp.symbols('x')
        func = sp.sympify(func_str)
        
        limite_inf_str = entrada_limite_inf.get().strip()
        limite_sup_str = entrada_limite_sup.get().strip()

        if limite_inf_str and limite_sup_str:
            limite_inf = float(sp.sympify(limite_inf_str))
            limite_sup = float(sp.sympify(limite_sup_str))
            integral_def = sp.integrate(func, (x, limite_inf, limite_sup))
            resultado_text_integral.delete("1.0", ctk.END)
            resultado_text_integral.insert(ctk.END, f"A integral definida da função de {limite_inf} a {limite_sup} é: {integral_def}\n")
        else:
            integral = sp.integrate(func, x)
            resultado_text_integral.delete("1.0", ctk.END)
            resultado_text_integral.insert(ctk.END, f"A integral indefinida da função é: {integral} + C\n")
    except Exception as e:
<<<<<<< HEAD
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a integral:")
=======
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a integral")
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4

def plot_func_tangente():
    try:
        x = sp.Symbol('x')
        func_str = entradaderiv.get()
        func = sp.sympify(func_str)
        point = float(sp.sympify(entradaponto.get()))
        derivada = sp.diff(func, x)
        coef_angular = derivada.subs(x, point)
        reta = func.subs(x, point) + coef_angular * (x - point)
        func_num = sp.lambdify(x, func, "numpy")
        reta_num = sp.lambdify(x, reta, "numpy")
        x_vals = np.linspace(-10, 10, 400)
        plt.figure()
        plt.plot(x_vals, func_num(x_vals), label=f"f(x) = {func_str}")
        plt.plot(x_vals, reta_num(x_vals), label=f"Tangente em x = {point}")
        plt.axhline(0, color='red', lw=0.8)
        plt.axvline(0, color='red', lw=0.8)
        plt.title('Gráfico das Funções')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
<<<<<<< HEAD
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico:")
=======
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico")
>>>>>>> bb775b5cbeb0656f8e02d3b3c5dac483d104bce4

def exemplo_raiz():
    example_text = ("Exemplo de Raiz Quadrada:\n"
        "Número: 256\n"
        "Definição: A raiz quadrada de um número é um valor que, quando multiplicado por si mesmo, "
        "resulta no número original.\n"
        "Cálculo: A raiz quadrada de 256 é 16, pois 16 * 16 = 256.\n"
        "Propriedades: A raiz quadrada de um número positivo é sempre um número positivo. "
        "Neste caso, a raiz quadrada de 256 é um valor exato e inteiro, 16.")
    resultado_text_raiz.delete("1.0", ctk.END)
    resultado_text_raiz.insert(ctk.END, example_text)

def exemplo_dominio_imagem():
    example_text = (
        "Exemplo de Domínio e Imagem:\n"
        "Função: f(x) = 1/(x-2)\n"
        "Domínio: Todos os valores de x, exceto x=2. Isso porque a função se torna indefinida quando x=2, "
        "pois resultaria em uma divisão por zero.\n"
        "Imagem: Todos os valores reais, exceto f(x)=0. A função nunca toca o eixo x, "
        "pois não há valor de x que faça a função igual a zero."
    )
    resultado_text_dom.delete("1.0", ctk.END)
    resultado_text_dom.insert(ctk.END, example_text)

def exemplo_limite():
    example_text = (
        "Exemplo de Limite:\n"
        "Função: f(x) = (x^2 - 1)/(x - 1)\n"
        "Para calcular o limite de f(x) quando x tende a 1, simplificamos a função:\n"
        "f(x) = (x + 1) para x ≠ 1.\n"
        "Então, o limite de f(x) quando x tende a 1 é 2.\n"
        "Lembre-se de que o limite se refere ao valor que a função se aproxima à medida que x se aproxima de 1."
    )
    resultado_text_limite.delete("1.0", ctk.END)
    resultado_text_limite.insert(ctk.END, example_text)

def exemplo_derivada():
    example_text = (
        "Exemplo de Derivada e Tangente:\n"
        "Função: f(x) = x^2\n"
        "Derivada: f'(x) = 2x. Isso representa a inclinação da função em qualquer ponto x.\n"
        "No ponto x=3, f'(3) = 6. Isso significa que a inclinação da tangente à curva no ponto (3, f(3)) é 6.\n"
        "A equação da reta tangente é dada por: y = f(3) + f'(3)*(x - 3)\n"
        "Neste caso, a reta tangente é y = 9 + 6(x - 3), simplificando: y = 6x - 9."
    )
    resultado_text_deriv.delete("1.0", ctk.END)
    resultado_text_deriv.insert(ctk.END, example_text)

def exemplo_integral():
    example_text = (
        "Exemplo de Integral:\n"
        "Função: f(x) = x^2\n"
        "Integral Indefinida: ∫x^2 dx = (1/3)x^3 + C, onde C é a constante de integração.\n"
        "Integral Definida de 0 a 2: ∫(de 0 a 2) x^2 dx = [(1/3)x^3] de 0 a 2 = (8/3) - 0 = 8/3.\n"
        "Isso representa a área sob a curva de f(x) entre x=0 e x=2."
    )
    resultado_text_integral.delete("1.0", ctk.END)
    resultado_text_integral.insert(ctk.END, example_text)

# =============================================================================
#   NOVAS FUNÇÕES DE INTERFACE COM CUSTOMTKINTER
# =============================================================================

# Classe que sobrescreve o método get() para realizar substituições usando regex.
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

def botao(pai, func, texto):
    btn = ctk.CTkButton(pai, text=texto, command=func, width=200, corner_radius=5, font=font)
    btn.pack(pady=10, padx=10)

# =============================================================================
#   CLASSE PRINCIPAL DA APLICAÇÃO
# =============================================================================

class InitialPage(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Página Inicial")
        self.geometry("400x200")
        self.configure(padx=20, pady=20)

        # Botão para abrir a calculadora
        open_calculator_btn = ctk.CTkButton(self, text="Abrir Calculadora DDX", command=self.open_calculator)
        open_calculator_btn.pack(pady=20)

        # Botão para abrir o manual
        manual_btn = ctk.CTkButton(self, text="Abrir Manual do DDX", 
                                    command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing'))
        manual_btn.pack(pady=20)

    def open_calculator(self):
        self.destroy()  # Fecha a página inicial
        app = App()  # Inicia a aplicação principal
        app.mainloop()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora DDX")
        self.geometry("1200x900")
        self.configure(padx=20, pady=20)
        self.create_widgets()

    def create_widgets(self):
        # Cria uma Tabview para organizar as funcionalidades
        tabview = ctk.CTkTabview(self, width=1100, height=700)
        tabview.pack(padx=20, pady=20, fill="both", expand=True)
        tabview.add("Domínio e Imagem")
        tabview.add("Derivadas")
        tabview.add("Limites")
        tabview.add("Raiz")
        tabview.add("Gráficos")
        tabview.add("Integrais")
        tabview.add("Manual")

        # --------------------- Aba: Domínio e Imagem ---------------------
        frame_dom = tabview.tab("Domínio e Imagem")
        global entradadom, resultado_text_dom
        entradadom = labeled_input(frame_dom, "Expressão:")
        botao(frame_dom, calculo_dominio_imagem, "Calcular")
        botao(frame_dom, exemplo_dominio_imagem, "Exemplo")
        resultado_text_dom = ctk.CTkTextbox(frame_dom, height=200, width=600, font=font)
        resultado_text_dom.pack(padx=10, pady=10)

        # --------------------- Aba: Derivadas ---------------------
        frame_deriv = tabview.tab("Derivadas")
        global entradaderiv, entradaponto, resultado_text_deriv
        entradaderiv = labeled_input(frame_deriv, "Função:")
        entradaponto = labeled_input(frame_deriv, "Ponto:")
        botao(frame_deriv, calculo_derivada, "Calcular")
        botao(frame_deriv, exemplo_derivada, "Exemplo")
        botao(frame_deriv, plot_func_tangente, "Plotar Tangente")
        resultado_text_deriv = ctk.CTkTextbox(frame_deriv, height=200, width=600, font=font)
        resultado_text_deriv.pack(padx=10, pady=10)

        # --------------------- Aba: Limites ---------------------
        frame_lim = tabview.tab("Limites")
        global entradalimit, entradavar, entradatend, direcao_var, resultado_text_limite
        entradalimit = labeled_input(frame_lim, "Função:")
        entradavar = labeled_input(frame_lim, "Variável:")
        entradatend = labeled_input(frame_lim, "Tendendo a:")
        direcao_var = ctk.StringVar(value="Ambos")
        option_menu = ctk.CTkOptionMenu(frame_lim, variable=direcao_var, values=["Esquerda", "Direita", "Ambos"])
        option_menu.pack(pady=10)
        botao(frame_lim, calculo_limite, "Calcular")
        botao(frame_lim, exemplo_limite, "Exemplo")
        resultado_text_limite = ctk.CTkTextbox(frame_lim, height=200, width=600, font=font)
        resultado_text_limite.pack(padx=10, pady=10)

        # --------------------- Aba: Raiz ---------------------
        frame_raiz = tabview.tab("Raiz")
        global entradaraiz, entradaindice, resultado_text_raiz
        entradaraiz = labeled_input(frame_raiz, "Número:")
        entradaindice = labeled_input(frame_raiz, "Índice:")
        botao(frame_raiz, raiz, "Calcular")
        botao(frame_raiz, exemplo_raiz, "Exemplo")
        resultado_text_raiz = ctk.CTkTextbox(frame_raiz, height=200, width=600, font=font)
        resultado_text_raiz.pack(padx=10, pady=10)

        # --------------------- Aba: Gráficos ---------------------
        frame_graf = tabview.tab("Gráficos")
        global entrada_grafico, intervalo, show_points_var, resultado_text_grafico
        entrada_grafico = labeled_input(frame_graf, "Função:")
        intervalo = labeled_input(frame_graf, "Intervalo (ex: -10,10):")
        show_points_var = ctk.BooleanVar(value=False)
        chk = ctk.CTkCheckBox(frame_graf, text="Mostrar pontos críticos e de inflexão", variable=show_points_var)
        chk.pack(pady=10)
        botao(frame_graf, plot_grafico, "Plotar")
        resultado_text_grafico = ctk.CTkTextbox(frame_graf, height=200, width=600, font=font)
        resultado_text_grafico.pack(padx=10, pady=10)

        # --------------------- Aba: Integrais ---------------------
        frame_int = tabview.tab("Integrais")
        global entrada_integrais, entrada_limite_inf, entrada_limite_sup, resultado_text_integral
        entrada_integrais = labeled_input(frame_int, "Função:")
        entrada_limite_inf = labeled_input(frame_int, "Limite inferior:")
        entrada_limite_sup = labeled_input(frame_int, "Limite superior:")
        botao(frame_int, calculo_integral, "Calcular")
        botao(frame_int, exemplo_integral, "Exemplo")
        resultado_text_integral = ctk.CTkTextbox(frame_int, height=200, width=600, font=font)
        resultado_text_integral.pack(padx=10, pady=10)

        # --------------------- Aba: Manual ---------------------
        frame_man = tabview.tab("Manual"
                                )
        manual_btn = ctk.CTkButton(frame_man, text="Manual do DDX", 
                                   command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing'))
        manual_btn.pack(pady=10)
# =============================================================================
#   EXECUÇÃO DA APLICAÇÃO
# =============================================================================

if __name__ == "__main__":
    initial_page = InitialPage()
    initial_page.mainloop()