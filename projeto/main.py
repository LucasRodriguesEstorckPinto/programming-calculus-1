import customtkinter as ctk
import tkinter.messagebox as messagebox
import webbrowser
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image


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


def formatar_intervalo(intervalo):
    """
    Formata um intervalo de SymPy em uma notação matemática mais amigável.
    
    Args:
        intervalo: Objeto de intervalo do SymPy
        
    Returns:
        String formatada com notação matemática amigável
    """
    if not isinstance(intervalo, (sp.Interval, sp.Union)):
        return str(intervalo)
    
    # Caso seja uma união de intervalos
    if isinstance(intervalo, sp.Union):
        intervalos_formatados = [formatar_intervalo(i) for i in intervalo.args]
        return " ∪ ".join(intervalos_formatados)
    
    # Formatação para um único intervalo
    esquerda = intervalo.left
    direita = intervalo.right
    
    # Formatação do limite esquerdo
    if esquerda == -sp.oo:
        inicio = "(-∞"
    else:
        valor_esq = float(esquerda.evalf())
        # Arredondar para melhor legibilidade
        if abs(valor_esq - round(valor_esq)) < 1e-10:
            valor_esq = int(round(valor_esq))
        elif abs(valor_esq) < 1000:
            valor_esq = round(valor_esq, 4)
            # Remover zeros desnecessários
            valor_esq = str(valor_esq).rstrip('0').rstrip('.') if '.' in str(valor_esq) else valor_esq
        
        if intervalo.left_open:
            inicio = f"({valor_esq}"
        else:
            inicio = f"[{valor_esq}"
    
    # Formatação do limite direito
    if direita == sp.oo:
        fim = "+∞)"
    else:
        valor_dir = float(direita.evalf())
        # Arredondar para melhor legibilidade
        if abs(valor_dir - round(valor_dir)) < 1e-10:
            valor_dir = int(round(valor_dir))
        elif abs(valor_dir) < 1000:
            valor_dir = round(valor_dir, 4)
            # Remover zeros desnecessários
            valor_dir = str(valor_dir).rstrip('0').rstrip('.') if '.' in str(valor_dir) else valor_dir
        
        if intervalo.right_open:
            fim = f"{valor_dir})"
        else:
            fim = f"{valor_dir}]"
    
    return f"{inicio}, {fim}"

def formatar_conjunto(conjunto):
    """
    Formata um conjunto matemático de forma amigável.
    
    Args:
        conjunto: Conjunto matemático (pode ser intervalo, união, ou texto)
        
    Returns:
        String formatada para fácil compreensão
    """
    if isinstance(conjunto, str):
        return conjunto
    
    # Formatação para intervalos e uniões
    if isinstance(conjunto, (sp.Interval, sp.Union)):
        return formatar_intervalo(conjunto)
    
    # Caso seja ℝ (conjunto dos reais)
    if conjunto == sp.S.Reals:
        return "ℝ (todos os números reais)"
    
    # Para outros tipos de conjuntos, retorna a string do objeto
    return str(conjunto)

def explicar_dominio(dominio, func_str):
    """
    Fornece uma explicação em linguagem simples sobre o domínio.
    
    Args:
        dominio: Domínio calculado
        func_str: String da função
        
    Returns:
        Explicação em texto do domínio
    """
    # Casos especiais comuns
    if dominio == sp.S.Reals:
        return "Todos os números reais (não há restrições)"
    
    if isinstance(dominio, str):
        return dominio
    
    # Para funções com raiz quadrada
    if "sqrt" in func_str or "**0.5" in func_str:
        return f"O domínio é {formatar_conjunto(dominio)}. As expressões dentro das raízes devem ser não-negativas."
    
    # Para funções racionais
    if "/" in func_str or "**-1" in func_str:
        return f"O domínio é {formatar_conjunto(dominio)}. O denominador não pode ser zero."
    
    # Para funções logarítmicas
    if "log" in func_str:
        return f"O domínio é {formatar_conjunto(dominio)}. Os argumentos de logaritmos devem ser positivos."
    
    # Para funções trigonométricas inversas
    if any(trig in func_str for trig in ["asin", "acos", "atan", "arcsin", "arccos", "arctan"]):
        if "asin" in func_str or "arcsin" in func_str:
            return f"O domínio é {formatar_conjunto(dominio)}. Para arcsen, o argumento deve estar entre -1 e 1."
        elif "acos" in func_str or "arccos" in func_str:
            return f"O domínio é {formatar_conjunto(dominio)}. Para arccos, o argumento deve estar entre -1 e 1."
        else:
            return f"O domínio é {formatar_conjunto(dominio)}."
    
    # Caso genérico
    return f"O domínio da função é {formatar_conjunto(dominio)}."

def explicar_imagem(imagem, func_str):
    """
    Fornece uma explicação em linguagem simples sobre a imagem.
    
    Args:
        imagem: Imagem calculada
        func_str: String da função
        
    Returns:
        Explicação em texto da imagem
    """
    # Casos especiais
    if isinstance(imagem, str):
        return imagem
    
    # Para funções com "comportamento especial"
    if "sin" in func_str or "cos" in func_str:
        if imagem == sp.Interval(-1, 1):
            return "A imagem é [-1, 1]. Funções seno e cosseno variam apenas entre -1 e 1."
    
    if "tan" in func_str:
        if "ℝ" in str(imagem):
            return "A imagem é ℝ (todos os números reais). A função tangente pode assumir qualquer valor real."
    
    # Para funções polinomiais
    try:
        if "x**" in func_str:
            if "x**2" in func_str and imagem == sp.Interval(0, sp.oo):
                return "A imagem é [0, +∞). Funções quadráticas do tipo ax² + bx + c (com a > 0) têm imagem limitada inferiormente."
            if imagem == sp.S.Reals:
                return "A imagem é ℝ (todos os números reais). Esta função polinomial assume todos os valores reais."
    except Exception:
        pass
    
    # Caso genérico
    return f"A imagem da função é {formatar_conjunto(imagem)}."

def calcular_dominio(func, x):
    """
    Calcula o domínio de uma função usando métodos analíticos e numéricos.
    
    Args:
        func: Expressão SymPy da função
        x: Variável SymPy
        
    Returns:
        Domínio da função como um objeto Interval ou uma descrição textual
    """
    
    try:
        # Trata casos especiais de funções trigonométricas
        if func.has(sp.tan):
            # Para tangente, o domínio exclui os pontos onde cos(x) = 0
            restricoes_trig = sp.solve(sp.cos(x) == 0, x)
            pontos_exclusao = [ponto for ponto in restricoes_trig if ponto.is_real]
            return f"ℝ - {{x | x = {sp.pi/2} + n·{sp.pi}, n ∈ ℤ}}"
            
        # Tenta o método analítico primeiro
        dominio = sp.calculus.util.continuous_domain(func, x, sp.S.Reals)
        return dominio
    except Exception:
        # Se o método analítico falhar, usa uma abordagem numérica
        try:
            # Identifica pontos problemáticos (divisão por zero, raízes negativas, etc.)
            denominadores = []
            restricoes = []
            
            # Verifica funções trigonométricas
            if func.has(sp.tan):
                restricoes.extend(sp.solve(sp.cos(x) == 0, x))
            if func.has(sp.cot):
                restricoes.extend(sp.solve(sp.sin(x) == 0, x))
            if func.has(sp.sec):
                restricoes.extend(sp.solve(sp.cos(x) == 0, x))
            if func.has(sp.csc):
                restricoes.extend(sp.solve(sp.sin(x) == 0, x))
            
            # Extrair denominadores para verificar divisões por zero
            for atom in func.atoms(sp.Pow):
                if atom.exp < 0:  # Potência negativa indica um denominador
                    denominadores.append(atom.base)
            
            # Adiciona expressões dentro de raízes quadradas para verificar domínio
            raizes = []
            for atom in func.atoms(sp.Pow):
                if atom.exp.is_real and atom.exp < 1 and atom.exp > 0 and atom.base.has(x):  # Raízes (não apenas quadradas)
                    # Para raízes de índice par, o argumento deve ser não-negativo
                    if 1/atom.exp % 2 == 0:
                        raizes.append(atom.base)
            
            # Adicionar expressões dentro de logaritmos
            logs = []
            for atom in func.atoms(sp.log):
                if atom.args[0].has(x):
                    logs.append(atom.args[0])
            
            # Resolver restrições de domínio
            for denom in denominadores:
                restricoes.extend(sp.solve(denom, x))
            
            for raiz in raizes:
                restricoes.extend(sp.solve(raiz < 0, x))
                
            for log_arg in logs:
                restricoes.extend(sp.solve(log_arg <= 0, x))
            
            # Se não conseguirmos identificar restrições analiticamente, usamos amostragem numérica
            if not restricoes:
                # Amostragem adaptativa para funções com comportamento diferente em diferentes regiões
                sample_regions = [
                    (-1000, -100, 50),  # (início, fim, número de pontos)
                    (-100, -1, 50),
                    (-1, 1, 100),        # Mais pontos próximo à origem
                    (1, 100, 50),
                    (100, 1000, 50)
                ]
                
                sample_points = []
                for inicio, fim, num in sample_regions:
                    sample_points.extend(np.linspace(inicio, fim, num))
                    
                valid_points = []
                for val in sample_points:
                    try:
                        result = float(func.subs(x, val).evalf())
                        if np.isfinite(result) and not np.isnan(result):
                            valid_points.append(val)
                    except Exception:
                        continue
                
                if valid_points:
                    return sp.Interval(min(valid_points), max(valid_points))
                else:
                    return "Domínio não determinado"
            else:
                # Converter as restrições em intervalos
                restricoes_reais = []
                for ponto in restricoes:
                    try:
                        valor = float(ponto.evalf())
                        if np.isfinite(valor) and not np.isnan(valor):
                            restricoes_reais.append(valor)
                    except Exception:
                        continue
                
                restricoes_reais.sort()
                
                # Remover duplicatas (dentro de uma tolerância numérica)
                if restricoes_reais:
                    pontos_unicos = [restricoes_reais[0]]
                    for ponto in restricoes_reais[1:]:
                        if abs(ponto - pontos_unicos[-1]) > 1e-10:
                            pontos_unicos.append(ponto)
                    restricoes_reais = pontos_unicos
                
                # Verificar cada intervalo entre as restrições
                intervalos_validos = []
                pontos_teste = restricoes_reais + [-float('inf'), float('inf')]
                pontos_teste.sort()
                
                for i in range(len(pontos_teste) - 1):
                    if pontos_teste[i+1] - pontos_teste[i] <= 1e-10:
                        continue
                        
                    # Determinar ponto de teste
                    if pontos_teste[i] == -float('inf'):
                        ponto_medio = pontos_teste[i+1] - 1
                    elif pontos_teste[i+1] == float('inf'):
                        ponto_medio = pontos_teste[i] + 1
                    else:
                        ponto_medio = (pontos_teste[i] + pontos_teste[i+1]) / 2
                        
                    try:
                        result = float(func.subs(x, ponto_medio).evalf())
                        if np.isfinite(result) and not np.isnan(result):
                            if pontos_teste[i] == -float('inf'):
                                intervalos_validos.append(sp.Interval(-sp.oo, pontos_teste[i+1], right_open=True))
                            elif pontos_teste[i+1] == float('inf'):
                                intervalos_validos.append(sp.Interval(pontos_teste[i], sp.oo, left_open=True))
                            else:
                                intervalos_validos.append(sp.Interval(pontos_teste[i], pontos_teste[i+1], left_open=True, right_open=True))
                    except Exception:
                        continue
                
                # Verificar intervalos adicionais para funções periódicas
                if func.has(sp.tan) or func.has(sp.cot) or func.has(sp.sec) or func.has(sp.csc):
                    return "Função periódica com domínio descontínuo. Verifique pontos de singularidade."
                
                if intervalos_validos:
                    # Unir intervalos adjacentes
                    if len(intervalos_validos) > 1:
                        return sp.Union(*intervalos_validos)
                    else:
                        return intervalos_validos[0]
                else:
                    return "Domínio não determinado"
        except Exception as e:
            return f"Erro ao calcular o domínio: {e}"

def calcular_imagem(func, x, dominio):
    """
    Calcula a imagem de uma função usando análise numérica e analítica,
    com detecção aprimorada de comportamento assintótico.
    
    Args:
        func: Expressão SymPy da função
        x: Variável SymPy
        dominio: Domínio da função calculado anteriormente
        
    Returns:
        Imagem da função como um intervalo ou uma descrição textual
    """
    
    try:
        # Detecção de funções especiais com imagens conhecidas
        func_str = str(func)
        
        # Funções trigonométricas com imagens conhecidas
        if func_str.strip() in ['sin(x)', 'cos(x)']:
            return sp.Interval(-1, 1)
        if 'tan(x)' == func_str.strip() or 'cot(x)' == func_str.strip():
            return "ℝ (exceto singularidades)"
        if 'sec(x)' == func_str.strip() or 'csc(x)' == func_str.strip():
            return sp.Union(sp.Interval(-sp.oo, -1), sp.Interval(1, sp.oo))
        
        # Funções exponenciais
        if 'exp(x)' == func_str.strip():
            return sp.Interval(0, sp.oo)
        if 'exp(-x)' == func_str.strip():
            return sp.Interval(0, 1)
            
        # Funções logarítmicas
        if 'log(x)' == func_str.strip():
            return sp.Interval(-sp.oo, sp.oo)
        
        # Polinômios de grau ímpar
        try:
            poly = sp.Poly(func, x)
            if poly.degree() % 2 == 1:  # Grau ímpar
                return "ℝ"
        except Exception:
            pass
            
        # Detecção de funções racionais
        try:
            numer, denom = func.as_numer_denom()
            if denom != 1:  # É uma função racional
                # Grau do numerador > grau do denominador indica comportamento assintótico infinito
                try:
                    poly_numer = sp.Poly(numer, x)
                    poly_denom = sp.Poly(denom, x)
                    if poly_numer.degree() > poly_denom.degree():
                        return "ℝ"
                except Exception:
                    pass
        except Exception:
            pass
            
        # Coletar pontos críticos e extremidades do domínio
        y_values = []
        
        # Pontos críticos (onde a derivada é zero)
        try:
            deriv = sp.diff(func, x)
            critical_points = sp.solve(deriv, x)
            
            for cp in critical_points:
                try:
                    cp_val = float(cp.evalf())
                    # Verificar se o ponto crítico está no domínio
                    if isinstance(dominio, sp.Interval) or isinstance(dominio, sp.Union):
                        if dominio.contains(cp_val):
                            y_val = func.subs(x, cp).evalf()
                            if y_val.is_real and not y_val.has(sp.oo, sp.zoo, sp.nan):
                                y_values.append(float(y_val))
                except Exception:
                    continue
        except Exception:
            pass
        
        # Calcular limites nos extremos do domínio e em pontos de descontinuidade
        try:
            # Extremidades do domínio
            limites_externos = []
            
            if isinstance(dominio, sp.Interval):
                intervalos = [dominio]
            elif isinstance(dominio, sp.Union):
                intervalos = dominio.args
            else:
                intervalos = []
                
            for intervalo in intervalos:
                # Limite no início do intervalo
                if intervalo.left == -sp.oo:
                    try:
                        limite_esq = sp.limit(func, x, -sp.oo)
                        if limite_esq.is_real and not limite_esq.has(sp.zoo, sp.nan):
                            if limite_esq == sp.oo:
                                return "Imagem inclui +∞"
                            elif limite_esq == -sp.oo:
                                return "Imagem inclui -∞"
                            else:
                                limites_externos.append(float(limite_esq))
                    except Exception:
                        # Testar valores muito negativos
                        for test_val in [-1e3, -1e4, -1e5, -1e6]:
                            try:
                                y_val = float(func.subs(x, test_val).evalf())
                                if np.isfinite(y_val):
                                    limites_externos.append(y_val)
                                elif y_val > 1e10:
                                    return "Imagem inclui +∞"
                                elif y_val < -1e10:
                                    return "Imagem inclui -∞"
                            except Exception:
                                continue
                else:
                    # Limite à direita do ponto esquerdo
                    if intervalo.left_open:
                        try:
                            limite_esq_dir = sp.limit(func, x, intervalo.left, '+')
                            if limite_esq_dir.is_real:
                                if limite_esq_dir == sp.oo:
                                    return "Imagem inclui +∞"
                                elif limite_esq_dir == -sp.oo:
                                    return "Imagem inclui -∞"
                                else:
                                    limites_externos.append(float(limite_esq_dir))
                        except Exception:
                            # Aproximação numérica
                            ponto_teste = float(intervalo.left) + 1e-6
                            try:
                                y_val = float(func.subs(x, ponto_teste).evalf())
                                if np.isfinite(y_val):
                                    limites_externos.append(y_val)
                                elif y_val > 1e10:
                                    return "Imagem inclui +∞"
                                elif y_val < -1e10:
                                    return "Imagem inclui -∞"
                            except Exception:
                                pass
                    else:
                        # Ponto esquerdo fechado
                        try:
                            y_val = float(func.subs(x, intervalo.left).evalf())
                            if np.isfinite(y_val):
                                limites_externos.append(y_val)
                            elif y_val > 1e10:
                                return "Imagem inclui +∞"
                            elif y_val < -1e10:
                                return "Imagem inclui -∞"
                        except Exception:
                            pass
                
                # Limite no fim do intervalo
                if intervalo.right == sp.oo:
                    try:
                        limite_dir = sp.limit(func, x, sp.oo)
                        if limite_dir.is_real and not limite_dir.has(sp.zoo, sp.nan):
                            if limite_dir == sp.oo:
                                return "Imagem inclui +∞"
                            elif limite_dir == -sp.oo:
                                return "Imagem inclui -∞"
                            else:
                                limites_externos.append(float(limite_dir))
                    except Exception:
                        # Testar valores muito positivos
                        for test_val in [1e3, 1e4, 1e5, 1e6]:
                            try:
                                y_val = float(func.subs(x, test_val).evalf())
                                if np.isfinite(y_val):
                                    limites_externos.append(y_val)
                                elif y_val > 1e10:
                                    return "Imagem inclui +∞"
                                elif y_val < -1e10:
                                    return "Imagem inclui -∞"
                            except Exception:
                                continue
                else:
                    # Limite à esquerda do ponto direito
                    if intervalo.right_open:
                        try:
                            limite_dir_esq = sp.limit(func, x, intervalo.right, '-')
                            if limite_dir_esq.is_real:
                                if limite_dir_esq == sp.oo:
                                    return "Imagem inclui +∞"
                                elif limite_dir_esq == -sp.oo:
                                    return "Imagem inclui -∞"
                                else:
                                    limites_externos.append(float(limite_dir_esq))
                        except Exception:
                            # Aproximação numérica
                            ponto_teste = float(intervalo.right) - 1e-6
                            try:
                                y_val = float(func.subs(x, ponto_teste).evalf())
                                if np.isfinite(y_val):
                                    limites_externos.append(y_val)
                                elif y_val > 1e10:
                                    return "Imagem inclui +∞"
                                elif y_val < -1e10:
                                    return "Imagem inclui -∞"
                            except Exception:
                                pass
                    else:
                        # Ponto direito fechado
                        try:
                            y_val = float(func.subs(x, intervalo.right).evalf())
                            if np.isfinite(y_val):
                                limites_externos.append(y_val)
                            elif y_val > 1e10:
                                return "Imagem inclui +∞"
                            elif y_val < -1e10:
                                return "Imagem inclui -∞"
                        except Exception:
                            pass
            
            y_values.extend(limites_externos)
        except Exception as e:
            pass
            
        # Encontrar pontos de descontinuidade e verificar limites laterais
        try:
            # Procurar por possíveis pontos de descontinuidade (onde o denominador é zero)
            numer, denom = func.as_numer_denom()
            descontinuidades = sp.solve(denom, x)
            
            for d in descontinuidades:
                try:
                    d_val = float(d.evalf())
                    
                    # Limite pela esquerda
                    try:
                        limite_esq = sp.limit(func, x, d_val, '-')
                        if limite_esq.is_real:
                            if limite_esq == sp.oo:
                                return "Imagem inclui +∞"
                            elif limite_esq == -sp.oo:
                                return "Imagem inclui -∞"
                    except Exception:
                        # Aproximação numérica
                        for distancia in [1e-3, 1e-4, 1e-5, 1e-6]:
                            try:
                                y_val = float(func.subs(x, d_val - distancia).evalf())
                                if abs(y_val) > 1e10:
                                    if y_val > 0:
                                        return "Imagem inclui +∞"
                                    else:
                                        return "Imagem inclui -∞"
                                break
                            except Exception:
                                continue
                    
                    # Limite pela direita
                    try:
                        limite_dir = sp.limit(func, x, d_val, '+')
                        if limite_dir.is_real:
                            if limite_dir == sp.oo:
                                return "Imagem inclui +∞"
                            elif limite_dir == -sp.oo:
                                return "Imagem inclui -∞"
                    except Exception:
                        # Aproximação numérica
                        for distancia in [1e-3, 1e-4, 1e-5, 1e-6]:
                            try:
                                y_val = float(func.subs(x, d_val + distancia).evalf())
                                if abs(y_val) > 1e10:
                                    if y_val > 0:
                                        return "Imagem inclui +∞"
                                    else:
                                        return "Imagem inclui -∞"
                                break
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception:
            pass
            
        # Amostragem numérica dentro do domínio com detecção de comportamento assintótico
        try:
            if isinstance(dominio, sp.Interval):
                # Para um único intervalo
                amostragem_intervalo(dominio, func, x, y_values)
            elif isinstance(dominio, sp.Union):
                # Para múltiplos intervalos
                for intervalo in dominio.args:
                    amostragem_intervalo(intervalo, func, x, y_values)
            else:
                # Fallback para domínio não reconhecido
                sample_points = np.linspace(-1000, 1000, 2000)
                for val in sample_points:
                    try:
                        y_val = func.subs(x, val).evalf()
                        if y_val.is_real:
                            if y_val == sp.oo:
                                return "Imagem inclui +∞"
                            elif y_val == -sp.oo:
                                return "Imagem inclui -∞"
                            elif not y_val.has(sp.oo, sp.zoo, sp.nan):
                                y_values.append(float(y_val))
                    except Exception:
                        continue
        except Exception:
            pass
            
        # Verificar singularidades e comportamento assintótico
        if not y_values:
            return "Imagem indefinida (não foram encontrados valores válidos)"
            
        # Remover valores duplicados e determinar o intervalo
        y_values_clean = []
        for y in y_values:
            if np.isfinite(y) and not np.isnan(y):
                # Evitar duplicatas
                if not any(abs(y - existing) < 1e-5 for existing in y_values_clean):
                    y_values_clean.append(y)
        
        if not y_values_clean:
            return "Imagem indefinida (não foram encontrados valores válidos)"
            
        min_val = min(y_values_clean)
        max_val = max(y_values_clean)
        
        # Verificar crescimento assintótico
        # Testar valores para verificar tendência ao infinito
        infinito_positivo = False
        infinito_negativo = False
        
        # Testar em faixas de valores grandes positivos
        for teste_x in [1e2, 1e3, 1e4, 1e5, 1e6]:
            try:
                y_val = float(func.subs(x, teste_x).evalf())
                if y_val > 1e10:
                    infinito_positivo = True
                    break
                elif y_val < -1e10:
                    infinito_negativo = True
                    break
            except Exception:
                continue
                
        # Testar em faixas de valores grandes negativos
        for teste_x in [-1e2, -1e3, -1e4, -1e5, -1e6]:
            try:
                y_val = float(func.subs(x, teste_x).evalf())
                if y_val > 1e10:
                    infinito_positivo = True
                    break
                elif y_val < -1e10:
                    infinito_negativo = True
                    break
            except Exception:
                continue
                
        # Verificar se há muita variação nos valores encontrados (indício de imagem não limitada)
        if max_val - min_val > 1e6:
            # Provavelmente a imagem não é limitada
            if infinito_positivo and infinito_negativo:
                return "ℝ"
            elif infinito_positivo:
                if min_val < -1e3:
                    return "ℝ"
                else:
                    return sp.Interval(min_val, sp.oo)
            elif infinito_negativo:
                if max_val > 1e3:
                    return "ℝ"
                else:
                    return sp.Interval(-sp.oo, max_val)
            else:
                # Verificar se os valores crescem muito rapidamente
                return sp.Interval(min_val, max_val)
        else:
            # A imagem parece ser limitada
            if infinito_positivo:
                return sp.Interval(min_val, sp.oo)
            elif infinito_negativo:
                return sp.Interval(-sp.oo, max_val)
            else:
                return sp.Interval(min_val, max_val)
    except Exception as e:
        return f"Erro ao calcular a imagem: {e}"

def amostragem_intervalo(intervalo, func, x, y_values):
    """
    Realiza amostragem em um intervalo com verificação de comportamento assintótico.
    
    Args:
        intervalo: Intervalo SymPy para amostragem
        func: Função a ser avaliada
        x: Variável simbólica
        y_values: Lista para armazenar os valores encontrados
        
    Returns:
        True se detectar comportamento infinito, False caso contrário
    """
    
    try:
        a = float(intervalo.left) if intervalo.left != -sp.oo else -1000
        b = float(intervalo.right) if intervalo.right != sp.oo else 1000
        
        # Garantir que a amostragem seja razoável
        if b - a > 2000:
            # Usar amostragem mais esparsa para intervalos grandes
            faixas = [
                (a, a + (b-a)*0.1, 50),
                (a + (b-a)*0.1, a + (b-a)*0.9, 100),
                (a + (b-a)*0.9, b, 50)
            ]
            
            for inicio, fim, num in faixas:
                sample_points = np.linspace(inicio, fim, num)
                for val in sample_points:
                    try:
                        y_val = func.subs(x, val).evalf()
                        if y_val.is_real:
                            if y_val == sp.oo:
                                return True
                            elif y_val == -sp.oo:
                                return True
                            elif not y_val.has(sp.oo, sp.zoo, sp.nan):
                                y_values.append(float(y_val))
                    except Exception:
                        continue
        else:
            # Para intervalos menores, usar mais pontos
            num_points = min(int((b - a) * 10), 1000)
            num_points = max(num_points, 50)  # Pelo menos 50 pontos
            
            sample_points = np.linspace(a, b, num_points)
            
            for val in sample_points:
                try:
                    y_val = func.subs(x, val).evalf()
                    if y_val.is_real:
                        if y_val == sp.oo:
                            return True
                        elif y_val == -sp.oo:
                            return True
                        elif not y_val.has(sp.oo, sp.zoo, sp.nan):
                            float_val = float(y_val)
                            if abs(float_val) > 1e10:
                                return True
                            y_values.append(float_val)
                except Exception:
                    continue
        
        # Verificação adicional em pontos próximos de descontinuidades
        # Tentar identificar possíveis descontinuidades
        try:
            numer, denom = func.as_numer_denom()
            descontinuidades = sp.solve(denom, x)
            
            for d in descontinuidades:
                try:
                    d_val = float(d.evalf())
                    # Verificar se a descontinuidade está no intervalo
                    if a <= d_val <= b:
                        # Verificar comportamento próximo à descontinuidade
                        for dist in [1e-2, 1e-3, 1e-4, 1e-5]:
                            for lado in [-1, 1]:  # -1 para esquerda, 1 para direita
                                ponto = d_val + lado * dist
                                try:
                                    y_val = float(func.subs(x, ponto).evalf())
                                    if abs(y_val) > 1e10:
                                        return True
                                    if np.isfinite(y_val) and not np.isnan(y_val):
                                        y_values.append(y_val)
                                except Exception:
                                    continue
                except Exception:
                    continue
        except Exception:
            pass
        
        return False
    except Exception:
        return False

def calculo_dominio_imagem():
    """
    Função principal que processa a entrada do usuário e calcula domínio e imagem.
    """
    global resultado_text_dom, entradadom
    try:
        func_str = entradadom.get()
        x = sp.symbols('x')
        
        # Substituir notações comuns para melhorar a interpretação
        func_str = func_str.replace("^", "**")
        func_str = func_str.replace("sen", "sin")
        func_str = func_str.replace("arctg", "atan")
        func_str = func_str.replace("arcsen", "asin")
        func_str = func_str.replace("arccos", "acos")
        
        try:
            func = sp.sympify(func_str)
        except Exception as e:
            resultado_text_dom.delete("1.0", ctk.END)
            resultado_text_dom.insert(ctk.END, f"Erro ao interpretar a função: {e}")
            return
        
        # Calcular domínio e imagem
        dominio = calcular_dominio(func, x)
        
        if isinstance(dominio, str) and "Erro" in dominio:
            imagem = "Não foi possível calcular a imagem devido ao domínio inválido."
        else:
            imagem = calcular_imagem(func, x, dominio)
        
        # Formatar o resultado para exibição
        dominio_explicado = explicar_dominio(dominio, func_str)
        imagem_explicada = explicar_imagem(imagem, func_str)
        
        resultado = f"""Resultados:
========================
Função: {func_str}

Domínio: {formatar_conjunto(dominio)}
{dominio_explicado}

Imagem: {formatar_conjunto(imagem)}
{imagem_explicada}
========================"""
        resultado_text_dom.delete("1.0", ctk.END)
        resultado_text_dom.insert(ctk.END, resultado)
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular domínio e imagem: {e}")

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
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a integral:")

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
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico:")


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

def abrir_explicacao_integral():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Integrais")
    janela_explicacao.geometry("500x300")

    texto_explicacao = """A integral de uma função representa a área sob a curva dessa função em um determinado intervalo. 
Ela é usada para calcular áreas, volumes e resolver problemas físicos como trabalho e deslocamento.

Fonte: Stewart, James. Cálculo. 8ª edição."""

    label_texto = ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left")
    label_texto.pack(padx=20, pady=20)

    botao_fechar = ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy)
    botao_fechar.pack(pady=10)

def abrir_explicacao_derivada():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Derivadas")
    janela_explicacao.geometry("500x300")

    texto_explicacao = """A derivada de uma função representa a taxa de variação dessa função em um determinado ponto.
Ela é usada para calcular velocidades, acelerações e resolver problemas físicos como otimização e crescimento populacional.

Fonte: Stewart, James. Cálculo. 8ª edição."""

    label_texto = ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left")
    label_texto.pack(padx=20, pady=20)

    botao_fechar = ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy)
    botao_fechar.pack(pady=10)

def abrir_explicacao_limites():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Limites")
    janela_explicacao.geometry("500x300")

    texto_explicacao = """O limite de uma função descreve o comportamento dessa função à medida que a variável independente se aproxima de um determinado valor.
Ele é usado para definir derivadas, integrais e resolver problemas envolvendo continuidade e comportamento assintótico.

Fonte: Stewart, James. Cálculo. 8ª edição."""

    label_texto = ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left")
    label_texto.pack(padx=20, pady=20)

    botao_fechar = ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy)
    botao_fechar.pack(pady=10)

def abrir_explicacao_dominios():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Limites")
    janela_explicacao.geometry("500x300")

    texto_explicacao = """O domínio de uma função é o conjunto de todos os valores de entrada para os quais a função está definida.
A imagem de uma função é o conjunto de todos os valores de saída que a função pode assumir.
Eles são usados para entender o comportamento e as restrições de funções em diversos contextos matemáticos e aplicados.

Fonte: Stewart, James. Cálculo. 8ª edição."""

    label_texto = ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left")
    label_texto.pack(padx=20, pady=20)

    botao_fechar = ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy)
    botao_fechar.pack(pady=10)

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

        # Botão para abrir a explicação
        botao_explicacao = ctk.CTkButton(frame_dom, text="O que são Domínios e Imagens?", command=abrir_explicacao_dominios)
        botao_explicacao.pack(pady=10)

        entradadom = labeled_input(frame_dom, "Expressão:")
        botao(frame_dom, calculo_dominio_imagem, "Calcular")
        botao(frame_dom, exemplo_dominio_imagem, "Exemplo")
        resultado_text_dom = ctk.CTkTextbox(frame_dom, height=200, width=600, font=font)
        resultado_text_dom.pack(padx=10, pady=10)

        # --------------------- Aba: Derivadas ---------------------
        frame_deriv = tabview.tab("Derivadas")
        global entradaderiv, entradaponto, resultado_text_deriv
        
        # Botão para abrir a explicação
        botao_explicacao = ctk.CTkButton(frame_deriv, text="O que é Derivada?", command=abrir_explicacao_derivada)
        botao_explicacao.pack(pady=10)

        entradaderiv = labeled_input(frame_deriv, "Função:")
        entradaponto = labeled_input(frame_deriv, "Ponto:")
        botao(frame_deriv, calculo_derivada, "Calcular")
        botao(frame_deriv, exemplo_derivada, "Exemplo")
        botao(frame_deriv, plot_func_tangente, "Plotar Tangente")
        resultado_text_deriv = ctk.CTkTextbox(frame_deriv, height=200, width=600, font=font)
        resultado_text_deriv.pack(padx=10, pady=10)

        #Carregar a imagem
        imagem_deriv = ctk.CTkImage(light_image=Image.open("deriva.png"), size=(200, 100))

        # Criar label para exibir a imagem
        label_imagem = ctk.CTkLabel(frame_deriv, image=imagem_deriv, text="")
        label_imagem.pack(pady=10)

        # --------------------- Aba: Limites ---------------------
        frame_lim = tabview.tab("Limites")
        global entradalimit, entradavar, entradatend, direcao_var, resultado_text_limite

        # Botão para abrir a explicação
        botao_explicacao = ctk.CTkButton(frame_lim, text="O que são Limites?", command=abrir_explicacao_limites)
        botao_explicacao.pack(pady=10)

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

        #Carregar a imagem
        imagem_lim = ctk.CTkImage(light_image=Image.open("limit.png"), size=(200, 100))

        # Criar label para exibir a imagem
        label_imagem = ctk.CTkLabel(frame_lim, image=imagem_lim, text="")
        label_imagem.pack(pady=10)

        # --------------------- Aba: Raiz ---------------------
        frame_raiz = tabview.tab("Raiz")
        global entradaraiz, entradaindice, resultado_text_raiz
        entradaraiz = labeled_input(frame_raiz, "Número:")
        entradaindice = labeled_input(frame_raiz, "Índice:")
        botao(frame_raiz, raiz, "Calcular")
        botao(frame_raiz, exemplo_raiz, "Exemplo")
        resultado_text_raiz = ctk.CTkTextbox(frame_raiz, height=200, width=600, font=font)
        resultado_text_raiz.pack(padx=10, pady=10)

        #Carregar a imagem
        imagem_raiz = ctk.CTkImage(light_image=Image.open("raiz.png"), size=(200, 100))

        # Criar label para exibir a imagem
        label_imagem = ctk.CTkLabel(frame_raiz, image=imagem_raiz, text="")
        label_imagem.pack(pady=10)

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
        
        # Botão para abrir a explicação
        botao_explicacao = ctk.CTkButton(frame_int, text="O que é Integral?", command=abrir_explicacao_integral)
        botao_explicacao.pack(pady=10)

        entrada_integrais = labeled_input(frame_int, "Função:")
        entrada_limite_inf = labeled_input(frame_int, "Limite inferior:")
        entrada_limite_sup = labeled_input(frame_int, "Limite superior:")
        botao(frame_int, calculo_integral, "Calcular")
        botao(frame_int, exemplo_integral, "Exemplo")
        resultado_text_integral = ctk.CTkTextbox(frame_int, height=200, width=600, font=font)
        resultado_text_integral.pack(padx=10, pady=10)

        #Carregar a imagem
        imagem_integral = ctk.CTkImage(light_image=Image.open("integral.png"), size=(300, 200))

        # Criar label para exibir a imagem
        label_imagem = ctk.CTkLabel(frame_int, image=imagem_integral, text="")
        label_imagem.pack(pady=10)

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