import customtkinter as ctk
import tkinter.messagebox as messagebox
import webbrowser
import sympy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
from PIL import Image
from functools import lru_cache
from scipy.optimize import fsolve
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sympy import Interval, Union, S, solve, log , Complement , FiniteSet, oo
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy import S, Interval, Union, log, Pow, FiniteSet
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solvers import solve
from tkinter import filedialog
from scipy.interpolate import interp1d


dados_x = None
dados_y = None
interpolar_var = None
botao_plot_dados = None
check_interpolar = None


matplotlib.use("TkAgg")
# Configuração do tema (dark, light ou system)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Fonte padrão para os widgets
font = ("Segoe UI", 14)

# Configurações globais
sp.init_printing()
x = sp.symbols('x')
n = sp.symbols('n', integer=True)


def calculo_derivadas_parciais():
    global entradafuncparcial, entradavarparcial, resultado_text_parcial
    try:
        func_str = entradafuncparcial.get()
        var_str = entradavarparcial.get().strip()

        # Identifica variáveis presentes na função
        variaveis = sorted(set(re.findall(r"[a-zA-Z]+", func_str)))
        vars_sympy = sp.symbols(" ".join(variaveis))
        expr = sp.sympify(func_str)

        resultado_text_parcial.delete("1.0", ctk.END)

        if var_str:  # Derivada parcial específica
            var = sp.Symbol(var_str)
            derivada = sp.diff(expr, var)
            resultado_text_parcial.insert(ctk.END, f"∂f/∂{var_str} = {derivada}\n")
        else:  # Todas as derivadas parciais
            for var in vars_sympy:
                derivada = sp.diff(expr, var)
                resultado_text_parcial.insert(ctk.END, f"∂f/∂{var} = {derivada}\n")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular a derivada parcial.\n{e}")

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


# Funções auxiliares
def validar_entrada_grafico(func_str, intervalo_str):
    """Valida a entrada do usuário para funções e intervalo."""
    if not func_str or not intervalo_str:
        raise ValueError("Entrada de função ou intervalo vazia.")
    func_list = [f.strip() for f in func_str.split(',')]
    try:
        lower, upper = map(float, intervalo_str.split(','))
        if lower >= upper:
            raise ValueError("O limite inferior deve ser menor que o superior.")
    except ValueError as e:
        raise ValueError(f"Intervalo inválido: {str(e)}")
    for f in func_list:
        try:
            sp.sympify(f)
        except sp.SympifyError:
            raise ValueError(f"Função inválida: {f}")
    return func_list, lower, upper

@lru_cache(maxsize=128)
def calcular_derivadas(func, x):
    """Calcula as derivadas primeira e segunda da função."""
    fprime = sp.diff(func, x)
    fsecond = sp.diff(fprime, x)
    return fprime, fsecond

def encontrar_assintota_obliqua(func, x):
    """Encontra assíntotas oblíquas, se existirem."""
    numer, denom = func.as_numer_denom()
    deg_numer = sp.degree(numer, gen=x)
    deg_denom = sp.degree(denom, gen=x)
    if deg_numer - deg_denom == 1:
        coef = sp.limit(func/denom, x, sp.oo)
        intercept = sp.limit(func - coef*x, x, sp.oo)
        return coef, intercept
    return None, None

def numerical_roots(expr, var, a, b, num_points=500):
    """Encontra raízes numéricas da expressão no intervalo [a, b]."""
    x_vals = np.linspace(a, b, num_points)
    roots = []
    expr_func = sp.lambdify(var, expr, 'numpy')
    for i in range(len(x_vals)-1):
        try:
            val1, val2 = expr_func(x_vals[i]), expr_func(x_vals[i+1])
            if np.isfinite(val1) and np.isfinite(val2) and np.sign(val1) * np.sign(val2) < 0:
                root_array = fsolve(lambda x: float(expr_func(x)) if np.isfinite(float(expr_func(x))) else 0,
                                    (x_vals[i] + x_vals[i+1])/2)
                root = float(root_array[0])  # Pegamos o primeiro elemento da lista
                if a <= root <= b and not any(abs(root - r) < 1e-6 for r in roots):
                    roots.append(root)
        except Exception:
            continue
    return sorted(roots)

def ajustar_amostragem(lower, upper, num_points_base=200):
    """Ajusta o número de pontos de amostragem com base no intervalo."""
    if upper - lower > 100:
        return np.linspace(lower, upper, num_points_base)
    return np.linspace(lower, upper, min(num_points_base * 2, 800))

def carregar_arquivo_pontos():
    global dados_x, dados_y, botao_plot_dados, check_interpolar, resultado_text_grafico

    file_path = filedialog.askopenfilename(filetypes=[("Arquivos de Texto", "*.txt")])
    if not file_path:
        return

    try:
        with open(file_path, 'r') as file:
            linhas = file.readlines()
            dados_lidos = []
            for linha in linhas:
                linha = re.sub(r'[,\t]+', ' ', linha.strip())  # trata vírgulas e tabs
                partes = linha.split()
                if len(partes) >= 2:
                    dados_lidos.append([float(partes[0]), float(partes[1])])

        dados = np.array(dados_lidos)
        dados_x = dados[:, 0]
        dados_y = dados[:, 1]

        if len(dados_x) < 2:
            raise ValueError("Mínimo de dois pontos necessários.")

        check_interpolar.pack(pady=5, anchor="w")
        botao_plot_dados.pack(pady=5, padx=5, anchor="w")
        resultado_text_grafico.insert(ctk.END, "\nArquivo carregado com sucesso. Pontos prontos para plotagem.\n")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        messagebox.showerror("Erro ao importar arquivo", f"Erro ao processar arquivo: {str(e)}")

def plotar_dados_importados():
    global dados_x, dados_y, grafico_canvas, grafico_toolbar, frame_grafico_container
    if dados_x is None or dados_y is None:
        return

    try:
        if frame_grafico_container:
            for widget in frame_grafico_container.winfo_children():
                widget.destroy()

        ordenados = sorted(zip(dados_x, dados_y), key=lambda par: par[0])
        x_ord, y_ord = zip(*ordenados)
        x_ord = np.array(x_ord)
        y_ord = np.array(y_ord)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Interpolação: só se houver ao menos 4 pontos
        if interpolar_var.get():
            if len(x_ord) >= 4:
                try:
                    f_interp = interp1d(x_ord, y_ord, kind='cubic')
                    x_interp = np.linspace(x_ord[0], x_ord[-1], 500)
                    y_interp = f_interp(x_interp)
                    ax.plot(x_interp, y_interp, label="Interpolado", linewidth=2.5, color='cyan', zorder=4)
                    resultado_text_grafico.insert(ctk.END, "\nCurva interpolada plotada com sucesso.\n")
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    messagebox.showwarning("Interpolação falhou", f"Erro ao interpolar: {str(e)}")
            else:
                resultado_text_grafico.insert(ctk.END, "\n⚠️ Pelo menos 4 pontos são necessários para interpolação cúbica.\n")

        # Pontos
        ax.scatter(x_ord, y_ord, color='red', s=60, zorder=5, label="Pontos")

        # Eixos
        ax.axhline(0, color='black', lw=1.2, linestyle='dashed')
        ax.axvline(0, color='black', lw=1.2, linestyle='dashed')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Gráfico dos dados importados")
        ax.legend()
        plt.tight_layout()

        # Embed no Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        grafico_canvas = canvas

        toolbar = NavigationToolbar2Tk(canvas, frame_grafico_container)
        toolbar.update()
        toolbar.pack()
        grafico_toolbar = toolbar

        resultado_text_grafico.insert(ctk.END, "\nGráfico plotado.\n")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        messagebox.showerror("Erro", f"Erro ao plotar os pontos: {str(e)}")


# Função principal do gráfico
def plot_grafico():
    global resultado_text_grafico, entrada_grafico, intervalo, show_points_var, grafico_canvas, grafico_toolbar, frame_grafico_container
    try:
        # Limpar gráfico anterior
        if frame_grafico_container:
            for widget in frame_grafico_container.winfo_children():
                widget.destroy()

        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'legend.fontsize': 12
        })

        func_str = entrada_grafico.get()
        intervalo_str = intervalo.get()
        func_list, lower, upper = validar_entrada_grafico(func_str, intervalo_str)

        func_sym_list = [sp.sympify(f) for f in func_list]
        func_numeric_list = [sp.lambdify(x, func, 'numpy') for func in func_sym_list]

        x_vals = ajustar_amostragem(lower, upper)

        fig, ax = plt.subplots(figsize=(10, 6))
        result_text = ""

        for i, (func_sym, func_numeric) in enumerate(zip(func_sym_list, func_numeric_list)):
            y_vals = func_numeric(x_vals)
            ax.plot(x_vals, y_vals, label=f'${sp.latex(func_sym)}$', linewidth=2.5, color=f'C{i}')

            # Assíntotas verticais
            try:
                if func_sym.has(sp.tan):
                    n_vals = range(int(lower/sp.pi)-1, int(upper/sp.pi)+2)
                    vertical_asymptotes = [sp.pi/2 + n*sp.pi for n in n_vals]
                else:
                    vertical_asymptotes = sp.singularities(func_sym, x)
                vertical_asymptotes = [asy for asy in vertical_asymptotes if asy.is_real]
                for asy in vertical_asymptotes:
                    asy_val = float(asy.evalf())
                    if lower < asy_val < upper:
                        ax.axvline(asy_val, color='magenta', linestyle='--', linewidth=2)
                        result_text += f'Assíntota vertical em x = {asy_val:.2f}\n'
            except Exception as e:
                print(f"Erro ao calcular assíntotas verticais: {e}")

            # Assíntotas horizontais
            try:
                lim_neg = sp.limit(func_sym, x, -sp.oo)
                lim_pos = sp.limit(func_sym, x, sp.oo)
                for lim, side in [(lim_neg, '-∞'), (lim_pos, '+∞')]:
                    if lim.is_real and not lim.has(sp.oo, sp.zoo):
                        lim_val = float(lim.evalf())
                        ax.axhline(lim_val, color='cyan', linestyle='--', linewidth=2)
                        result_text += f'Assíntota horizontal em y = {lim_val:.2f} (limite em {side})\n'
            except Exception as e:
                print(f"Erro ao calcular assíntotas horizontais: {e}")

            # Assíntotas oblíquas
            try:
                coef, intercept = encontrar_assintota_obliqua(func_sym, x)
                if coef is not None and intercept is not None:
                    ax.axline((0, float(intercept)), slope=float(coef), color='orange', linestyle='--')
                    result_text += f'Assíntota oblíqua: y = {float(coef):.2f}x + {float(intercept):.2f}\n'
            except Exception as e:
                print(f"Erro ao calcular assíntota oblíqua: {e}")

            # Pontos críticos e inflexões
            fprime, fsecond = calcular_derivadas(func_sym, x)
            cp = numerical_roots(fprime, x, lower, upper)
            ip = numerical_roots(fsecond, x, lower, upper)

            if show_points_var.get():
                colors = ['#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#377eb8']
                markers = ['^', 'v', 'D', 'o', 's']
                for p, color, marker in zip(cp, colors[:len(cp)], markers[:len(cp)]):
                    try:
                        y_p = float(func_sym.subs(x, p).evalf())
                        fsecond_val = float(fsecond.subs(x, p).evalf())
                        point_type = "Máximo" if fsecond_val < 0 else "Mínimo" if fsecond_val > 0 else "Sela"
                    except Exception:
                        y_p = float(func_sym.subs(x, p).evalf())
                        point_type, color, marker = "Crítico", '#984ea3', 'o'

                    ax.scatter(p, y_p, color=color, marker=marker, s=100, edgecolors='black', zorder=6)
                    ax.annotate(
                        f'{point_type}\n({p:.2f}, {y_p:.2f})',
                        xy=(p, y_p), xytext=(p + 0.4, y_p + (0.4 if point_type == "Máximo" else -0.4)),
                        textcoords='data', fontsize=10, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none'),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5), zorder=7
                    )
                    result_text += f'{point_type} em ({p:.2f}, {y_p:.2f})\n'

                for p in ip:
                    y_p = float(func_sym.subs(x, p).evalf())
                    ax.scatter(p, y_p, color='#377eb8', marker='s', s=100, edgecolors='black', zorder=6)
                    ax.annotate(
                        f'Inflexão\n({p:.2f}, {y_p:.2f})',
                        xy=(p, y_p), xytext=(p + 0.4, y_p + 0.4), textcoords='data',
                        fontsize=10, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#377eb8', ec='none'),
                        arrowprops=dict(arrowstyle='-|>', color='#377eb8', lw=1.5), zorder=7
                    )
                    result_text += f'Inflexão em ({p:.2f}, {y_p:.2f})\n'
            else:
                result_text += "Pontos não explicitados (checkbox desativado).\n"

            growth_points = sorted([lower] + cp + [upper])
            for i in range(len(growth_points) - 1):
                mid = (growth_points[i] + growth_points[i+1]) / 2
                try:
                    derivative_mid = float(fprime.subs(x, mid).evalf())
                    if derivative_mid > 0:
                        result_text += f'Crescimento em [{growth_points[i]:.2f}, {growth_points[i+1]:.2f}]\n'
                    elif derivative_mid < 0:
                        result_text += f'Decrescimento em [{growth_points[i]:.2f}, {growth_points[i+1]:.2f}]\n'
                    else:
                        result_text += f'Constante em [{growth_points[i]:.2f}, {growth_points[i+1]:.2f}]\n'
                except Exception:
                    continue

        ax.axhline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.axvline(0, color='black', lw=1.2, linestyle='dashed', zorder=3)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title('Gráfico das Funções', fontsize=18, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_grafico_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        grafico_canvas = canvas

        toolbar = NavigationToolbar2Tk(canvas, frame_grafico_container)
        toolbar.update()
        toolbar.pack()
        grafico_toolbar = toolbar

        resultado_text_grafico.delete("1.0", ctk.END)
        resultado_text_grafico.insert(ctk.END, result_text + "\nGráfico plotado com sucesso!")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao plotar o gráfico: {str(e)}")


 

# Funções auxiliares
def validar_entrada(func_str):
    pattern = r'^[a-zA-Z0-9\s+\-*/().^sincoslogexp]+$'
    if not re.match(pattern, func_str):
        raise ValueError("Entrada inválida: use apenas caracteres matemáticos válidos.")
    return func_str.replace("^", "**").replace("sen", "sin").replace("arctg", "atan").replace("arcsen", "asin").replace("arccos", "acos")

def formatar_intervalo(intervalo):
    if not isinstance(intervalo, (sp.Interval, sp.Union, str)):
        return str(intervalo)

    if isinstance(intervalo, str):
        return intervalo

    if isinstance(intervalo, sp.Union):
        intervalos_formatados = [formatar_intervalo(i) for i in intervalo.args]
        return " ∪ ".join(intervalos_formatados)

    esquerda = intervalo.left
    direita = intervalo.right

    if esquerda == -sp.oo:
        inicio = "(-∞"
    else:
        valor_esq = float(esquerda.evalf())
        if abs(valor_esq - round(valor_esq)) < 1e-10:
            valor_esq = int(round(valor_esq))
        elif abs(valor_esq) < 1000:
            valor_esq_str = str(round(valor_esq, 4))
            valor_esq = valor_esq_str.rstrip('0').rstrip('.') if '.' in valor_esq_str else valor_esq_str

        inicio = f"[{valor_esq}" if not intervalo.left_open else f"({valor_esq}"

    if direita == sp.oo:
        fim = "+∞)"
    else:
        valor_dir = float(direita.evalf())
        if abs(valor_dir - round(valor_dir)) < 1e-10:
            valor_dir = int(round(valor_dir))
        elif abs(valor_dir) < 1000:
            valor_dir_str = str(round(valor_dir, 4))
            valor_dir = valor_dir_str.rstrip('0').rstrip('.') if '.' in valor_dir_str else valor_dir_str

        fim = f"{valor_dir}]" if not intervalo.right_open else f"{valor_dir})"

    return f"{inicio}, {fim}"


def formatar_conjunto(conjunto):
    if isinstance(conjunto, str):
        return conjunto

    if isinstance(conjunto, (sp.Interval, sp.Union)):
        return formatar_intervalo(conjunto)

    if conjunto == sp.S.Reals:
        return "ℝ (todos os números reais)"

    # Para floats, converte para string simples
    if isinstance(conjunto, float):
        return str(conjunto)

    return str(conjunto)



def explicar_dominio(dominio, func_str=""):
    """
    Gera uma explicação simbólica do domínio calculado.
    """
    if isinstance(dominio, str):
        return dominio  # Erro tratado anteriormente

    # ℝ completo
    if dominio == S.Reals:
        return "Todos os números reais (não há restrições na função)."

    explicacao = []

    if isinstance(dominio, Complement):
        conj_principal = dominio.args[0]
        conj_excluido = dominio.args[1]
        if isinstance(conj_excluido, FiniteSet):
            pontos = ", ".join([f"x ≠ {p}" for p in conj_excluido])
            explicacao.append(f"A função não está definida em {pontos} devido à presença de divisão por zero.")
        dominio = conj_principal

    if isinstance(dominio, Union):
        for intervalo in dominio.args:
            explicacao.append(f"Intervalo permitido: {intervalo}")
    elif isinstance(dominio, Interval):
        if dominio.left == 0 and dominio.right == oo:
            explicacao.append("O domínio é (0, +∞) pois o argumento do logaritmo ou denominador deve ser positivo.")
        elif dominio.left == 1 and dominio.right == oo:
            explicacao.append("O domínio é (1, +∞) pois o argumento de ln(x - 1) precisa ser maior que zero.")
        elif dominio.left == 3 and dominio.right == oo:
            explicacao.append("O domínio é [3, +∞) pois a raiz quadrada exige que a expressão dentro dela seja não-negativa.")
        else:
            explicacao.append(f"O domínio é {dominio} devido a restrições simbólicas da função.")

    elif isinstance(dominio, FiniteSet):
        explicacao.append(f"Domínio restrito a valores específicos: {', '.join(str(v) for v in dominio)}")

    return " ".join(explicacao)
def explicar_imagem(imagem, func_str):
    if isinstance(imagem, str):
        return imagem

    if "sin" in func_str or "cos" in func_str:
        if imagem == sp.Interval(-1, 1):
            return "A imagem é [-1, 1]. Funções seno e cosseno variam apenas entre -1 e 1."

    if "tan" in func_str:
        if "ℝ" in str(imagem):
            return "A imagem é ℝ (todos os números reais). A função tangente pode assumir qualquer valor real."

    try:
        if "x**" in func_str:
            if "x**2" in func_str and imagem == sp.Interval(0, sp.oo):
                return "A imagem é [0, +∞). Funções quadráticas do tipo ax² + bx + c (com a > 0) têm imagem limitada inferiormente."
            if imagem == sp.S.Reals:
                return "A imagem é ℝ (todos os números reais). Esta função polinomial assume todos os valores reais."
    except Exception:
        pass

    return f"A imagem da função é {formatar_conjunto(imagem)}."


def calcular_dominio(func, x):
    """
    Calcula o domínio de uma função simbólica real de uma variável.
    Identifica e trata: logaritmos, raízes pares e divisões.
    """
    try:
        dominio_total = S.Reals
        restricoes = []

        # 1. Tratamento de denominadores (divisões por zero)
        numerador, denominador = func.as_numer_denom()
        if denominador.has(x):
            zeros_denom = solve(denominador, x)
            for raiz in zeros_denom:
                if raiz.is_real:
                    dominio_total = dominio_total - FiniteSet(raiz)

        # 2. Tratamento de radicais pares (raiz par de número negativo)
        for atom in func.atoms(Pow):
            if atom.exp.is_Rational and atom.exp.q == 2:  # Expoente 1/2, 3/2, etc.
                base = atom.base
                if base.has(x):
                    cond = base >= 0
                    dominio_local = solve_univariate_inequality(cond, x, relational=False)
                    restricoes.append(dominio_local)

        # 3. Tratamento de logaritmos (argumento > 0)
        for expr in func.atoms(log):
            arg = expr.args[0]
            if arg.has(x):
                cond = arg > 0
                dominio_local = solve_univariate_inequality(cond, x, relational=False)
                restricoes.append(dominio_local)

        # 4. Restrições adicionais (caso sejam geradas inequações simbólicas)
        for r in restricoes:
            dominio_total = dominio_total.intersect(r)

        return dominio_total

    except Exception as e:
         return f"Erro ao calcular o domínio: {str(e)}"

def calcular_imagem(func, x, dominio):
    try:
        func_str = str(func)

        # Casos especiais
        if func_str.strip() in ['sin(x)', 'cos(x)']:
            return sp.Interval(-1, 1)
        if func_str.strip() in ['tan(x)', 'cot(x)']:
            return sp.S.Reals
        if func_str.strip() in ['sec(x)', 'csc(x)']:
            return sp.Union(sp.Interval.open(-sp.oo, -1), sp.Interval.open(1, sp.oo))
        if func_str.strip() == 'exp(x)':
            return sp.Interval.open(0, sp.oo)
        if func_str.strip() == 'exp(-x)':
            return sp.Interval.open(0, 1)
        if func_str.strip() == 'log(x)':
            return sp.S.Reals

        # Detectar racional do tipo f(x) = constante / g(x) + c
        try:
            numer, denom = func.as_numer_denom()

            if denom.has(x) and not denom.has(sp.sin, sp.cos, sp.tan):
                # Verificar se f(x) = (k / g(x)) + c
                deslocamento = 0
                racional_puro = False

                if numer.is_constant():
                    deslocamento = 0
                    racional_puro = True
                else:
                    # Tentar separar em forma (k/g(x)) + c
                    if isinstance(func, sp.Add):
                        for termo in func.args:
                            if not termo.has(x):
                                deslocamento += float(termo)
                            else:
                                num_t, den_t = termo.as_numer_denom()
                                if not num_t.is_constant():
                                    racional_puro = False
                                else:
                                    racional_puro = True
                    elif isinstance(func, sp.Sub):
                        termos = func.as_ordered_terms()
                        if len(termos) == 2:
                            if not termos[1].has(x):
                                deslocamento -= float(termos[1])
                                num_t, den_t = termos[0].as_numer_denom()
                                if num_t.is_constant():
                                    racional_puro = True

                if racional_puro:
                    return sp.Union(
                        sp.Interval.open(-sp.oo, deslocamento),
                        sp.Interval.open(deslocamento, sp.oo)
                    )
        except Exception:
            pass

        # Polinômio de grau ímpar assume todos os reais
        try:
            poly = sp.Poly(func, x)
            if poly.degree() % 2 == 1:
                return sp.S.Reals
        except Exception:
            pass

        # Amostragem numérica para casos gerais
        y_values = []

        try:
            deriv = sp.diff(func, x)
            critical_points = sp.solve(deriv, x)
            for cp in critical_points:
                if cp.is_real and not cp.has(sp.oo, -sp.oo):
                    cp_val = float(cp.evalf())
                    if isinstance(dominio, (sp.Interval, sp.Union)) and dominio.contains(cp_val):
                        y_val = func.subs(x, cp).evalf()
                        if y_val.is_real and not y_val.has(sp.oo, sp.zoo, sp.nan):
                            y_values.append(float(y_val))
        except Exception:
            pass

        if isinstance(dominio, sp.Interval):
            intervalos = [dominio]
        elif isinstance(dominio, sp.Union):
            intervalos = dominio.args
        elif isinstance(dominio, str):
            intervalos = [sp.Interval.open(-1000, 1000)]  # fallback
        else:
            intervalos = []

        for intervalo in intervalos:
            a = float(intervalo.left) if intervalo.left != -sp.oo else -1000
            b = float(intervalo.right) if intervalo.right != sp.oo else 1000

            num_points = min(int((b - a) * 10), 1000)
            num_points = max(num_points, 50)
            sample_points = np.linspace(a, b, num_points)

            for val in sample_points:
                try:
                    y_val = func.subs(x, val).evalf()
                    if y_val.is_real and not y_val.has(sp.oo, sp.zoo, sp.nan):
                        y_values.append(float(y_val))
                except Exception:
                    continue

        if not y_values:
            return "Imagem indefinida (não foram encontrados valores válidos)"

        min_val = min(y_values)
        max_val = max(y_values)

        return sp.Interval(min_val, max_val)

    except Exception as e:
        return f"Erro ao calcular a imagem: {str(e)}"


def calculo_dominio_imagem():
    """Função principal que processa a entrada do usuário e calcula domínio e imagem."""
    global resultado_text_dom, entradadom, grafico_label

    try:
        func_str = entradadom.get()
        func_str = validar_entrada(func_str)

        func = sp.sympify(func_str)

        # Calcular domínio e imagem
        dominio = calcular_dominio(func, x)

        if isinstance(dominio, str) and "Erro" in dominio:
            imagem = "Não foi possível calcular a imagem devido ao domínio inválido."
        else:
            imagem = calcular_imagem(func, x, dominio)

        # Formatar o resultado
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

        # Remover gráfico anterior, se existir
        if 'grafico_label' in globals() and grafico_label is not None:
            grafico_label.destroy()

        # Gerar novo gráfico
        try:
            x_vals = np.linspace(-10, 10, 1000)
            y_vals = [float(func.subs(x, val).evalf()) for val in x_vals if np.isfinite(float(func.subs(x, val).evalf()))]
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=func_str)
            plt.title(f"Gráfico de {func_str}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)
            plt.savefig("grafico.png")
            plt.close()

            # Carregar e exibir nova imagem
            img = ctk.CTkImage(Image.open("grafico.png"), size=(400, 300))
            grafico_label = ctk.CTkLabel(master=resultado_text_dom.master, image=img, text="")
            grafico_label.pack(pady=10, after=resultado_text_dom)

        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao calcular domínio e imagem: {str(e)}")

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

def aplicar_lhopital(f_str, g_str, ponto_str, direcao='Ambos'):
    from sympy import limit, sympify, diff, simplify
    f = sympify(f_str)
    g = sympify(g_str)
    ponto = sympify(ponto_str)
    passos = []

    def calcular(expr, lado):
        return limit(expr, x, ponto, dir=lado)

    lados = ["+"] if direcao in ["+"] else ["-"] if direcao == "-" else ["+", "-"]

    for lado in lados:
        passos.append(f"Analisando limite lateral: {'direita' if lado == '+' else 'esquerda'}")

        try:
            lim_f = calcular(f, lado)
            lim_g = calcular(g, lado)
            passos.append(f"  lim(x→{ponto}{lado}) {f} = {lim_f}")
            passos.append(f"  lim(x→{ponto}{lado}) {g} = {lim_g}")
        except Exception as e:
            passos.append(f"  Erro ao calcular limites: {e}")
            continue

        # Verifica se a forma é indeterminada válida para L'Hôpital
        formas_validas = [
    abs(lim_f.evalf()) < 1e-10 and abs(lim_g.evalf()) < 1e-10,
    lim_f.is_infinite and lim_g.is_infinite]


        if not any(formas_validas):
            passos.append("  ❌ A Regra de L’Hôpital NÃO se aplica — forma não é indeterminada.")
            continue

        # Aplicação iterativa
        passos.append("  ✅ Forma indeterminada detectada. Aplicando L'Hôpital:")
        i = 1
        num, den = f, g
        while i <= 10:
            num_deriv = diff(num, x)
            den_deriv = diff(den, x)
            passos.append(f"    Iteração {i}:")
            passos.append(f"      f'(x) = {num_deriv}")
            passos.append(f"      g'(x) = {den_deriv}")

            try:
                lim_num = calcular(num_deriv, lado)
                lim_den = calcular(den_deriv, lado)
                passos.append(f"      lim(x→{ponto}{lado}) f'(x) = {lim_num}")
                passos.append(f"      lim(x→{ponto}{lado}) g'(x) = {lim_den}")
            except Exception as e:
                passos.append(f"      Erro ao calcular limites das derivadas: {e}")
                break

            if lim_den != 0 and lim_den.is_number and lim_num.is_number:
                resultado = simplify(lim_num / lim_den)
                passos.append(f"      ✅ Resultado final após {i} iteração(ões): {resultado}")
                break

            num, den = num_deriv, den_deriv
            i += 1

            if i > 10:
                passos.append("      ❌ Número máximo de iterações atingido.")
                break

        passos.append("")

    return passos



def calculo_lhopital():
    global entrada_num, entrada_den, entrada_ponto, direcao_lhopital, resultado_text_lhopital
    try:
        num = entrada_num.get()
        den = entrada_den.get()
        ponto = entrada_ponto.get()
        direcao = direcao_lhopital.get()

        passos = aplicar_lhopital(num, den, ponto, direcao)
        resultado_text_lhopital.delete("1.0", ctk.END)
        for passo in passos:
            resultado_text_lhopital.insert(ctk.END, passo + "\n")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {e}")


def exemplo_lhopital(self):
    entrada_num.delete(0, ctk.END)
    entrada_den.delete(0, ctk.END)
    entrada_ponto.delete(0, ctk.END)
    entrada_num.insert(0, "sin(x)")
    entrada_den.insert(0, "x")
    entrada_ponto.insert(0, "0")

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

def exemplo_lhopital(self):
        entrada_num.delete(0, ctk.END)
        entrada_den.delete(0, ctk.END)
        entrada_ponto.delete(0, ctk.END)
        entrada_num.insert(0, "sin(x)")
        entrada_den.insert(0, "x")
        entrada_ponto.insert(0, "0")

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

def exemplo_derivada_parcial():
    entradafuncparcial.delete(0, ctk.END)
    entradavarparcial.delete(0, ctk.END)
    entradafuncparcial.insert(0, "x**2 * y + sin(z)")
    entradavarparcial.insert(0, "x")

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

def abrir_explicacao_derivadas_parciais():
    pass

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

def abrir_explicacao_lhopital():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre a Regra de L'Hôpital")
    janela_explicacao.geometry("600x320")

    texto = """A Regra de L'Hôpital é usada para resolver limites que apresentam formas indeterminadas, 
como 0/0 ou ∞/∞.

Sejam f(x) e g(x) funções deriváveis em um intervalo aberto contendo 'a', e se:
- lim(x→a) f(x) = 0 e lim(x→a) g(x) = 0  ou
- lim(x→a) f(x) = ∞ e lim(x→a) g(x) = ∞

Então:
    lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x)
desde que esse limite da derivada exista.

A regra pode ser aplicada repetidamente até a indeterminação desaparecer."""

    label = ctk.CTkLabel(janela_explicacao, text=texto, justify="left", wraplength=580, font=("Segoe UI", 14))
    label.pack(padx=20, pady=20)

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
    frame = ctk.CTkFrame(parent)
    frame.pack(anchor="w", pady=5, padx=5, fill="x")

    label = ctk.CTkLabel(frame, text=label_text, font=font)
    label.pack(anchor="w", padx=5)

    entry = ctk.CTkEntry(frame, width=400, height=30, corner_radius=5, font=font)
    entry.pack(padx=5, pady=5, anchor="w")

    return entry


def botao(parent, func, texto):
    btn = ctk.CTkButton(parent, text=texto, command=func, width=200, corner_radius=5, font=font)
    btn.pack(pady=5, padx=5, anchor="w")


def validar_expressao_em_tempo_real(entry_widget):
    try:
        expr = entry_widget.get()
        sp.sympify(expr)
        entry_widget.configure(border_color="green")
    except Exception:
        entry_widget.configure(border_color="red")

def aplicar_validacao_em_tempo_real(entry_widget):
    var = ctk.StringVar()
    entry_widget.configure(textvariable=var)
    var.trace_add("write", lambda *args: validar_expressao_em_tempo_real(entry_widget))


# ====================== TELA INICIAL =========================
class InitialPage(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Página Inicial")
        self.geometry("500x300")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)

        ctk.CTkLabel(self, text="Bem-vindo à Calculadora DDX", font=("Segoe UI", 20, "bold")).pack(pady=20)

        open_calculator_btn = ctk.CTkButton(self, text="Abrir Calculadora DDX", command=self.open_calculator, width=250)
        open_calculator_btn.pack(pady=10)

        manual_btn = ctk.CTkButton(
            self,
            text="Abrir Manual do DDX",
            command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing'),
            width=250
        )
        manual_btn.pack(pady=10)

    def open_calculator(self):
        self.destroy()
        app = App()
        app.mainloop()


# ====================== APLICAÇÃO PRINCIPAL =========================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora DDX")
        self.geometry("1400x800")
        self.minsize(1000, 700)

        self.create_widgets()

    def create_widgets(self):
        tabview = ctk.CTkTabview(self)
        tabview.pack(padx=10, pady=10, fill="both", expand=True)

        abas = ["Domínio e Imagem", "Derivadas", "Derivadas Parciais",  "Limites", "Raiz", "Gráficos", "L'Hôpital", "Integrais", "Manual"]
        frames = {aba: tabview.add(aba) for aba in abas}

        self.aba_dominio(frames["Domínio e Imagem"])
        self.aba_derivadas(frames["Derivadas"])
        self.aba_derivadas_parciais(frames["Derivadas Parciais"])
        self.aba_limites(frames["Limites"])
        self.aba_raiz(frames["Raiz"])
        self.aba_graficos(frames["Gráficos"])
        self.aba_lhopital(frames["L'Hôpital"])
        self.aba_integrais(frames["Integrais"])
        self.aba_manual(frames["Manual"])

    # ====================== ESTRUTURA PADRÃO DAS ABAS =========================
    def estrutura_aba(self, frame):
        container = ctk.CTkFrame(frame)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = ctk.CTkFrame(container)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        right_frame = ctk.CTkFrame(container)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        return left_frame, right_frame

    # ====================== ABA DOMÍNIO =========================
    def aba_dominio(self, frame):
        global entradadom, resultado_text_dom
        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="O que são Domínios e Imagens?", command=abrir_explicacao_dominios).pack(pady=5, anchor="w")

        entradadom = labeled_input(left, "Expressão:")
        aplicar_validacao_em_tempo_real(entradadom)
        botao(left, calculo_dominio_imagem, "Calcular")
        botao(left, exemplo_dominio_imagem, "Exemplo")

        resultado_text_dom = ctk.CTkTextbox(right, font=font)
        resultado_text_dom.pack(fill="both", expand=True)

    # ====================== ABA DERIVADAS =========================
    def aba_derivadas(self, frame):
        global entradaderiv, entradaponto, resultado_text_deriv
        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="O que é Derivada?", command=abrir_explicacao_derivada).pack(pady=5, anchor="w")

        entradaderiv = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradaderiv)
        entradaponto = labeled_input(left, "Ponto:")

        botao(left, calculo_derivada, "Calcular")
        botao(left, exemplo_derivada, "Exemplo")
        botao(left, plot_func_tangente, "Plotar Tangente")

        resultado_text_deriv = ctk.CTkTextbox(right, font=font)
        resultado_text_deriv.pack(fill="both", expand=True)

        img = ctk.CTkImage(Image.open("deriva.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)
    
    
    # ====================== ABA DERIVADAS PARCIAIS =========================
    def aba_derivadas_parciais(self, frame):
        global entradafuncparcial, entradavarparcial, resultado_text_parcial
        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="O que são Derivadas Parciais?", command=abrir_explicacao_derivadas_parciais).pack(pady=5, anchor="w")

        entradafuncparcial = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradafuncparcial)
        entradavarparcial = labeled_input(left, "Variável (vazio = todas):")

        botao(left, calculo_derivadas_parciais, "Calcular")
        botao(left, exemplo_derivada_parcial, "Exemplo")

        resultado_text_parcial = ctk.CTkTextbox(right, font=font)
        resultado_text_parcial.pack(fill="both", expand=True)

        img = ctk.CTkImage(Image.open("partial_derivative.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)


    # ====================== ABA LIMITES =========================
    def aba_limites(self, frame):
        global entradalimit, entradavar, entradatend, direcao_var, resultado_text_limite
        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="O que são Limites?", command=abrir_explicacao_limites).pack(pady=5, anchor="w")

        entradalimit = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradalimit)
        entradavar = labeled_input(left, "Variável:")
        entradatend = labeled_input(left, "Tendendo a:")
        aplicar_validacao_em_tempo_real(entradatend)

        direcao_var = ctk.StringVar(value="Ambos")
        ctk.CTkOptionMenu(left, variable=direcao_var, values=["Esquerda", "Direita", "Ambos"]).pack(pady=5, anchor="w")

        botao(left, calculo_limite, "Calcular")
        botao(left, exemplo_limite, "Exemplo")

        resultado_text_limite = ctk.CTkTextbox(right, font=font)
        resultado_text_limite.pack(fill="both", expand=True)

        img = ctk.CTkImage(Image.open("limit.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    # ====================== ABA RAIZ =========================
    def aba_raiz(self, frame):
        global entradaraiz, entradaindice, resultado_text_raiz
        left, right = self.estrutura_aba(frame)

        entradaraiz = labeled_input(left, "Número:")
        aplicar_validacao_em_tempo_real(entradaraiz)
        entradaindice = labeled_input(left, "Índice:")
        aplicar_validacao_em_tempo_real(entradaindice)

        botao(left, raiz, "Calcular")
        botao(left, exemplo_raiz, "Exemplo")

        resultado_text_raiz = ctk.CTkTextbox(right, font=font)
        resultado_text_raiz.pack(fill="both", expand=True)

        img = ctk.CTkImage(Image.open("raiz.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    # ====================== ABA GRÁFICOS =========================
    def aba_graficos(self, frame):
        global entrada_grafico, intervalo, show_points_var, resultado_text_grafico, frame_grafico_container

        left, right = self.estrutura_aba(frame)

        entrada_grafico = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entrada_grafico)

        intervalo = labeled_input(left, "Intervalo (ex: -10,10):")
        aplicar_validacao_em_tempo_real(intervalo)

        show_points_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Mostrar pontos críticos e de inflexão", variable=show_points_var).pack(pady=5, anchor="w")

        botao(left, plot_grafico, "Plotar")

        global interpolar_var, botao_plot_dados, check_interpolar
        ctk.CTkButton(left, text="Importar arquivo de pontos", command=carregar_arquivo_pontos).pack(pady=10, anchor="w")
        interpolar_var = ctk.BooleanVar(value=False)
        check_interpolar = ctk.CTkCheckBox(left, text="Interpolar curva", variable=interpolar_var)
        botao_plot_dados = ctk.CTkButton(left, text="Plotar dados importados", command=plotar_dados_importados)

        resultado_text_grafico = ctk.CTkTextbox(right, font=font, height=150)
        resultado_text_grafico.pack(fill="x", pady=(0, 10))

        # Frame onde o gráfico será embutido
        frame_grafico_container = ctk.CTkFrame(right)
        frame_grafico_container.pack(fill="both", expand=True)

    # ====================== ABA LHOPITAL =========================
    def aba_lhopital(self, frame):
        global entrada_num, entrada_den, entrada_ponto, direcao_lhopital, resultado_text_lhopital

        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="Quando usar L'Hôpital?", command=abrir_explicacao_lhopital).pack(pady=5, anchor="w")

        entrada_num = labeled_input(left, "Função Numerador:")
        aplicar_validacao_em_tempo_real(entrada_num)
        entrada_den = labeled_input(left, "Função Denominador:")
        aplicar_validacao_em_tempo_real(entrada_den)
        entrada_ponto = labeled_input(left, "Tendendo a:")
        aplicar_validacao_em_tempo_real(entrada_ponto)

        direcao_lhopital = ctk.StringVar(value="+")
        ctk.CTkOptionMenu(left, variable=direcao_lhopital, values=["+", "-"]).pack(pady=5, anchor="w")

        botao(left, self.calculo_lhopital, "Aplicar L'Hôpital")
        botao(left, self.exemplo_lhopital, "Exemplo")

        resultado_text_lhopital = ctk.CTkTextbox(right, font=font)
        resultado_text_lhopital.pack(fill="both", expand=True)


    # ====================== ABA INTEGRAIS =========================
    def aba_integrais(self, frame):
        global entrada_integrais, entrada_limite_inf, entrada_limite_sup, resultado_text_integral
        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="O que é Integral?", command=abrir_explicacao_integral).pack(pady=5, anchor="w")

        entrada_integrais = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entrada_integrais)
        entrada_limite_inf = labeled_input(left, "Limite inferior:")
        aplicar_validacao_em_tempo_real(entrada_limite_inf)
        entrada_limite_sup = labeled_input(left, "Limite superior:")
        aplicar_validacao_em_tempo_real(entrada_limite_sup)

        botao(left, calculo_integral, "Calcular")
        botao(left, exemplo_integral, "Exemplo")

        resultado_text_integral = ctk.CTkTextbox(right, font=font)
        resultado_text_integral.pack(fill="both", expand=True)

        img = ctk.CTkImage(Image.open("integral.png"), size=(300, 180))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    
    def calculo_lhopital(self):
        global entrada_num, entrada_den, entrada_ponto, direcao_lhopital, resultado_text_lhopital
        try:
            num = entrada_num.get()
            den = entrada_den.get()
            ponto = entrada_ponto.get()
            direcao = direcao_lhopital.get()

            passos = aplicar_lhopital(num, den, ponto, direcao)
            resultado_text_lhopital.delete("1.0", ctk.END)
            for passo in passos:
                resultado_text_lhopital.insert(ctk.END, passo + "\n")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")


    def aba_lhopital(self, frame):
        global entrada_num, entrada_den, entrada_ponto, direcao_lhopital, resultado_text_lhopital

        left, right = self.estrutura_aba(frame)

        ctk.CTkButton(left, text="Quando usar L'Hôpital?", command=abrir_explicacao_lhopital).pack(pady=5, anchor="w")

        entrada_num = labeled_input(left, "Função Numerador:")
        aplicar_validacao_em_tempo_real(entrada_num)
        entrada_den = labeled_input(left, "Função Denominador:")
        aplicar_validacao_em_tempo_real(entrada_den)
        entrada_ponto = labeled_input(left, "Tendendo a:")
        aplicar_validacao_em_tempo_real(entrada_ponto)

        direcao_lhopital = ctk.StringVar(value="+")
        ctk.CTkOptionMenu(left, variable=direcao_lhopital, values=["+", "-"]).pack(pady=5, anchor="w")

        ctk.CTkButton(left, text="Aplicar L'Hôpital", command=self.calculo_lhopital).pack(pady=5, anchor="w")
        ctk.CTkButton(left, text="Exemplo", command=self.exemplo_lhopital).pack(pady=5, anchor="w")

        resultado_text_lhopital = ctk.CTkTextbox(right, font=font)
        resultado_text_lhopital.pack(fill="both", expand=True)


    # ====================== ABA MANUAL =========================
    def aba_manual(self, frame):
        ctk.CTkButton(
            frame,
            text="Abrir Manual do DDX",
            command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing'),
            width=300
        ).pack(pady=20)


# ====================== EXECUÇÃO =========================
if __name__ == "__main__":
    initial_page = InitialPage()
    initial_page.mainloop()