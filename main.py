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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import filedialog
from scipy.interpolate import interp1d

# Importa as classes de cálculo
from calculators import (
    DomainImageCalculator,
    DerivativeCalculator,
    PartialDerivativeCalculator,
    LimitCalculator,
    RootCalculator,
    GraphingCalculator,
    LHopitalCalculator,
    IntegralCalculator
)

matplotlib.use("TkAgg")
# Configuração do tema (dark, light ou system)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Fonte padrão para os widgets
font = ("Segoe UI", 14)

# ====================== FUNÇÕES DE EXEMPLO E EXPLICAÇÃO  =========================
def exemplo_lhopital(num_entry, den_entry, point_entry):
    num_entry.delete(0, ctk.END)
    den_entry.delete(0, ctk.END)
    point_entry.delete(0, ctk.END)
    num_entry.insert(0, "sin(x)")
    den_entry.insert(0, "x")
    point_entry.insert(0, "0")

def exemplo_raiz(result_text_widget):
    example_text = ("Exemplo de Raiz Quadrada:\n"
        "Número: 256\n"
        "Definição: A raiz quadrada de um número é um valor que, quando multiplicado por si mesmo, "
        "resulta no número original.\n"
        "Cálculo: A raiz quadrada de 256 é 16, pois 16 * 16 = 256.\n"
        "Propriedades: A raiz quadrada de um número positivo é sempre um número positivo. "
        "Neste caso, a raiz quadrada de 256 é um valor exato e inteiro, 16.")
    result_text_widget.delete("1.0", ctk.END)
    result_text_widget.insert(ctk.END, example_text)

def exemplo_dominio_imagem(result_text_widget):
    example_text = (
        "Exemplo de Domínio e Imagem:\n"
        "Função: f(x) = 1/(x-2)\n"
        "Domínio: Todos os valores de x, exceto x=2. Isso porque a função se torna indefinida quando x=2, "
        "pois resultaria em uma divisão por zero.\n"
        "Imagem: Todos os valores reais, exceto f(x)=0. A função nunca toca o eixo x, "
        "pois não há valor de x que faça a função igual a zero."
    )
    result_text_widget.delete("1.0", ctk.END)
    result_text_widget.insert(ctk.END, example_text)

def exemplo_limite(result_text_widget):
    example_text = (
        "Exemplo de Limite:\n"
        "Função: f(x) = (x^2 - 1)/(x - 1)\n"
        "Para calcular o limite de f(x) quando x tende a 1, simplificamos a função:\n"
        "f(x) = (x + 1) para x ≠ 1.\n"
        "Então, o limite de f(x) quando x tende a 1 é 2.\n"
        "Lembre-se de que o limite se refere ao valor que a função se aproxima à medida que x se aproxima de 1."
    )
    result_text_widget.delete("1.0", ctk.END)
    result_text_widget.insert(ctk.END, example_text)

def exemplo_derivada(result_text_widget):
    example_text = (
        "Exemplo de Derivada e Tangente:\n"
        "Função: f(x) = x^2\n"
        "Derivada: f'(x) = 2x. Isso representa a inclinação da função em qualquer ponto x.\n"
        "No ponto x=3, f'(3) = 6. Isso significa que a inclinação da tangente à curva no ponto (3, f(3)) é 6.\n"
        "A equação da reta tangente é dada por: y = f(3) + f'(3)*(x - 3)\n"
        "Neste caso, a reta tangente é y = 9 + 6(x - 3), simplificando: y = 6x - 9."
    )
    result_text_widget.delete("1.0", ctk.END)
    result_text_widget.insert(ctk.END, example_text)

def exemplo_derivada_parcial(func_entry, var_entry):
    func_entry.delete(0, ctk.END)
    var_entry.delete(0, ctk.END)
    func_entry.insert(0, "x**2 * y + sin(z)")
    var_entry.insert(0, "x")

def exemplo_integral(result_text_widget):
    example_text = (
        "Exemplo de Integral:\n"
        "Função: f(x) = x^2\n"
        "Integral Indefinida: ∫x^2 dx = (1/3)x^3 + C, onde C é a constante de integração.\n"
        "Integral Definida de 0 a 2: ∫(de 0 a 2) x^2 dx = [(1/3)x^3] de 0 a 2 = (8/3) - 0 = 8/3.\n"
        "Isso representa a área sob a curva de f(x) entre x=0 e x=2."
    )
    result_text_widget.delete("1.0", ctk.END)
    result_text_widget.insert(ctk.END, example_text)

def abrir_explicacao_integral():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Integrais")
    janela_explicacao.geometry("500x300")
    texto_explicacao = """A integral de uma função representa a área sob a curva dessa função em um determinado intervalo.
Ela é usada para calcular áreas, volumes e resolver problemas físicos como trabalho e deslocamento.

Fonte: Stewart, James. Cálculo. 8ª edição."""
    ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left").pack(padx=20, pady=20)
    ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy).pack(pady=10)

def abrir_explicacao_derivada():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Derivadas")
    janela_explicacao.geometry("500x300")
    texto_explicacao = """A derivada de uma função representa a taxa de variação dessa função em um determinado ponto.
Ela é usada para calcular velocidades, acelerações e resolver problemas físicos como otimização e crescimento populacional.

Fonte: Stewart, James. Cálculo. 8ª edição."""
    ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left").pack(padx=20, pady=20)
    ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy).pack(pady=10)

def abrir_explicacao_limites():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Limites")
    janela_explicacao.geometry("500x300")
    texto_explicacao = """O limite de uma função descreve o comportamento dessa função à medida que a variável independente se aproxima de um determinado valor.
Ele é usado para definir derivadas, integrais e resolver problemas envolvendo continuidade e comportamento assintótico.

Fonte: Stewart, James. Cálculo. 8ª edição."""
    ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left").pack(padx=20, pady=20)
    ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy).pack(pady=10)

def abrir_explicacao_derivadas_parciais():
    
    messagebox.showinfo("Explicação", "Derivadas parciais são usadas para funções de múltiplas variáveis. Elas medem a taxa de variação da função em relação a uma variável, mantendo as outras constantes.")

def abrir_explicacao_dominios():
    janela_explicacao = ctk.CTkToplevel()
    janela_explicacao.title("Explicação sobre Domínio e Imagem")
    janela_explicacao.geometry("500x300")
    texto_explicacao = """O domínio de uma função é o conjunto de todos os valores de entrada para os quais a função está definida.
A imagem de uma função é o conjunto de todos os valores de saída que a função pode assumir.
Eles são usados para entender o comportamento e as restrições de funções em diversos contextos matemáticos e aplicados.

Fonte: Stewart, James. Cálculo. 8ª edição."""
    ctk.CTkLabel(janela_explicacao, text=texto_explicacao, wraplength=450, justify="left").pack(padx=20, pady=20)
    ctk.CTkButton(janela_explicacao, text="Fechar", command=janela_explicacao.destroy).pack(pady=10)

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
    ctk.CTkLabel(janela_explicacao, text=texto, justify="left", wraplength=580, font=("Segoe UI", 14)).pack(padx=20, pady=20)

# ====================== COMPONENTES DA UI  =========================
def labeled_input(parent, label_text):
    frame = ctk.CTkFrame(parent)
    frame.pack(anchor="w", pady=5, padx=5, fill="x")
    ctk.CTkLabel(frame, text=label_text, font=font).pack(anchor="w", padx=5)
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

# ====================== TELA INICIAL  =========================
class InitialPage(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Página Inicial")
        self.geometry("500x300")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)
        ctk.CTkLabel(self, text="Bem-vindo à Calculadora DDX", font=("Segoe UI", 20, "bold")).pack(pady=20)
        ctk.CTkButton(self, text="Abrir Calculadora DDX", command=self.open_calculator, width=250).pack(pady=10)
        ctk.CTkButton(self, text="Abrir Manual do DDX", command=lambda: webbrowser.open('https://drive.google.com/file/d/1Kn4UD3txfoK37DOliF8L4ePNe53l3nui/view?usp=sharing'), width=250).pack(pady=10)

    def open_calculator(self):
        self.destroy()
        app = App()
        app.mainloop()

# ====================== APLICAÇÃO PRINCIPAL  =========================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora DDX")
        self.geometry("1400x800")
        self.minsize(1000, 700)
        
        # Instancia as classes de cálculo
        self.domain_calc = DomainImageCalculator()
        self.derivative_calc = DerivativeCalculator()
        self.partial_derivative_calc = PartialDerivativeCalculator()
        self.limit_calc = LimitCalculator()
        self.root_calc = RootCalculator()
        self.graph_calc = GraphingCalculator()
        self.lhopital_calc = LHopitalCalculator()
        self.integral_calc = IntegralCalculator()

        self.create_widgets()

    def create_widgets(self):
        tabview = ctk.CTkTabview(self)
        tabview.pack(padx=10, pady=10, fill="both", expand=True)

        abas = ["Domínio e Imagem", "Derivadas", "Derivadas Parciais",  "Limites", "Raiz", "Gráficos", "L'Hospital", "Integrais", "Manual"]
        frames = {aba: tabview.add(aba) for aba in abas}

        self.aba_dominio(frames["Domínio e Imagem"])
        self.aba_derivadas(frames["Derivadas"])
        self.aba_derivadas_parciais(frames["Derivadas Parciais"])
        self.aba_limites(frames["Limites"])
        self.aba_raiz(frames["Raiz"])
        self.aba_graficos(frames["Gráficos"])
        self.aba_lhopital(frames["L'Hospital"])
        self.aba_integrais(frames["Integrais"])
        self.aba_manual(frames["Manual"])

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
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="O que são Domínios e Imagens?", command=abrir_explicacao_dominios).pack(pady=5, anchor="w")
        entradadom = labeled_input(left, "Expressão:")
        aplicar_validacao_em_tempo_real(entradadom)
        resultado_text_dom = ctk.CTkTextbox(right, font=font)
        resultado_text_dom.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_dominio_imagem(entradadom.get(), resultado_text_dom), "Calcular")
        botao(left, lambda: exemplo_dominio_imagem(resultado_text_dom), "Exemplo")

    def handle_dominio_imagem(self, func_str, result_widget):
        resultado = self.domain_calc.calculate(func_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA DERIVADAS =========================
    def aba_derivadas(self, frame):
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="O que é Derivada?", command=abrir_explicacao_derivada).pack(pady=5, anchor="w")
        entradaderiv = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradaderiv)
        entradaponto = labeled_input(left, "Ponto:")
        resultado_text_deriv = ctk.CTkTextbox(right, font=font)
        resultado_text_deriv.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_derivada(entradaderiv.get(), entradaponto.get(), resultado_text_deriv), "Calcular")
        botao(left, lambda: exemplo_derivada(resultado_text_deriv), "Exemplo")
        botao(left, lambda: self.handle_plot_tangente(entradaderiv.get(), entradaponto.get()), "Plotar Tangente")
        
        img = ctk.CTkImage(Image.open("deriva.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    def handle_derivada(self, func_str, point_str, result_widget):
        resultado = self.derivative_calc.calculate(func_str, point_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    def handle_plot_tangente(self, func_str, point_str):
        self.derivative_calc.plot_tangent(func_str, point_str)

    # ====================== ABA DERIVADAS PARCIAIS =========================
    def aba_derivadas_parciais(self, frame):
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="O que são Derivadas Parciais?", command=abrir_explicacao_derivadas_parciais).pack(pady=5, anchor="w")
        entradafuncparcial = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradafuncparcial)
        entradavarparcial = labeled_input(left, "Variável (vazio = todas):")
        resultado_text_parcial = ctk.CTkTextbox(right, font=font)
        resultado_text_parcial.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_derivada_parcial(entradafuncparcial.get(), entradavarparcial.get(), resultado_text_parcial), "Calcular")
        botao(left, lambda: exemplo_derivada_parcial(entradafuncparcial, entradavarparcial), "Exemplo")

    def handle_derivada_parcial(self, func_str, var_str, result_widget):
        resultado = self.partial_derivative_calc.calculate(func_str, var_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA LIMITES =========================
    def aba_limites(self, frame):
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="O que são Limites?", command=abrir_explicacao_limites).pack(pady=5, anchor="w")
        entradalimit = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entradalimit)
        entradavar = labeled_input(left, "Variável:")
        entradatend = labeled_input(left, "Tendendo a:")
        direcao_var = ctk.StringVar(value="Ambos")
        ctk.CTkOptionMenu(left, variable=direcao_var, values=["Esquerda", "Direita", "Ambos"]).pack(pady=5, anchor="w")
        resultado_text_limite = ctk.CTkTextbox(right, font=font)
        resultado_text_limite.pack(fill="both", expand=True)

        botao(left, lambda: self.handle_limite(entradalimit.get(), entradavar.get(), entradatend.get(), direcao_var.get(), resultado_text_limite), "Calcular")
        botao(left, lambda: exemplo_limite(resultado_text_limite), "Exemplo")
        
        img = ctk.CTkImage(Image.open("limit.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    def handle_limite(self, func_str, var_str, tend_str, dir_str, result_widget):
        resultado = self.limit_calc.calculate(func_str, var_str, tend_str, dir_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA RAIZ =========================
    def aba_raiz(self, frame):
        left, right = self.estrutura_aba(frame)
        entradaraiz = labeled_input(left, "Número:")
        aplicar_validacao_em_tempo_real(entradaraiz)
        entradaindice = labeled_input(left, "Índice:")
        aplicar_validacao_em_tempo_real(entradaindice)
        resultado_text_raiz = ctk.CTkTextbox(right, font=font)
        resultado_text_raiz.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_raiz(entradaraiz.get(), entradaindice.get(), resultado_text_raiz), "Calcular")
        botao(left, lambda: exemplo_raiz(resultado_text_raiz), "Exemplo")

        img = ctk.CTkImage(Image.open("raiz.png"), size=(250, 120))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    def handle_raiz(self, num_str, index_str, result_widget):
        resultado = self.root_calc.calculate(num_str, index_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA GRÁFICOS =========================
    def aba_graficos(self, frame):
        left, right = self.estrutura_aba(frame)
        entrada_grafico = labeled_input(left, "Função (ou funções, separadas por vírgula):")
        aplicar_validacao_em_tempo_real(entrada_grafico)
        intervalo = labeled_input(left, "Intervalo (ex: -10,10):")
        show_points_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Mostrar pontos críticos e de inflexão", variable=show_points_var).pack(pady=5, anchor="w")
        
        resultado_text_grafico = ctk.CTkTextbox(right, font=font, height=150)
        resultado_text_grafico.pack(fill="x", pady=(0, 10))
        
        frame_grafico_container = ctk.CTkFrame(right)
        frame_grafico_container.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_plot(
            entrada_grafico.get(),
            intervalo.get(),
            show_points_var.get(),
            resultado_text_grafico,
            frame_grafico_container
        ), "Plotar")

    def handle_plot(self, func_str, interval_str, show_points, result_widget, container):
        try:
            for widget in container.winfo_children():
                widget.destroy()
            
            result_text, fig = self.graph_calc.plot_graph(func_str, interval_str, show_points)
            
            result_widget.delete("1.0", ctk.END)
            result_widget.insert(ctk.END, result_text)
            
            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, container)
            toolbar.update()
            toolbar.pack()
            
        except ValueError as e:
            messagebox.showerror("Erro de Entrada", str(e))
        except Exception as e:
            messagebox.showerror("Erro ao Plotar", f"Ocorreu um erro inesperado: {e}")

    # ====================== ABA LHOPITAL =========================
    def aba_lhopital(self, frame):
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="Quando usar L'Hospital?", command=abrir_explicacao_lhopital).pack(pady=5, anchor="w")
        entrada_num = labeled_input(left, "Função Numerador:")
        aplicar_validacao_em_tempo_real(entrada_num)
        entrada_den = labeled_input(left, "Função Denominador:")
        aplicar_validacao_em_tempo_real(entrada_den)
        entrada_ponto = labeled_input(left, "Tendendo a:")
        direcao_lhopital = ctk.StringVar(value="+")
        ctk.CTkOptionMenu(left, variable=direcao_lhopital, values=["+", "-"]).pack(pady=5, anchor="w")
        resultado_text_lhopital = ctk.CTkTextbox(right, font=font)
        resultado_text_lhopital.pack(fill="both", expand=True)

        botao(left, lambda: self.handle_lhopital(entrada_num.get(), entrada_den.get(), entrada_ponto.get(), direcao_lhopital.get(), resultado_text_lhopital), "Aplicar L'Hospital")
        botao(left, lambda: exemplo_lhopital(entrada_num, entrada_den, entrada_ponto), "Exemplo")

    def handle_lhopital(self, num_str, den_str, point_str, direction_str, result_widget):
        resultado = self.lhopital_calc.calculate(num_str, den_str, point_str, direction_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA INTEGRAIS =========================
    def aba_integrais(self, frame):
        left, right = self.estrutura_aba(frame)
        ctk.CTkButton(left, text="O que é Integral?", command=abrir_explicacao_integral).pack(pady=5, anchor="w")
        entrada_integrais = labeled_input(left, "Função:")
        aplicar_validacao_em_tempo_real(entrada_integrais)
        entrada_limite_inf = labeled_input(left, "Limite inferior:")
        entrada_limite_sup = labeled_input(left, "Limite superior:")
        resultado_text_integral = ctk.CTkTextbox(right, font=font)
        resultado_text_integral.pack(fill="both", expand=True)
        
        botao(left, lambda: self.handle_integral(entrada_integrais.get(), entrada_limite_inf.get(), entrada_limite_sup.get(), resultado_text_integral), "Calcular")
        botao(left, lambda: exemplo_integral(resultado_text_integral), "Exemplo")

        img = ctk.CTkImage(Image.open("integral.png"), size=(300, 180))
        ctk.CTkLabel(right, image=img, text="").pack(pady=10)

    def handle_integral(self, func_str, lim_inf_str, lim_sup_str, result_widget):
        resultado = self.integral_calc.calculate(func_str, lim_inf_str, lim_sup_str)
        result_widget.delete("1.0", ctk.END)
        result_widget.insert(ctk.END, resultado)

    # ====================== ABA MANUAL =========================
    def aba_manual(self, frame):
        ctk.CTkButton(frame, text="Abrir Manual do DDX", command=lambda: webbrowser.open('https://docs.google.com/document/d/1hvcUL36juGBm_8lsdOpPrMLWzmYnGvakKHaMj1BbxlY/edit?usp=sharing'), width=300).pack(pady=20)

# ====================== EXECUÇÃO =========================
if __name__ == "__main__":
    initial_page = InitialPage()
    initial_page.mainloop()