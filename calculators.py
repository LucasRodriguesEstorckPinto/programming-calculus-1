import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re
from functools import lru_cache
from scipy.optimize import fsolve
from sympy import Interval, Union, S, solve, log, Complement, FiniteSet, oo, Pow
from sympy.solvers.inequalities import solve_univariate_inequality
from tkinter import messagebox

# Símbolos globais para os cálculos
x = sp.symbols('x')
n = sp.symbols('n', integer=True)

class DomainImageCalculator:
    def calculate(self, func_str):
        if not func_str:
            return "Erro: A expressão da função não pode estar vazia."
        try:
            func = sp.sympify(func_str)
            dominio = self._calcular_dominio(func, x)

            if "Erro" in str(dominio):
                imagem = "Não foi possível calcular a imagem devido ao domínio inválido."
            else:
                imagem = self._calcular_imagem(func, x, dominio)

            dominio_fmt = self._formatar_conjunto(dominio)
            imagem_fmt = self._formatar_conjunto(imagem)

            resultado = f"""Resultados:
========================
Função: {func_str}

Domínio: {dominio_fmt}
{self._explicar_dominio(dominio)}

Imagem: {imagem_fmt}
{self._explicar_imagem(imagem, func_str)}
========================"""
            return resultado

        except sp.SympifyError:
            return "Erro: Expressão da função inválida. Verifique a sintaxe."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"

    def _calcular_dominio(self, func, var):
        try:
            return sp.calculus.util.continuous_domain(func, var, S.Reals)
        except Exception as e:
            return f"Erro ao calcular o domínio: {e}"
    
    def _calcular_imagem(self, func, var, domain):
        try:
            return sp.calculus.util.function_range(func, var, domain)
        except Exception as e:
             return f"Erro ao calcular a imagem: {e}"

    def _formatar_intervalo(self, intervalo):
        # Implementação simplificada
        return str(intervalo).replace("Intersection", "∩").replace("Union", "∪")

    def _formatar_conjunto(self, conjunto):
        if isinstance(conjunto, (sp.Interval, sp.Union, sp.Intersection, sp.Complement, sp.FiniteSet)):
            return self._formatar_intervalo(conjunto)
        if conjunto == S.Reals:
            return "ℝ (todos os números reais)"
        return str(conjunto)

    def _explicar_dominio(self, dominio):
        if dominio == S.Reals:
            return "Explicação: A função está definida para todos os números reais."
        return f"Explicação: A função possui restrições, resultando no domínio acima."

    def _explicar_imagem(self, imagem, func_str):
        if "sin" in func_str or "cos" in func_str:
            if imagem == sp.Interval(-1, 1):
                return "Explicação: A imagem de sen(x) ou cos(x) é o intervalo [-1, 1]."
        return "Explicação: A imagem representa todos os valores de saída possíveis da função."


class DerivativeCalculator:
    def calculate(self, func_str, point_str):
        if not func_str:
            return "Erro: A função não pode estar vazia."
        try:
            func = sp.sympify(func_str)
            derivada = sp.diff(func, x)
            resultado = f"A derivada da função é: {derivada}\n"

            if point_str:
                point = sp.sympify(point_str)
                coef_angular = derivada.subs(x, point)
                reta = func.subs(x, point) + coef_angular * (x - point)
                resultado += f"A equação da reta tangente no ponto x={point} é: y = {sp.simplify(reta)}\n"
            
            return resultado
        except (sp.SympifyError, ValueError):
            return "Erro: Função ou ponto inválido. Verifique a sintaxe."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"

    def plot_tangent(self, func_str, point_str):
        if not func_str or not point_str:
            messagebox.showerror("Erro", "Função e ponto são necessários para plotar a tangente.")
            return
        try:
            func = sp.sympify(func_str)
            point = float(sp.sympify(point_str))
            derivada = sp.diff(func, x)
            coef_angular = derivada.subs(x, point)
            reta = func.subs(x, point) + coef_angular * (x - point)
            
            func_num = sp.lambdify(x, func, "numpy")
            reta_num = sp.lambdify(x, reta, "numpy")
            
            x_vals = np.linspace(point - 10, point + 10, 400)
            y_func = func_num(x_vals)
            y_reta = reta_num(x_vals)

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_func, label=f'f(x) = {func_str}')
            plt.plot(x_vals, y_reta, label=f'Tangente em x = {point}', linestyle='--')
            plt.scatter([point], [func.subs(x, point)], color='red', zorder=5) # Ponto de tangência
            plt.axhline(0, color='gray', lw=0.5)
            plt.axvline(0, color='gray', lw=0.5)
            plt.title('Função e Reta Tangente')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Erro ao Plotar", f"Não foi possível gerar o gráfico: {e}")


class PartialDerivativeCalculator:
    def calculate(self, func_str, var_str):
        if not func_str:
            return "Erro: A função não pode estar vazia."
        try:
            variaveis_func = sorted(set(re.findall(r"[a-zA-Z]+", func_str)))
            vars_sympy = sp.symbols(" ".join(variaveis_func))
            expr = sp.sympify(func_str)
            
            resultado = ""
            if var_str.strip():
                var = sp.Symbol(var_str.strip())
                derivada = sp.diff(expr, var)
                resultado = f"∂f/∂{var} = {derivada}\n"
            else:
                for var in vars_sympy:
                    derivada = sp.diff(expr, var)
                    resultado += f"∂f/∂{var} = {derivada}\n"
            return resultado if resultado else "Nenhuma variável encontrada para derivar."

        except sp.SympifyError:
            return "Erro: Expressão da função inválida."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"


class LimitCalculator:
    def calculate(self, func_str, var_str, tend_str, dir_str):
        # Validação para garantir que os campos não estão vazios
        if not func_str or not var_str or not tend_str:
            return "Erro: Preencha os campos de função, variável e ponto de tendência."
        try:
            func = sp.sympify(func_str)
            variavel = sp.symbols(var_str)
            # A linha crucial: sympify trata 'a' como um símbolo, não um número
            valor_tendencia = sp.sympify(tend_str)

            if dir_str == "Ambos":
                limite_esquerda = sp.limit(func, variavel, valor_tendencia, dir='-')
                limite_direita = sp.limit(func, variavel, valor_tendencia, dir='+')
                if limite_esquerda == limite_direita:
                    return f"O limite da função é: {limite_esquerda}"
                else:
                    return f"O limite não existe.\nLimite à esquerda: {limite_esquerda}\nLimite à direita: {limite_direita}"
            else:
                direcao = '-' if dir_str == "Esquerda" else '+'
                limite = sp.limit(func, variavel, valor_tendencia, dir=direcao)
                return f"O limite da função pela {dir_str.lower()} é: {limite}"

        except (sp.SympifyError, TypeError):
            return "Erro: Verifique a sintaxe da função, variável ou ponto de tendência."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"



class RootCalculator:
    def calculate(self, num_str, index_str):
        if not num_str or not index_str:
            return "Erro: Número e índice devem ser preenchidos."
        try:
            numero = float(num_str)
            indice = int(index_str)
            
            if numero < 0 and indice % 2 == 0:
                return "Erro: Raiz de índice par para número negativo não é real."
            
            resultado = numero**(1/indice)
            return f"A raíz {indice}-ésima de {numero} é: {resultado:.6f}"

        except ValueError:
            return "Erro: Insira um número e um índice válidos."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"

class LHopitalCalculator:
    def calculate(self, num_str, den_str, point_str, direction_str):
        if not num_str or not den_str or not point_str:
            return "Erro: Todos os campos são obrigatórios."
        try:
            f = sp.sympify(num_str)
            g = sp.sympify(den_str)
            ponto = sp.sympify(point_str)
            passos = self._aplicar_lhopital(f, g, ponto, direction_str)
            return "\n".join(passos)
        except sp.SympifyError:
            return "Erro: Verifique a sintaxe das funções ou do ponto."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"

    def _aplicar_lhopital(self, f, g, ponto, direcao):
        passos = []
        lim_f = sp.limit(f, x, ponto, dir=direcao)
        lim_g = sp.limit(g, x, ponto, dir=direcao)
        passos.append(f"Analisando o limite para x → {ponto}{direcao}")
        passos.append(f"  lim f(x) = {lim_f}")
        passos.append(f"  lim g(x) = {lim_g}")

        if (lim_f.is_infinite and lim_g.is_infinite) or (lim_f == 0 and lim_g == 0):
             passos.append("  ✅ Forma indeterminada detectada. Aplicando L'Hospital:")
        else:
            passos.append("  ❌ A Regra de L’Hospital NÃO se aplica — a forma não é indeterminada.")
            return passos

        num, den = f, g
        for i in range(1, 6): # Máximo de 5 iterações
            num_deriv = sp.diff(num, x)
            den_deriv = sp.diff(den, x)
            passos.append(f"    Iteração {i}:")
            passos.append(f"      f'(x) = {num_deriv}")
            passos.append(f"      g'(x) = {den_deriv}")
            
            resultado = sp.limit(num_deriv / den_deriv, x, ponto, dir=direcao)
            passos.append(f"      lim f'(x)/g'(x) = {resultado}")

            if not resultado.is_infinite and resultado != 0:
                 passos.append(f"      ✅ Resultado final: {resultado}")
                 return passos
            
            num, den = num_deriv, den_deriv
        
        passos.append("      ❌ Número máximo de iterações atingido.")
        return passos


class IntegralCalculator:
    def calculate(self, func_str, lim_inf_str, lim_sup_str):
        if not func_str:
            return "Erro: A função não pode estar vazia."
        try:
            func = sp.sympify(func_str)
            if lim_inf_str and lim_sup_str:
                lim_inf = sp.sympify(lim_inf_str)
                lim_sup = sp.sympify(lim_sup_str)
                integral = sp.integrate(func, (x, lim_inf, lim_sup))
                return f"A integral definida de {lim_inf} a {lim_sup} é: {integral.evalf()}"
            else:
                integral = sp.integrate(func, x)
                return f"A integral indefinida é: {integral} + C"
        except (sp.SympifyError, ValueError):
            return "Erro: Verifique a sintaxe da função ou dos limites."
        except Exception as e:
            return f"Ocorreu um erro inesperado: {e}"


class GraphingCalculator:
    def plot_graph(self, func_str, interval_str, show_points):
        if not func_str or not interval_str:
            raise ValueError("A função e o intervalo são obrigatórios.")

        func_list_str = [f.strip() for f in func_str.split(',')]
        try:
            lower, upper = map(float, interval_str.split(','))
            if lower >= upper:
                raise ValueError("O limite inferior do intervalo deve ser menor que o superior.")
        except ValueError:
            raise ValueError("Formato do intervalo inválido. Use o formato 'min,max', por exemplo: -10,10.")
        
        func_sym_list = [sp.sympify(f) for f in func_list_str]
        
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        result_text = ""

        x_vals = np.linspace(lower, upper, 800)

        for i, func_sym in enumerate(func_sym_list):
            func_numeric = sp.lambdify(x, func_sym, 'numpy')
            y_vals = func_numeric(x_vals)
            
            ax.plot(x_vals, y_vals, label=f'${sp.latex(func_sym)}$', linewidth=2)

            # (A lógica detalhada de assíntotas e pontos críticos pode ser adicionada aqui se necessário)
            # Por simplicidade, a lógica original mais complexa foi omitida, mas pode ser reintegrada.

        if show_points and len(func_sym_list) == 1:
             result_text += self._add_critical_points(func_sym_list[0], lower, upper, ax)

        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gráfico das Funções')
        ax.legend()
        plt.tight_layout()
        
        result_text += "\nGráfico plotado com sucesso!"
        return result_text, fig

    def _add_critical_points(self, func_sym, lower, upper, ax):
        # Lógica simplificada para pontos críticos
        text = ""
        try:
            fprime = sp.diff(func_sym, x)
            fsecond = sp.diff(fprime, x)

            # Encontra raízes numericamente para evitar problemas com solve
            fprime_num = sp.lambdify(x, fprime, 'numpy')
            crit_points = set()
            x_samples = np.linspace(lower, upper, 2000)
            for i in range(len(x_samples) - 1):
                if np.sign(fprime_num(x_samples[i])) != np.sign(fprime_num(x_samples[i+1])):
                    sol = fsolve(fprime_num, (x_samples[i] + x_samples[i+1]) / 2)[0]
                    if lower <= sol <= upper:
                        crit_points.add(round(sol, 4))
            
            for p in sorted(list(crit_points)):
                y_p = func_sym.subs(x, p)
                fsecond_val = fsecond.subs(x, p)
                
                point_type = "Ponto Crítico"
                color = 'purple'
                if fsecond_val > 0:
                    point_type = "Mínimo Local"
                    color = 'green'
                elif fsecond_val < 0:
                    point_type = "Máximo Local"
                    color = 'red'
                
                ax.scatter(p, y_p, color=color, s=50, zorder=5)
                text += f"{point_type} em x ≈ {p:.2f}\n"
        except Exception as e:
            text += f"Não foi possível calcular os pontos críticos: {e}\n"
        return text