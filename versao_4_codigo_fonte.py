import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import ttk
matplotlib.use('TkAgg')

#FUNÇÕES

def inputstr(pai):
    entry = tk.Entry(pai , width=40)
    entry.pack(pady=10)
    return entry

def botao(pai , func):
    button = tk.Button(pai, text="Calcular", command=func, pady=2 , padx=2)
    button.pack()

def calculo_derivada():
    return 0


def calculo_limite():
    return 0

def calculo_raizes():
    return 0
    
def textresult(pai):
    return tk.Label(pai , text="Resultado:" ,pady=3)
# criando janela principal

app = tk.Tk()
app.title('SympleCalc')
app.geometry("800x800")


notebook = ttk.Notebook(app)
notebook.place(x=0,y=0,width=800 , height=800)

aba_derivada = ttk.Frame(notebook)
notebook.add(aba_derivada, text='Derivadas')

aba_limite = ttk.Frame(notebook)
notebook.add(aba_limite , text='Limites')

aba_raizes = ttk.Frame(notebook)
notebook.add(aba_raizes , text= "Raízes quadradas")

#ITENS ABA DERIVADA

lb1 = tk.Label(aba_derivada , text='Insira abaixo a função:')
lb1.pack()
entradaderiv = inputstr(aba_derivada)
lb6 = tk.Label(aba_derivada , text='Insira o ponto:')
entradaponto = inputstr(aba_derivada)
botaoderiv = botao(aba_derivada , calculo_derivada)
textresult(aba_derivada).pack()
resultado_text = tk.Text(aba_derivada, height=10, width=40)
resultado_text.pack()

#ITENS ABA LIMITES

lb2 = tk.Label(aba_limite , text='Insira abaixo o limite:')
lb2.pack()
entradalimit = inputstr(aba_limite)
lb3 = tk.Label(aba_limite , text='Insira a variável:')
lb3.pack()
entradavar = inputstr(aba_limite)
lb4 = tk.Label(aba_limite , text='variavel tende para que numero?')
lb4.pack()
entradatend = inputstr(aba_limite)
botaolimit = botao(aba_limite , calculo_limite)
textresult(aba_limite).pack()
resultado_text = tk.Text(aba_limite, height=10, width=40)
resultado_text.pack()

#ITENS ABA RAIZ

lb5 = tk.Label(aba_raizes , text='insira o número: ')
lb5.pack()
entradaraiz = inputstr(aba_raizes)
botaoraiz = botao(aba_raizes , calculo_raizes)
textresult(aba_raizes).pack()
resultado_text = tk.Text(aba_raizes, height=10, width=40)
resultado_text.pack()







app.mainloop()