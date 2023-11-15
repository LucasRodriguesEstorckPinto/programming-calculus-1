import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import ttk
matplotlib.use('TkAgg')


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

lb1 = tk.Label(aba_derivada , text='aaaa')

app.mainloop()