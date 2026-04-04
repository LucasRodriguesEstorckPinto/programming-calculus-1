# DDX - Sistema Avançado de Simulação e Modelagem Matemática

## ⚠️ AVISO LEGAL: PROTEÇÃO DE PROPRIEDADE INTELECTUAL E DIREITOS AUTORAIS

Este repositório contém a documentação técnica oficial e o código-fonte original do programa de computador denominado **DDX**, submetido ao registro de propriedade intelectual junto ao Instituto Nacional da Propriedade Industrial (INPI) sob a categoria de Simulação e Modelagem (SM01) e aplicação em Cálculo (MT05).

**Titularidade Patrimonial:** Os registros de propriedade intelectual gerados por este projeto são de titularidade exclusiva da **Fundação Universidade do Estado do Rio de Janeiro – UERJ**. É expressamente vedada a exploração econômica, reprodução não autorizada, distribuição ou comercialização dos direitos de propriedade intelectual (incluindo este código-fonte) sem a devida autorização legal institucional.

**Autoria e Invenção:**
Desenvolvimento tecnológico idealizado e executado pelo consórcio de autores:
* **Lucas Rodrigues Estorck Pinto** (Autor Principal / Desenvolvedor - Engenharia da Computação)
* **Prof. Dr. Germano Amaral Monerat** (Coautor / Docente)
* **Profa. Dra. Silvia Mara da Costa Campos** (Coautora / Docente)
* **Prof. Dr. Rodrigo Lamblet Mafort** (Coautor / Docente)
* **Profa. Dra. Loena Marins do Couto** (Coautora / Docente)
* **Prof. Dr. Eduardo Vasquez Corrêa Silva** (Coautor / Docente)

A integridade legal deste depósito está atrelada à preservação criptográfica (Hash SHA-256/SHA-512) destes arquivos. Qualquer modificação desconstituirá a prova de integridade eletrônica do depósito perante o Poder Judiciário.

---

## 💻 Visão Geral Técnica do Código-Fonte

O **DDX** é uma aplicação desktop robusta desenvolvida em **Python**, projetada para resolver, analisar e visualizar problemas complexos de cálculo diferencial e integral. A arquitetura do software foi construída com foco em precisão simbólica, métodos numéricos avançados e uma interface gráfica moderna e responsiva.

### Stack Tecnológico e Bibliotecas Utilizadas
A aplicação integra múltiplas bibliotecas de alto nível para garantir performance e exatidão nos cálculos:
* **Interface Gráfica (GUI):** `customtkinter` e `tkinter` para a renderização de uma interface em Dark Mode, com sistema de abas e validação de expressões matemáticas em tempo real.
* **Cálculo Simbólico (CAS):** `sympy` é o motor matemático central, responsável por resolver derivadas, integrais, limites, simplificações algébricas e cálculo estrito de domínios e imagens.
* **Computação Numérica:** `numpy` e `scipy` (especificamente `scipy.optimize.fsolve` e `scipy.interpolate.interp1d`) para a descoberta de raízes numéricas, interpolação cúbica de dados discretos e ajustes de amostragem em funções complexas.
* **Visualização de Dados:** `matplotlib` integrado diretamente ao Tkinter (`FigureCanvasTkAgg`), permitindo a plotagem interativa de gráficos cartesianos, assíntotas, pontos críticos e de inflexão.

### Arquitetura de Funcionalidades Principais

O código está estruturado em uma classe principal de aplicação (`App`) que gerencia a renderização de instâncias isoladas para cada domínio matemático. As principais *features* implementadas incluem:

1. **Análise de Domínio e Imagem:**
   * Algoritmo complexo que detecta singularidades, restrições de denominadores, potências fracionárias e argumentos logarítmicos usando o SymPy, retornando uma análise formal de conjuntos (`Interval`, `Union`, `FiniteSet`).
2. **Motor de Limites e L'Hôpital:**
   * Cálculo de limites laterais e bilaterais.
   * Implementação iterativa da Regra de L'Hôpital para formas indeterminadas ($0/0$ ou $\infty/\infty$), demonstrando o passo a passo da derivação do numerador e denominador.
3. **Diferenciação Avançada:**
   * **Derivadas Simples e de Ordem Superior:** Cálculo simbólico com avaliação de retas tangentes em pontos específicos.
   * **Derivadas Parciais:** Suporte a funções multivariáveis, isolando variáveis para diferenciação.
   * **Derivação Implícita:** Utiliza a função `sp.idiff` para encontrar $dy/dx$ em equações não isoladas.
4. **Integração Numérica e Simbólica:**
   * Solução de integrais definidas (com limites de integração) e indefinidas (retornando a constante de integração).
5. **Plotagem Gráfica Inteligente:**
   * Suporte a funções normais e funções definidas por partes (piecewise).
   * Identificação e plotagem automática de:
     * Assíntotas (Verticais, Horizontais e Oblíquas).
     * Pontos Críticos (Máximos, Mínimos e Selas).
     * Pontos de Inflexão e preenchimento visual de áreas de concavidade (positiva/negativa).
     * Intervalos de crescimento e decrescimento.
   * Importação e interpolação cúbica de pontos de dados a partir de arquivos `.txt`.
