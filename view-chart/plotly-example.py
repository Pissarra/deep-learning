import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

# Setores

votos = (842.201, 488.775, 553.424, 424.307, 272.500, 381.512, 261.386)

candidatos = ['Candidato A', 'Candidato B', 'Candidato C', 'Candidato D', 'Candidato E', 'Candidato F', 'Candidato G']

# determina as cores das variáveis no gráfico

cores = ['gold', 'salmon', 'orange', 'oranged', 'seagreen', 'skyblue', 'yellowgreen']

# o atributo explode indica que fatia do gráfico será destacada. No exemplo abaixo, será a primeira fatia.

explode = (0.1, 0, 0, 0, 0, 0, 0)  # explode a 1a fatia

# atribui um título ao gráfico

plt.title('Eleição 2022 - Votos por candidato')

# adiciona Legenda

plt.legend(candidatos)

# centraliza o gráfico

plt.axis('equal')

# ajusta o espaçamento para evitar o recorte do rótulo

plt.tight_layout()

# autopact atribui os valores em percentual nas fatias

# plt.pie(votos,  labels=candidatos, autopct='%1.1f%%')
#
# plt.show()

# 02 - Sunburst


data = dict(

    character=["Maria", "Augusto", "Sergio", "Henrique", "Nuno", "Abelardo", "Jorge", "Manoel", "Ana"],

    parent=["", "Maria", "Maria", "Sergio", "Maria", "Maria", "Jorge", "Maria", ""],

    value=[1, 1, 1, 1, 1, 1, 1, 1, 1])

fig = px.sunburst(

    data,

    names='character',

    parents='parent',

    values='value',

)

fig.show()

# Aplicar o tema padrão

sns.set_theme()

# Abrir um data set exemplo, este está disponível na própria documentação do Seaborn

print(sns)

tips = sns.load_dataset("tip")

# Criar uma visualização

sns.relplot(data=tips, x="tip", y="total_bill", col="time", hue="smoker", style="smoker", size="size")