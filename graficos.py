import numpy as np
import matplotlib.pyplot as plt
import functions
import os

#import matplotlib.patches as patches

def plot_orbita(titulo, X, Y, cor, cor_ponto, mu2, mu1, titgraf):
    plt.figure(titulo)
    plt.plot(X, Y, color=cor, label=titulo)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.25)
    plt.scatter(-mu2, 0, color=cor_ponto, label=r'$M_1$',s=100)
    plt.scatter(mu1, 0, color=cor_ponto, label=r'$M_2$',s=50)
    #plt.legend(loc=1)
    plt.title(r'%s' % titgraf)
    
    caminho = os.path.join("Órbitas", f"{titulo}.jpg")
    
    # salvando Figura
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    
    plt.close(titulo)

def plot_curvas(mu1, mu2, titulo, niveis_C, pi):
    #----------------------------Grid-------------------------------------------
    # Cria (grid) para avaliar a função
    x_vals = np.linspace(-2.0, 2.0, 400)
    y_vals = np.linspace(-2.0, 2.0, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calcula o valor de C chamando a sua função modificada diretamente!
    Z = functions.jacobi(mu1, mu2, X, Y)
    
    # Blindagem contra a divisão por zero exatamente em cima das massas
    # Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=0.0)

    #---------------------------Gráfico------------------------------------------

    plt.figure(figsize=(8, 8))

    # Desenha as Curvas de Velocidade Zero (Contornos)
    curvas = plt.contour(X, Y, Z, levels=niveis_C, colors='blue', linewidths=1)
    plt.clabel(curvas, inline=True, fontsize=6) # Adiciona o valor de C em cima da linha

    # Plotando as massas mu1 e mu2
    plt.scatter(-mu2, 0, color='black', s=120, label=r'$m_1$')
    plt.scatter(mu1, 0, color='black', s=50, label=r'$m_2$')

    # Plotando o ponto inicial:
    plt.scatter(pi[0], pi[1], s=30, color='purple', marker='o', label=r'$(x_0, y_0)$')

    # Plotando os pontos de equilíbrio
    pontos = functions.pontos(mu1, mu2)
    for malu in pontos:
        if malu[1] != 0:
            plt.scatter(malu[0], malu[1], s=30, color='green', marker='x')
        else:
            plt.scatter(malu[0], malu[1], s=30, color='red', marker='x')

    # Customizações do gráfico
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal') # Mantém a proporção 1:1 perfeita
    plt.legend(loc='upper right')

    caminho = os.path.join("Curvas de velocidade zero", titulo)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')

    plt.close(titulo)