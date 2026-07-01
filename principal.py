# Programa que propaga uma órbita no PR3C  - no sistema girante com origem no centro de massa
# Calcula as curvas de veloidade zero junto com os pontos de equilíbrio

# Biliotecas
import os
import numpy as np
import functions
from graficos import plot_orbita, plot_curvas

with open('parametros.txt', 'r') as param:
    parametros = []
    for _ in range(7):
        next(param)
        linha = next(param)      
        parametros.append(linha.strip())

    tempo_inicial = float(parametros[0])
    tempo_final = float(parametros[1])
    num_passos = int(parametros[2])
    m1 = float(parametros[3])
    m2 = float(parametros[4])
    raio = float(parametros[5])
    tipo = parametros[6]

    # Parâmetros de massa (normalizado)
    G = 6.6743e-11
    mu2 = m2 / (m1 + m2)
    mu1 = 1.0 - mu2 # m1/m1+m2

    next(param)
    for indice, linha in enumerate(param):
        elementos = linha.strip().split()
        # valores = [x, y, Vx, Vy, período]
        estado_inicial = [float(x) for x in elementos[:6]]
    
        if tipo == 'e':
            # Problema direto
            posicao_i, velocidade_i = functions.direto(estado_inicial, m1, m2)
            estado_inicial = functions.normalizar_e_girar(posicao_i, velocidade_i, tempo_inicial, raio, G*(m1 + m2))
        
        # Vetor de estado inicial
        # valores (x,y,z,Vx,Vy,Vz) de M3 normalizados.
        estado = np.array(estado_inicial)

        # Preparação da solução e Integração da órbita  
        solucao = np.zeros((num_passos + 1, 6))
        solucao[0, :] = estado
        #tempo = tempo_inicial
        #estado = estado_inicial.copy()
        passo_tempo = (tempo_final - tempo_inicial) / num_passos

        # Integrador
        solucao = functions.rk4(functions.derivada, estado, tempo_inicial, passo_tempo, num_passos, mu2, mu1)

        # Salvando os dados de saida
        tempos = np.linspace(tempo_inicial, tempo_final, num_passos+1)
        dados = np.column_stack((tempos, solucao))
        Path = os.path.join("Soluções", f"Solucao{indice}.dat") # --> salva na pasta "Soluções"
        np.savetxt(Path, dados)

        # salvando pontos de equilíbrio
        functions.salvar_pontos(mu1, mu2, "pontos_de_equilíbrio.dat")
        
        # Gráficos
        plot_orbita(f'Órbita{indice}', solucao[:, 0], solucao[:, 1], 'blue', 'black', mu2, mu1, 'Órbita no PR3C')

        # Plotando curvas de velocidade zero:
        pi = [estado_inicial[0], estado_inicial[1]]
        jacobi = [functions.jacobi(mu1, mu2, estado_inicial[0], estado_inicial[1])]

        # níveis da constante de Jacobi (curvas de velocidade zero)
        niveis_C = np.linspace(1, 4, 20)
        plot_curvas(mu1, mu2, "Curvas de Velocidade Zero (PR3C)", niveis_C, pi, jacobi)