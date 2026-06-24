import numpy as np
import math as m
import csv
import os

# Função para calcular a derivada do estado (x,y,z, Vx, Vy, Vz)
def derivada(t, estado, mu2, mu1):
#	
    r1 = (np.sqrt((estado[0] + mu2)**2 + (estado[1])**2 + (estado[2])**2))**3
    r2 = (np.sqrt((estado[0]-1.0 - mu2)**2 + (estado[1])**2 + (estado[2])**2))**3
    f6 =   2.0*estado[4] + estado[0] - (mu1*(estado[0] + mu2))/ r1 - (mu2*(estado[0] -1.0 + mu2))/ r2
    f7 = - 2.0*estado[3] + estado[1] - (mu1*estado[1])/ r1 - (mu2*estado[1])/ r2
    f8 =   (mu1*estado[2])/ r1 - (mu2*estado[2])/ r2
#    
    return np.array([estado[3], estado[4], estado[5], f6, f7, f8])

# Função Runge Kutta de 4 ordem para integrar a EDO    
def rk4(derivada, estado_inicial, tempo_inicial, passo_tempo, num_passos, mu2, mu1):
    n = len(estado_inicial)
    solucao = np.zeros((num_passos + 1, n))

    estado = estado_inicial.copy()
    tempo = tempo_inicial

    solucao[0, :] = estado

    for k in range(num_passos):
        k1 = passo_tempo * derivada(tempo, estado, mu2, mu1)
        k2 = passo_tempo * derivada(tempo + passo_tempo / 2, estado + k1 / 2, mu2, mu1)
        k3 = passo_tempo * derivada(tempo + passo_tempo / 2, estado + k2 / 2, mu2, mu1)
        k4 = passo_tempo * derivada(tempo + passo_tempo, estado + k3, mu2, mu1)

        estado = estado + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
        tempo = tempo + passo_tempo

        solucao[k + 1, :] = estado

    return solucao

def pontos(mu1, mu2):
    # Parametro para L1 e L2
    alpha = (mu2 / (3*mu1))**(1/3)
    
    # Ponto L1 (entre as duas massas)
    r = alpha - (alpha**2 / 3) - (alpha**3 / 9) - (23 * alpha**4)/81
    L1 = [(1 - mu2) - r, 0]

    # Ponto L2 (após a massa mu2)
    r = alpha + (alpha**2 / 3) - (alpha**3 / 9) - (31 * alpha**4)/81
    L2 = [(1 - mu2) + r, 0]

    # Ponto L3 (após a massa mu1)
    beta = ((- 7/12) * (mu2 / mu1)) + ((7/12) * (mu2 / mu1)**2) - ((13223/20736) * (mu2 / mu1)**3)
    L3 = [- mu2 - (1 + beta), 0]
    
    # Ponto L4 (vértice de cima do triângulo equilátero)
    L4 = [0.5 - mu2, (3**0.5) / 2]
    
    # Ponto L5 (vértice de baixo do triângulo equilátero)
    L5 = [0.5 - mu2, -(3**0.5 / 2)]

    return L1, L2, L3, L4, L5

def salvar_pontos(mu1, mu2, nome_arquivo):
    L1, L2, L3, L4, L5 = pontos(mu1, mu2)
    
    # Achata as coordenadas para virar uma lista de 10 números
    coords_pontos = L1 + L2 + L3 + L4 + L5
    
    caminho = os.path.join("Pontos de equilíbrio", nome_arquivo)

    # Abre o arquivo e grava a linha
    with open(caminho, mode="w", newline="") as f:
        f.write('# Esse e o arquivo das coordenadas x e y dos pontos de equilibrio que vao de L1 a L5, ou seja: xL1 yL1 xL2 yL2 xL3 yL3 xL4 yL4 xL5 yL5\n\n')

        # Usando '\t' para separar por colunas tabuladas
        escritor = csv.writer(f, delimiter="\t")
        escritor.writerow(coords_pontos)

def jacobi(mu1, mu2, x, y):
    r1 = np.sqrt((x + mu2)**2 + y**2)
    r2 = np.sqrt((x - mu1)**2 + y**2)

    # Equação da Constante de Jacobi (Curvas de velocidade zero)
    jacobi = x**2 + y**2 + 2 * (mu1 / r1 + mu2 / r2)
    return jacobi

def newtonraphson(M, e):
    un = M  # Chute inicial
    for _ in range(100): # Evita loops infinitos se não convergir
        u = un - (un - e * m.sin(un) - M) / (1 - e * m.cos(un))
        if abs(u - un) < 1e-5:
            return u
        un = u
    return un

def direto(val, m1, m2):
    # val = [a, e, M, i, Ω, ω]
    # Índices corretos:
    
    a = val[0]
    e = val[1]
    M = val[2]
    i = val[3]
    Omega = val[4]
    omega = val[5]

    # 1. Resolve Anomalia Excêntrica com índice correto (e = val[1])
    u = newtonraphson(M, e)

    # 2. Mean motion (Movimento Médio) em unidades físicas reais
    G = 6.6743e-11
    n = m.sqrt(G * (m1 + m2) / a**3)

    # 3. Vetor Posição e Velocidade no Plano da própria órbita
    r = a * (1 - e * m.cos(u))
    
    x = a * (m.cos(u) - e)
    y = a * m.sin(u) * m.sqrt(1 - e**2)
    posicao = np.array([x, y, 0.0])

    Vx = (-n * a**2 * m.sin(u)) / r
    Vy = (n * a**2 * m.cos(u) * m.sqrt(1 - e**2)) / r
    velocidade = np.array([Vx, Vy, 0.0])

    # 4. Matriz de Rotação para o inercial 3D
    cos_O, sin_O = m.cos(Omega), m.sin(Omega)
    cos_i, sin_i = m.cos(i), m.sin(i)
    cos_w, sin_w = m.cos(omega), m.sin(omega)

    m = np.array([
        [cos_O*cos_w - sin_O*cos_i*sin_w, -cos_O*sin_w - sin_O*cos_i*cos_w,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*cos_i*sin_w, -sin_O*sin_w + cos_O*cos_i*cos_w, -cos_O*sin_i],
        [sin_i*sin_w,                      sin_i*cos_w,                      cos_i]
    ])

    # Posição e Velocidade finais em um Referencial INERCIAL de coordenadas reais (ex: km e km/s)
    pos_inercial = m @ posicao
    vel_inercial = m @ velocidade

    return pos_inercial, vel_inercial

def normalizar_e_girar(posicao, velocidade, tempo_inicial, raio, mu):
    # 1. Fatores de escala para normalização
    # São como uma unidade padrão
    L = raio
    T = np.sqrt((raio**3) / mu)
    V = L / T
    
    # 2. Noramlização dos valore inerciais
    pos_inercial = np.array(posicao) / L
    vel_inercial = np.array(velocidade) / V
    tempo = tempo_inicial / T
    
    # 3. Rotacionar para o referencial girante (PR3C)
    xi, yi, zi = pos_inercial
    Vxi, Vyi, Vzi = vel_inercial
    
    # Posições no sistema girante
    xg = xi*np.cos(tempo) + yi*np.sin(tempo)
    yg = -xi*np.sin(tempo) + yi*np.cos(tempo)
    zg = zi

    # Velocidades no sistema girante
    Vxg = (Vxi + yi) * np.cos(tempo) + (Vyi - xi) * np.sin(tempo)
    Vyg = -(Vxi + yi) * np.sin(tempo) + (Vyi - xi) * np.cos(tempo)
    Vzg = Vzi
    
    # vetor de estado inicial
    return [xg, yg, zg, Vxg, Vyg, Vzg]