import numpy as np                                                              # Librería con funciones y arreglos de cómputo numérico.
import matplotlib.pyplot as plt                                                 # Nos permitirá graficar los resultados.
from Graphs import Mesh_Transient                                               # Las rutina para graficar está en el archivo Graphs.


def u(x,y,t,a,b):                                                               # Se define la función u.
    u = 0.2*np.exp((-(x-.5-a*t)**2-(y-.3-b*t)**2)/.01)                          # Se agrega la expresión para la condición.
    return u                                                                    # Regresa el valor de la condición.


m    = 41                                                                       # Número de elementos que tendrá la discretización en x.
n    = 41                                                                       # Número de elementos que tendrá la discretización en y.
t    = 100                                                                      # Número de pasos en el tiempo.
a    = 0.2                                                                      # Velocidad de transporte en x.
b    = 0.2                                                                      # Velocidad de transporte en y.

x    = np.linspace(0,1,m)                                                       # Se hace la discretización del intervalo [0, 1] para x.
y    = np.linspace(0,1,n)                                                       # Se hace la discretización del intervalo [0, 1] para y.
dx   = x[1] - x[0]                                                              # Se calcula dx para el método de Diferencias Finitas.
dy   = y[1] - y[0]                                                              # Se calcula dy para el método de Diferencias Finitas.
x, y = np.meshgrid(x,y)                                                         # Se genera la malla de la región [0,1]X[0,1].

T    = np.linspace(0,1,t)                                                       # Se hace la discretización temporal del intervalo [0,1]
dt   = T[1] - T[0]                                                              # Se calcula dt para el método de Diferencias Finitas.

u_ap = np.zeros([m,n,t])                                                        # Se inicializa la variable para la solución aproximada.


for i in range(m):                                                              # Para cada uno de los nodos en x.
    for j in range(n):                                                          # Para cada uno de los nodos en y.
        u_ap[i,j,0] = u(x[i,j], y[i,j], T[0], a, b)                             # Se asigna la condición inicial.

for k in range(t):                                                              # Para cada paso de tiempo.
    for i in range(m):                                                          # Se recorren los nodos en x.
        u_ap[i,0,k] = u(x[i,0], y[i,0], T[k], a, b)                             # Se agrega la condición de frontera.
    for j in range(n):                                                          # Se recorren los nodos en y.
        u_ap[0,j,k] = u(x[0,j], y[0,j], T[k], a, b)                             # Se agrega la condición de frontera.


for k in range(1, t):                                                           # Para todos los pasos de tiempo.
    for i in range(1,m):                                                        # Para todos los nodos en x.
        for j in range(1,n):                                                    # Para todos los nodos en y.
            u_ap[i,j,k] = u_ap[i,j,k-1] - dt*( \
                          (a/dx)*(u_ap[i,j,k-1] - u_ap[i-1,j,k-1]) + \
                          (b/dy)*(u_ap[i,j,k-1] - u_ap[i,j-1,k-1]))             # Se calcula la aproximación.


u_ex  = np.zeros([200,200,t])                                                   # Se inicializa u_ex para guardar la solución exacta.
x2    = np.linspace(0,1,200)                                                    # Se hace una malla "fina" para mostrar mejor la solución.
y2    = np.linspace(0,1,200)                                                    # Se hace una malla "fina" para mostrar mejor la solución.
x2,y2 = np.meshgrid(x2,y2)                                                      # Se hace una malla "fina" para mostrar mejor la solución.
for k in range(t):                                                              # Para todos los tiempos.
    for i in range(200):                                                        # Para todos los nodos en x.
        for j in range(200):                                                    # Para todos los nodos en y.
            u_ex[i,j,k] = u(x2[i,j], y2[i,j], T[k], a, b)                       # Se guarda la solución exacta.


Mesh_Transient(x, y, u_ap, x2, y2, u_ex)