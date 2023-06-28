import numpy as np                                                              # Librería con funciones y arreglos de cómputo numérico.
import matplotlib.pyplot as plt                                                 # Nos permitirá graficar los resultados.
from Graphs import Mesh_Transient                                               # Las rutina para graficar está en el archivo Graphs.


def u(x,y,t,v):                                                                 # Se define la función u.
    u = np.exp(-2*np.pi**2*v*t)*np.cos(np.pi*x)*np.cos(np.pi*y);                # Se agrega la expresión para la condición.
    return u                                                                    # Regresa el valor de la condición.


m    = 21                                                                       # Número de elementos que tendrá la discretización en x.
n    = 21                                                                       # Número de elementos que tendrá la discretización en y.
t    = 500                                                                      # Número de pasos en el tiempo.
v    = 0.1                                                                      # Coeficiente de Difusión.

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
        u_ap[i,j,0] = u(x[i,j], y[i,j], T[0], v)                                # Se asigna la condición inicial.

for k in range(t):                                                              # Para cada paso de tiempo.
    for i in range(m):                                                          # Se recorren los nodos en x.
        u_ap[i,0,k]  = u(x[i,0], y[i,0], T[k], v)                               # Se agrega la condición de frontera.
        u_ap[i,-1,k] = u(x[i,-1], y[i,-1], T[k], v)                             # Se agrega la condición de frontera.
    for j in range(n):                                                          # Se recorren los nodos en y.
        u_ap[0,j,k]  = u(x[0,j], y[0,j], T[k], v)                               # Se agrega la condición de frontera.
        u_ap[-1,j,k] = u(x[-1,j], y[-1,j], T[k], v)                             # Se agrega la condición de frontera.


for k in range(1, t):                                                           # Para todos los pasos de tiempo.
    for i in range(1,m-1):                                                      # Para todos los nodos en x.
        for j in range(1,n-1):                                                  # Para todos los nodos en y.
            u_ap[i,j,k] = u_ap[i,j,k-1] + dt*( \
                        (v/dx**2)* \
                        (u_ap[i+1,j,k-1] - 2*u_ap[i,j,k-1] + u_ap[i-1,j,k-1]) + \
                        (v/dy**2)* \
                        (u_ap[i,j+1,k-1] - 2*u_ap[i,j,k-1] + u_ap[i,j-1,k-1]))  # Se calcula la aproximación.
            

m2    = 100
n2    = 100
u_ex  = np.zeros([m2,n2,t])                                                     # Se inicializa u_ex para guardar la solución exacta.
x2    = np.linspace(0,1,m2)                                                     # Se hace una malla "fina" para mostrar mejor la solución.
y2    = np.linspace(0,1,n2)                                                     # Se hace una malla "fina" para mostrar mejor la solución.
x2,y2 = np.meshgrid(x2,y2)                                                      # Se hace una malla "fina" para mostrar mejor la solución.
for k in range(t):                                                              # Para todos los tiempos.
    for i in range(m2):                                                         # Para todos los nodos en x.
        for j in range(n2):                                                     # Para todos los nodos en y.
            u_ex[i,j,k] = u(x2[i,j], y2[i,j], T[k], v)                          # Se guarda la solución exacta.


Mesh_Transient(x, y, u_ap, x2, y2, u_ex)