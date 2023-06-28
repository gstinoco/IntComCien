import numpy as np                                                              # Librería con funciones y arreglos de cómputo numérico.
import matplotlib.pyplot as plt                                                 # Nos permitirá graficar los resultados.

def u(x,t,v):                                                                   # Se define la función u.
    u = np.exp(-np.pi**2*v*t)*np.cos(np.pi*x)                                   # El valor de la función.
    return u                                                                    # Se regresa el valor de u.

m    = 5                                                                       # Número de elementos que tendrá la discretización.
t    = 100                                                                      # Número de pasos en el tiempo.
v    = 0.3                                                                      # Coeficiente de difusión.

x    = np.linspace(0,1,m)                                                       # Se hace la discretización del intervalo [0, 1].
T    = np.linspace(0,1,t)                                                       # Se hace la discretización temporal del intervalo [0,1]
dx   = x[1] - x[0]                                                              # Se calcula dx para el método de Diferencias Finitas.
dt   = T[1] - T[0]                                                              # Se calcula dt para el método de Diferencias Finitas.
u_ap = np.zeros([m,t])                                                          # Se inicializa la variable para la solución aproximada.

for i in range(m):                                                              # Para cada uno de los nodos.
    u_ap[i,0] = u(x[i], T[0],v)                                                 # Se asigna la condición inicial.

for k in range(t):                                                              # Para cada paso de tiempo.
    u_ap[0,k]  = u(x[0], T[k],v)                                                # Se agrega la condición de frontera en x = 0.
    u_ap[-1,k] = u(x[-1], T[k],v)                                               # Se agrega la condición de frontera en x = 1.


for k in range(1, t):                                                           # Para todos los pasos de tiempo.
    for i in range(1,m-1):                                                      # Para todos los nodos.
        u_ap[i,k] = u_ap[i,k-1] + dt*(v/dx**2)*( \
                    u_ap[i+1, k-1] - 2*u_ap[i, k-1] + u_ap[i-1,k-1])            # Se calcula la solución aproximada
    
u_ex = np.zeros([200,t])                                                        # Se inicializa u_ex para guardar la solución exacta.
x2   = np.linspace(0,1,200)                                                     # Se hace una malla "fina" para mostrar mejor la solución.
for k in range(t):                                                              # Para todos los tiempos.
    for i in range(200):                                                        # Para todos los nodos.
        u_ex[i,k] = u(x2[i], T[k], v)                                           # Se guarda la solución exacta.

fig, (ax1, ax2) = plt.subplots(1, 2)                                            # Se hace una figura con dos figuras incrustadas.
plt.rcParams["figure.figsize"] = (10,5)                                         # Se define el tamaño de la figura principal.
plt.suptitle('Ecuación de Difusión')                                           # Se pone un título a la figura principal.
min  = u_ex.min()                                                               # Se encuentra el valor mínimo de la solución.
max  = u_ex.max()                                                               # Se encuentra el valor máximo de la solución.
p = int(np.ceil(t/100))                                                         # Se decide cuantos pasos de tiempo mostrar.

for i in range(0,t,p):                                                          # Para el tiempo desde 0 hasta 1.
    ax1.plot(x, u_ap[:,i])                                                      # Se grafica la solución aproximada en la primera figura incrustada.
    ax1.set_ylim([min,max])                                                     # Se fijan los ejes en y.
    ax1.set_title('Solución Aproximada')                                        # Se pone el título de la primera figura incrustada.
    
    ax2.plot(x2, u_ex[:,i])                                                     # Se grafica la solución exacta en la segunda figura incrustada.
    ax2.set_ylim([min,max])                                                     # Se fijan los ejes en y.
    ax2.set_title('Solución Exacta')                                            # Se pone el título de la segunda figura incrustada.
    
    plt.pause(0.01)                                                             # Se muestra la figura.
    ax1.clear()                                                                 # Se limpia la gráfica de la primera figura.
    ax2.clear()                                                                 # Se limpia la gráfica de la segunda figura.

plt.show()                                                                      # Se muestra el último paso de tiempo.
