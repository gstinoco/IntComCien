# %% [markdown]
# # Poisson en 2D
# 
# A continuación se presentan ejemplos de implementaciones para calcular numericamente la solución de la ecuación de Poisson en dos dimensiones espaciales.
# 
# El problema a resolver es
# \begin{align}
#     \frac{\partial^2\phi}{\partial x^2}+ \frac{\partial^2\phi}{\partial y^2} = -f
# \end{align}
# 
# Sujeto a las condiciones:
# \begin{align}
#     \phi(x,y)_\Omega = 2e^{2x+y}
# \end{align}
# y
# \begin{align}
#     f(x,y) = 10e^{2x+y}
# \end{align}

# %% [markdown]
# ## Importación de librerias
# 
# Primero se hace la importación de las librerías necesarias para trabajar con cuestiones de computación científica en Python.

# %%
import numpy as np                                                              # Librería con funciones y arreglos de cómputo numérico.
import matplotlib.pyplot as plt                                                 # Nos permitirá graficar los resultados

# %% [markdown]
# ## Funciones para las condiciones
# Se definen las funciones que actuarán como las condiciones del problema frontera.
# 
# Estas pueden cambiarse dependiendo del problema a modelar con la finalidad de cambiar el problema que se pretende resolver.
# 
# En este caso, las funciones que se usarán son:
# \begin{align}
#     \phi(x,y) = 2e^{2x+y}
# \end{align}
# y
# \begin{align}
#     f(x,y) = 10e^{2x+y}
# \end{align}

# %%
def phi(x, y):                                                                  # Se define la función phi.
    phi = 2*np.exp(2*x+y)                                                       # Se agrega la expresión para la condición.
    return phi                                                                  # Regresa el valor de la condición evaluada.

def f(x,y):                                                                     # Se define la función f.
    f = 10*np.exp(2*x+y)                                                        # Se agrega la expresión para la condición.
    return f                                                                    # Regresa el valor de la condición evaluada.

# %% [markdown]
# ## Poisson 2D Matricial

# %% [markdown]
# ### Inicialización de Variables
# Se inicializan algunas variables para la generación de la malla de la región en la cual se pretende resolver el problema y acelerar algunos cálculos.

# %%
m      = 21                                                                     # Número de elementos que tendrá la discretización.
x      = np.linspace(0,1,m)                                                     # Se hace la discretización en x del intervalo [0, 1].
y      = np.linspace(0,1,m)                                                     # Se hace la discretización en y del intervalo [0, 1].
h      = x[2] - x[1]                                                            # Se calcula h para el método de Diferencias Finitas.
x, y   = np.meshgrid(x,y)                                                       # Se hace la malla de la región para trabajar [0,1]x[0,1].
A      = np.zeros([(m-2)*(m-2),(m-2)*(m-2)])                                    # Se inicializa la matriz A con ceros.
rhs    = np.zeros([(m-2)*(m-2),1])                                              # Se inicializa el vector rhs con ceros.
phi_ap = np.zeros([m,m])                                                        # Se inicializa la variable para la solución aproximada.
phi_ex = np.zeros([m,m])                                                        # Se inicializa la respectiva variable con ceros.

# %% [markdown]
# ### Condiciones de Frontera
# 
# Se deben de agregar las condiciones de frontera, para poder calcular correctamente la aproximación. En este caso, se debe de considerar que se tienen 4 fronteras diferentes.

# %%
for i in range(1,m-1):                                                          # Se recorren los nodos de la frontera.
    temp       = i-1                                                            # Se asigna el valor de temp como los nodos de la primera frontera en x.
    rhs[temp] += phi(x[i,0], y[i,0])                                            # Se guarda la condición de frontera en el lado derecho.
    temp       = (i-1) + (m-2)*((m-1)-2)                                        # Se asigna el valor de temp como los nodos de la última frontera en x.
    rhs[temp] += phi(x[i,m-1], y[i,m-1])                                        # Se guarda la condición de frontera en el lado derecho.
    temp       = (m-2)*(i-1)                                                    # Se asigna el valor de temp para los nodos de la primera frontera en y.
    rhs[temp] += phi(x[0,i], y[0,i])                                            # Se guarda la condición de frontera en el lado derecho.
    temp       = ((m-1)-2) + (m-2)*(i-1)                                        # Se asgina el valor de temp para los nodos de la última frontera en y.
    rhs[temp] += phi(x[m-1,i], y[m-1,i])                                        # Se guarda la condición de frontera en el lado derecho.

for i in range(1,m-1):                                                          # Para todos los nodos en x.
    for j in range(1,m-2):                                                      # Para todos los nodos en y.
        temp       = (i-1) + (m-2)*(j-1)                                        # Se buscan los nodos que son frontera.
        rhs[temp] += -(h**2)*f(x[i,j], y[i,j])                                  # Se agrega f al lado derecho.

# %% [markdown]
# ### Matriz de Diferencias Finitas
# Se ensambla la Matriz A, tridiagonal por bloques, de Diferencias Finitas.

# %%
dB   = np.diag(4*np.ones(m-2))                                                  # Se hace una matriz diagonal con 4s.
dBp1 = np.diag(1*np.ones((m-2)-1), k=1)                                         # Se hace una matriz identidad negativa inferior.
dBm1 = np.diag(1*np.ones((m-2)-1), k=-1)                                        # Se hace una matriz identidad negativa superior.
B    = (dB - dBp1 - dBm1)                                                       # Se ensamblan los bloques de la diagonal.
I    = -np.identity(m-2)                                                        # Se hace una matriz identidad negativa.
temp = 1                                                                        # Se inicializa un contador para guardar lo anterior en la matriz A.

for i in range(0,(m-2)*(m-2),(m-2)):                                            # Para cada uno de los bloques.
    A[i:temp*(m-2), i:temp*(m-2)] = B                                           # Se hace la diagonal interior de bloques B.
    if temp*(m-2) < (m-2)*(m-2):                                                # Si estamos arriba o abajo de la diagonal interior.
        A[temp*(m-2):temp*(m-2)+(m-2), i:temp*(m-2)] = I                        # Se pone una identidad negativa en la diagonal superior.
        A[i:temp*(m-2), temp*(m-2):temp*(m-2)+(m-2)] = I                        # Se pone una identidad negativa en la diagonal inferior.
    temp += 1                                                                   # Se aumenta el contador.

# %% [markdown]
# ### Resolver el problema
# Se resuelve el problema lineal
# \begin{align}
#     Au = rhs
# \end{align}
# 
# Además de que se agregan las condiciones de frontera a la aproximación calculada.

# %%
A  = np.linalg.pinv(A)                                                          # La inversa nos servirá para multiplicar por el lado derecho.
u = A@rhs                                                                       # Se multiplica la inversa por el lado derecho.
u = np.reshape(u, (m-2,m-2)).transpose()                                        # Se convierte el vector columna en matriz.

phi_ap[1:(m-1), 1:(m-1)] = u                                                    # Se guarda la aproximación calculada dentro de phi_ap.
for i in range(m):                                                              # Para todos los nodos.
    phi_ap[i,0] = phi(x[i,0],y[i,0])                                            # Se guarda la condición de frontera x = 0 en la aproximación.
    phi_ap[i,m-1] = phi(x[i,m-1],y[i,m-1])                                      # Se guarda la condición de frontera x = m en la aproximación.
    phi_ap[0,i] = phi(x[0,i],y[0,i])                                            # Se guarda la condición de frontera y = 0 en la aproximación.
    phi_ap[m-1,i] = phi(x[m-1,i],y[m-1,i])                                      # Se guarda la condición de frontera y = m en la aproximación.

# %% [markdown]
# ### Solución Exacta
# Se hace el cálculo de la solución exacta. En este caso, el problema toma como condiciones de frontera la solución exacta $\phi$.

# %%
for i in range(m):                                                              # Para cada uno de los elementos de la discretización en x.
    for j in range(m):                                                          # Para cada uno de los elementos de la discretización en y.
        phi_ex[i,j] = phi(x[i,j], y[i,j])                                       # Se asigna la solución exacta.

# %% [markdown]
# ### Graficación
# Se grafican la solución aproximada y la solución exacta lado a lado para ver que tan buena (o mala) es la aproximación.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})           # Se hace una figura con dos figuras incrustadas.
plt.rcParams["figure.figsize"] = (10,5)                                         # Se define el tamaño de la figura principal.
plt.suptitle('Ecuación de Poisson')                                             # Se pone un título a la figura principal.
min  = phi_ex.min()                                                             # Se encuentra el valor mínimo de la solución para ajustar la gráfica.
max  = phi_ex.max()                                                             # Se encuentra el valor máximo de la solución para ajustar la gráfica.

ax1.set_title('Solución Aproximada')                                            # Se pone el título de la primera figura incrustada.
ax1.plot_surface(x, y, phi_ap)                                                  # Se grafica la solución aproximada en la primera figura incrustada.
ax1.set_zlim([min,max])                                                         # Se establecen los ejes de la gráfica.

ax2.set_title('Solución Exacta')                                                # Se pone el título de la segunda figura incrustada.
ax2.plot_surface(x, y, phi_ex)                                                  # Se grafica la solución exacta en la segunda figura incrustada.
ax2.set_zlim([min,max])                                                         # Se establecen los ejes de la gráfica.
plt.show()                                                                      # Se muestra la figura.

# %% [markdown]
# ## Poisson 2D Iterativo

# %% [markdown]
# ### Inicialización de Variables
# Se inicializan algunas variables para la generación de la malla de la región en la cual se pretende resolver el problema y acelerar algunos cálculos.

# %%
m      = 21                                                                     # Número de elementos que tendrá la discretización.
x      = np.linspace(0,1,m)                                                     # Se hace la discretización en x del intervalo [0, 1].
y      = np.linspace(0,1,m)                                                     # Se hace la discretización en y del intervalo [0, 1].
h      = x[2] - x[1]                                                            # Se calcula h para el método de Diferencias Finitas.
x, y   = np.meshgrid(x,y)                                                       # Se hace la malla de la región para trabajar [0,1]x[0,1].
err    = 1                                                                      # Se inicializa una diferencia, err, en 1 para asegurarnos de que haga por lo menos una iteración.
tol    = np.sqrt(np.finfo(float).eps)                                           # Se establece una tolerancia que será la raíz cuadrada del épsilon de la computadora.
phi_ap = np.zeros([m,m])                                                        # Se inicializa la variable para la solución aproximada.
phi_ex = np.zeros([m,m])                                                        # Se inicializa la respectiva variable con ceros.

# %% [markdown]
# ### Condiciones de Frontera
# 
# Se deben de agregar las condiciones de frontera, para poder calcular correctamente la aproximación. En este caso, se debe de considerar que se tienen 4 fronteras diferentes.

# %%
for i in range(m):                                                              # Se recorren todos los nodos de frontera.
    phi_ap[i,0]  = phi(x[i,0], y[i,0])                                          # Se asigna el valor para la frontera derecha
    phi_ap[i,-1] = phi(x[i,-1], y[i,-1])                                        # Se asigna el valor para la frontera izquierda
    phi_ap[0,i]  = phi(x[0,i], y[0,i])                                          # Se asigna el valor para la frontera superior
    phi_ap[-1,i] = phi(x[-1,i], y[-1,i])                                        # Se asigna el valor para la frontera inferior

# %% [markdown]
# ### Resolver el problema
# En este caso, el problema se resolverá iterativamente, tomando en cuenta la tolerancia para el error cometido de itación en interación establecida anteriormente.

# %%
while err >= tol:
    err = 0
    for i in range(1,m-1):
        for j in range(1,m-1):
            t = (1/4)*(phi_ap[i-1,j] + phi_ap[i+1,j] + \
                phi_ap[i,j-1] + phi_ap[i,j+1] - \
                h**2*f(x[i,j],y[i,j]))
            err = max(err, abs(t - phi_ap[i,j]))
            phi_ap[i,j] = t

# %% [markdown]
# ### Solución Exacta
# Se hace el cálculo de la solución exacta. En este caso, el problema toma como condiciones de frontera la solución exacta $\phi$.

# %%
for i in range(m):                                                              # Para cada uno de los elementos de la discretización en x.
    for j in range(m):                                                          # Para cada uno de los elementos de la discretización en y.
        phi_ex[i,j] = phi(x[i,j], y[i,j])                                       # Se asigna la solución exacta.

# %% [markdown]
# ### Graficación
# Se grafican la solución aproximada y la solución exacta lado a lado para ver que tan buena (o mala) es la aproximación.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})           # Se hace una figura con dos figuras incrustadas.
plt.rcParams["figure.figsize"] = (10,5)                                         # Se define el tamaño de la figura principal.
plt.suptitle('Ecuación de Poisson')                                             # Se pone un título a la figura principal.
min  = phi_ex.min()                                                             # Se encuentra el valor mínimo de la solución para ajustar la gráfica.
max  = phi_ex.max()                                                             # Se encuentra el valor máximo de la solución para ajustar la gráfica.

ax1.set_title('Solución Aproximada')                                            # Se pone el título de la primera figura incrustada.
ax1.plot_surface(x, y, phi_ap)                                                  # Se grafica la solución aproximada en la primera figura incrustada.
ax1.set_zlim([min,max])                                                         # Se establecen los ejes de la gráfica.

ax2.set_title('Solución Exacta')                                                # Se pone el título de la segunda figura incrustada.
ax2.plot_surface(x, y, phi_ex)                                                  # Se grafica la solución exacta en la segunda figura incrustada.
ax2.set_zlim([min,max])                                                         # Se establecen los ejes de la gráfica.
plt.show()                                                                      # Se muestra la figura.


