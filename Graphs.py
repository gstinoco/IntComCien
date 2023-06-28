import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def Mesh_Transient(x, y, u_ap, x2, y2, u_ex):
    """
    Mesh_Transient

    This function graphs the approximated and theoretical solutions of the problem being solved at several time levels.
    Both solutions are presented side by side to help perform graphical comparisons between both solutions.
    
    Input:
        x           m x n           Array           Array with the x-coordinates of the nodes.
        y           m x n           Array           Array with the y-coordinates of the nodes.
        u_ap        m x n x t       Array           Array with the computed solution.
        u_ex        m x n x t       Array           Array with the theoretical solution.
    
    Output:
        None
    """
    t    = len(u_ex[0,0,:])
    step = int(np.ceil(t/50))
    min  = u_ex.min()
    max  = u_ex.max()
    T    = np.linspace(0,1,t)
    
    for k in np.arange(0,t,step):
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))
        tin = float(T[k])
        plt.suptitle('Solución al tiempo t = %1.3f s.' %tin)
        
        ax1.plot_surface(x, y, u_ap[:,:,k], cmap=cm.coolwarm)
        ax1.set_zlim([min, max])
        ax1.set_title('Aproximación')

        ax2.plot_surface(x2, y2, u_ex[:,:,k], cmap=cm.coolwarm)
        ax2.set_zlim([min, max])
        ax2.set_title('Solución Exacta')

        plt.pause(0.01)
        plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw = {"projection": "3d"}, figsize=(8, 4))
    tin = float(T[t-1])
    plt.suptitle('Solución al tiempo t = %1.3f s.' %tin)
    
    ax1.plot_surface(x, y, u_ap[:,:,t-1], cmap=cm.coolwarm)
    ax1.set_zlim([min, max])
    ax1.set_title('Aproximación')

    ax2.plot_surface(x2, y2, u_ex[:,:,t-1], cmap=cm.coolwarm)
    ax2.set_zlim([min, max])
    ax2.set_title('Solución Exacta')

    plt.show()