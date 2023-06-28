import numpy as np

def Mesh(x, y, f, v, t, implicit = False, lam = 0.5):
    """
    2D Diffusion Equation implemented in Logically Rectangular Meshes.

    This routine calculates an approximation to the solution of Diffusion equation in 2D using a Generalized Finite Differences scheme in logically rectangular meshes.
    
    The problem to solve is:
     
    \frac{\partial u}{\partial t}= v\nabla^2 u
     
    Input:
        x           m x n           Array           Array with the coordinates in x of the nodes.
        y           m x n           Array           Array with the coordinates in y of the nodes.
        f                           Function        Function declared with the boundary condition.
        v                           Real            Diffusion coefficient.
        t                           Integer         Number of time steps considered.
        implicit                    Logical         Select whether or not use an implicit scheme.
                                                        True: Implicit scheme used.
                                                        False: Explicit scheme used (Default).
        lam                         Real            Lambda parameter for the implicit scheme.
                                                        Must be between 0 and 1 (Default: 0.5).
    
    Output:
        u_ap        m x n x t       Array           Array with the approximation computed by the routine.
        u_ex        m x n x t       Array           Array with the theoretical solution.
    """

    # Variable initialization
    m    = len(x[:,0])                                                              # The number of nodes in x.
    n    = len(x[0,:])                                                              # The number of nodes in y.
    T    = np.linspace(0,1,t)                                                       # Time discretization.
    dt   = T[1] - T[0]                                                              # dt computation.
    u_ap = np.zeros([m, n, t])                                                      # u_ap initialization with zeros.
    u_ex = np.zeros([m, n, t])                                                      # u_ex initialization with zeros.
    urr  = np.zeros([m*n, 1])                                                       # u_rr initialization with zeros.

    # Boundary conditions
    for k in np.arange(t):
        for i in np.arange(m):                                                      # For each of the nodes on the x boundaries.
            u_ap[i, 0,   k] = f(x[i, 0], y[i, 0], T[k], v)                          # The boundary condition is assigned at the first y.
            u_ap[i, n-1, k] = f(x[i, n-1], y[i, n-1], T[k], v)                      # The boundary condition is assigned at the last y.
        for j in np.arange(n):                                                      # For each of the nodes on the y boundaries.
            u_ap[0,   j, k] = f(x[0, j], y[0, j], T[k], v)                          # The boundary condition is assigned at the first x.
            u_ap[m-1, j, k] = f(x[m-1, j], y[m-1, j], T[k], v)                      # The boundary condition is assigned at the last x.
  
    # Initial condition
    for i in np.arange(m):                                                          # For each of the nodes on x.
        for j in np.arange(n):                                                      # For each of the nodes on y.
            u_ap[i, j, 0] = f(x[i, j], y[i, j], T[0], v)                            # The initial condition is assigned.

    # Computation of K with Gammas
    L  = np.vstack([[0], [0], [2*v*dt], [0], [2*v*dt]])                             # The values of the differential operator are assigned.
    K  = Gammas(x, y, L)                                                            # K computation that include the Gammas.

    if implicit == False:                                                           # For the explicit scheme.
        K2 = np.identity(m*n) + K                                                   # Kp with an explicit formulation.
    else:                                                                           # For the implicit scheme.
        K2 = np.linalg.pinv(np.identity(m*n) \
                            - (1-lam)*K)@(np.identity(m*n) + lam*K)                 # Kp with an explicit formulation.

    # A Generalized Finite Differences Method
    for k in np.arange(1,t):                                                        # For each time step.
        for i in np.arange(m):                                                      # For each of the nodes on x.
            for j in np.arange(n):                                                  # For each of the nodes on y.
                urr[i + j*m, 0] = u_ap[i, j, k-1]                                   # urr as a row vector with all the solution.
                
        un = K2@urr                                                                 # New time level is computed.

        for i in np.arange(1,m-1):                                                  # For each of the interior nodes on x.
            for j in np.arange(1,n-1):                                              # For each of the interior nodes on y.
                u_ap[i, j, k] = un[i + j*m]                                         # u_ap values are assigned.

    # Theoretical Solution
    for k in np.arange(t):                                                          # For all the time steps.
        for i in np.arange(m):                                                      # For all the nodes on x.
            for j in np.arange(n):                                                  # For all the nodes on y.
                u_ex[i, j, k] = f(x[i, j], y[i, j], T[k], v)                        # The theoretical solution is computed.

    return u_ap, u_ex


def Gammas(x, y, L):
    """
    2D Logically Rectangular Meshes Gammas Computation.
     
    This function computes the Gamma values for Logically Rectangular Meshes, and assemble the K matrix for the computations.
     
    Input:
        x           m x n           Array           Array with the coordinates in x of the nodes.
        y           m x n           Array           Array with the coordinates in y of the nodes.
        L           5 x 1           Array           Array with the values of the differential operator.
     
     Output:
        K           m x m           Array           K Matrix with the computed Gammas.
    """
    # Variable initialization
    m  = len(x[:,0])                                                                # The number of nodes in x.
    n  = len(x[0,:])                                                                # The number of nodes in y.
    K  = np.zeros([(m)*(n), (m)*(n)])                                               # K initialization with zeros.

    # Gammas computation and Matrix assembly
    for i in np.arange(1,m-1):                                                      # For each of the inner nodes on x.
        for j in np.arange(1,n-1):                                                  # For each of the inner nodes on y.
            u  = np.array(x[i-1:i+2, j-1:j+2])                                      # u is formed with the x-coordinates of the stencil.
            v  = np.array(y[i-1:i+2, j-1:j+2])                                      # v is formed with the y-coordinates of the stencil
            dx = np.hstack([u[0,0] - u[1,1], u[1,0] - u[1,1], \
                            u[2,0] - u[1,1], u[0,1] - u[1,1], \
                            u[2,1] - u[1,1], u[0,2] - u[1,1], \
                            u[1,2] - u[1,1], u[2,2] - u[1,1]])                      # dx computation.
            dy = np.hstack([v[0,0] - v[1,1], v[1,0] - v[1,1], \
                            v[2,0] - v[1,1], v[0,1] - v[1,1], \
                            v[2,1] - v[1,1], v[0,2] - v[1,1], \
                            v[1,2] - v[1,1], v[2,2] - v[1,1]])                      # dy computation
            M  = np.vstack([[dx], [dy], [dx**2], [dx*dy], [dy**2]])                 # M matrix is assembled.
            M  = np.linalg.pinv(M)                                                  # The pseudoinverse of matrix M.
            YY = M@L                                                                # M*L computation.
            Gamma = np.vstack([-sum(YY), YY])                                       # Gamma values are found.
            p           = m*(j) + i                                                 # Variable to find the correct position in the Matrix.
            K[p, p]     = Gamma[0]                                                  # Gamma 0 assignation
            K[p, p-1-m] = Gamma[1]                                                  # Gamma 1 assignation
            K[p, p-m]   = Gamma[2]                                                  # Gamma 2 assignation
            K[p, p+1-m] = Gamma[3]                                                  # Gamma 3 assignation
            K[p, p-1]   = Gamma[4]                                                  # Gamma 4 assignation
            K[p, p+1]   = Gamma[5]                                                  # Gamma 5 assignation
            K[p, p-1+m] = Gamma[6]                                                  # Gamma 6 assignation
            K[p, p+m]   = Gamma[7]                                                  # Gamma 7 assignation
            K[p, p+1+m] = Gamma[8]                                                  # Gamma 8 assignation
    
    for j in np.arange(n):                                                          # For all the nodes in y.
        K[m*j, m*j] = 0                                                             # Zeros for the boundary nodes.
    
    for i in np.arange(1,m-1):                                                      # For all the nodes in x.
        p = i+(n-1)*m                                                               # Indexes for the boundary nodes.
        K[i, i] = 0                                                                 # Zeros for the boundary nodes.
        K[p, p] = 0                                                                 # Zeros for the boundary nodes.
    
    return K