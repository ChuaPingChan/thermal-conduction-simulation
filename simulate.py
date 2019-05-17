# ME3291 Homework Assignment: Temperature Distribution over Plate
# Name: Chua Ping Chan
# Matric no.: A0126623L

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calc_machine_epsilon(n):
    '''
    n(int): Number of significant figures.
    return(float): Allowable error in percentage.
    '''
    return 0.5 * 10**(2-n)

def relative_error(x_new, x_old):
    """
    x_new(int or float): Current approximation
    x_old(int or float): Previous approximation
    return(float): Relative error in percentage
    """
    if (x_new <= machine_epsilon):
        return 0
    return abs(x_new - x_old)/x_new * 100

def print_plate(end_idx, start_idx=-1, interval=1):
    """Print plate for debugging.
    start_idx[int]
    end_idx[int]
    interval[int]
    """
    if (start_idx != -1):
        for k in range(start_idx, end_idx, interval):
            print('k =', k)
            print(np.flip(plate[k],axis=0))
            print()
    else:
        for k in range(end_idx):
            print('k =', k)
            print(np.flip(plate[k],axis=0))
            print()
    return

def status(idx=-1, disp=False):
    print('count =', count)
    print('time = {:1.4f}s'.format(time))
    if disp: print(plate[idx])
    return

def check_error(x_new, x_old, e_allow = 1):
    """
    x_new(int or float): Current approximation
    x_old(int or float): Previous approximation
    e_allow(int or float): Allowable relative error in percentage
    return(bool): True if error is within allowable range
    """
    return relative_error(x_new, x_old) <= e_allow

def stabalised(plate, machine_epsilon, track=False, count_int=100):
    if len(plate) < 2:
        return False
    x_new = np.copy(plate[-1])
    x_old = np.copy(plate[-2])
    error_matrix = abs(x_new - x_old)/x_new * 100
    max_error = np.nanmax(error_matrix)
    ##############################
    if track and (count%count_int == 0):    # Track max_error
        status()
        print('max_error =', max_error)
        print()
    ##############################
    return max_error <= machine_epsilon

def solve_1a_explicit(plate, stop_time=np.inf):
    """
    dt(int/float): size of time increment
    n(int): number of times to increment
    """
    global time, count, dt
    # Overwrite dt
    dt_new = (dx**2 + dy**2) / 8    # Criteria for convergence
    if dt_new < dt:
        dt = dt_new
        dt_changed = True
    else:
        dt_changed = False

    plate = apply_boundary_conditions_1(plate)
    while not stabalised(plate, machine_epsilon, track=True) and time < stop_time:
        time += dt
        count += 1
        new = np.copy(plate[-1])
        for j in range(1,N):
            for i in range(1,M):
                # print(str(i),str(j))
                new[j,i] = ((plate[-1,j,i+1] - 2 * plate[-1,j,i] + plate[-1,j,i-1])/dx**2
                            + (plate[-1,j+1,i] - 2 * plate[-1,j,i] + plate[-1,j-1,i])/dy**2) * k * dt + plate[-1,j,i]
        plate = np.concatenate((plate, new[np.newaxis,:,:]))
    if dt_changed:
        print('NOTE!\ndt changed to {:1.7f}s'.format(dt))
    return plate

def apply_boundary_conditions_1(plate):
    # Apply boundary conditions of question 1
    plate[:, -1, :] = 1.0    # top
##    plate[:, :, -1] = 1.0    # right
##    plate[:, 0, :] = 1.0    # bottom
##    plate[:, :, 0] = 1.0    # left
    return plate

def apply_boundary_conditions_2(plate):
    # Apply boundary conditions of question 2
    plate[:, -1, :] = 1.0
    return plate

def update_plate_1(plate, lin_arr):
    """
    Updates plate according to question 1.
    Boundary conditions needs to be updated.
    """
    lin_arr = np.reshape(lin_arr,(N-1,M-1))
    lin_arr = np.pad(lin_arr, 1, 'constant', constant_values=0)
    plate = np.concatenate((plate, lin_arr[np.newaxis,:,:]))
    plate = apply_boundary_conditions_1(plate)
    return plate

def update_plate_2(plate, lin_arr):
    """
    Updates plate according to question 2.
    Boundary conditions needs to be updated.
    """
    lin_arr = np.reshape(lin_arr,(N-1,M))
    lin_arr = np.pad(lin_arr, 1, 'constant', constant_values=0)
    lin_arr = lin_arr[:,:-1]
    plate = np.concatenate((plate, lin_arr[np.newaxis,:,:]))
    plate = apply_boundary_conditions_2(plate)
    return plate

##def gauss_seidel(plate, A, b, x=None, num_iter=None):
##    """Solves the equation Ax=b using the Gauss-Seidel method."""
##    global count
##    # Create an initial guess if needed
##    if x is None:
##        x = np.zeros(len(A[0]))
##
##    L = np.tril(A)
##    U = A - L
##
##    while not stabalised(plate, machine_epsilon):
##        count += 1
##        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
##        # update plate (global)
##        sol = np.reshape(x,(N-1,M-1))
##        sol = np.pad(sol, 1, 'constant', constant_values = 0)
##        plate = np.concatenate((plate, sol[np.newaxis,:,:]))
##    plate = apply_boundary_conditions_1(plate)
##    return plate

def numpy_solver(A, b):
    """ Solve Ax = b """
##    print('Using NumPy\'s solver')
    return np.linalg.solve(A,b)

def solve_1a_implicit(plate, dt, stop_time=np.inf, method=None):
    """
    Solution using numpy's linear matrix solving implementation.

    Note:
    Requires modification of matrix A & b construction and func that applies
    boundary condition when when boundary condition changes.
    """
    global time, count
    plate = apply_boundary_conditions_1(plate)
    mlen = (M-1)*(N-1)

    # Construct matrix A of Ax=b.
    # Note that A is dependent on differencing scheme and equation used.
    A = ( (4*lam + 1) * np.eye(mlen)
         - lam * np.eye(mlen,k=1)
         - lam * np.eye(mlen,k=M-1)
         - lam * np.eye(mlen,k=-1)
         - lam * np.eye(mlen,k=1-M) )
    for i in range(M-1, mlen, M-1):
        A[i-1, i] = 0
        A[i, i-1] = 0

    while time < stop_time and not stabalised(plate, machine_epsilon, track=True):
        time += dt
        count += 1

        # Construct matrix b
        # Note that b is dependent on boundary conditions and equation used.
        b = np.zeros((M-1, N-1))
        b += plate[-1, 1:-1, 1:-1]  # Update from previous iteration
        # Apply boundary conditions
        b[-1,:] += lam * plate[0, -1, 1:-1]
        b = b.flatten()

        # Solve system of equations
        if method == 'jacobi':
            x = jacobi(A, b)    # jacobi method
        else:
            x = numpy_solver(A, b)   # np linear matrix eq solver
        # Update plate with solution
        plate = update_plate_1(plate, x)

    return plate

def jacobi(A, b, x=None, num_iter=None):     # modifying
    """Solves the equation Ax=b via the Jacobi iterative method."""
##    print('Using Jacobi method')
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diagflat(np.diag(A))
    D_inv = np.diagflat(1/np.diag(A))
    C = D - A

    # Iterate
    x_history = x[np.newaxis,:]
    if num_iter is not None:    # For debugging
        for i in range(num_iter):
                x = np.diag((b + np.dot(C,x)) * D_inv)
        return x
    else:
        while not stabalised(x_history, machine_epsilon):
            x = np.diag((b + np.dot(C,x)) * D_inv)
            x_history = np.concatenate( (x_history,x[np.newaxis,:]) )

        # For debugging
##        print('x_history =')
##        print(x_history)
##        print()

        return x

def solve_1b_implicit(plate, method=None):
    """Solves steady state implicitly method."""
    global time, count
    time = np.inf

    plate = apply_boundary_conditions_1(plate)
    mlen = (M-1)*(N-1)
    # Construct matrix A of Ax=b.
    # Note that A is dependent on differencing scheme and equation used.
    A = (4 * np.eye(mlen)
         - 1 * np.eye(mlen,k=1)
         - 1 * np.eye(mlen,k=M-1)
         - 1 * np.eye(mlen,k=-1)
         - 1 * np.eye(mlen,k=1-M))
    for i in range(M-1, mlen, M-1):
        A[i-1, i] = 0
        A[i, i-1] = 0

    # Construct matrix u
    # u = plate[-1, 1:-1, 1:-1].flatten()

    # Construct matrix b
    # Note that b is dependent on boundary conditions
    b = np.zeros((M-1, N-1))
    b[-1,:] = plate[0, -1, 1:-1]   # top
##    b[0,:] = plate[0, 0, 1:-1]    # bottom
##    b[:,0] = plate[0, 1:-1, 0]  # left
##    b[:,-1] = plate[0, 1:-1, -1]   # right
    b = np.reshape(b, mlen) # Alternative: Use ndarray.flatten()

    # Solve system of equations
    if method == 'jacobi':
        x = jacobi(A, b)    # jacobi method
    else:
        x = numpy_solver(A, b)   # np linear matrix eq solver

    # Update plate
    plate = update_plate_1(plate, x)

    return plate

def solve_2_implicit(plate, dt, stop_time=np.inf, method=None):
    """
    Solution using numpy's linear matrix solving implementation.

    Note:
    Requires modification of matrix A & b construction and func that applies
    boundary condition when when boundary condition changes.
    """
    global time, count
    plate = apply_boundary_conditions_2(plate)
    mlen = (M-1)*(N)
    shape = (N-1,M)

    # Construct matrix A of Ax=b.
    # Note that A is dependent on differencing scheme, boundaryconditions
    # and the governing equation used.
    A = ( (4*lam + 1) * np.eye(mlen)
         - lam * np.eye(mlen,k=1)
         - lam * np.eye(mlen,k=M)
         - lam * np.eye(mlen,k=-1)
         - lam * np.eye(mlen,k=-M) )
    for i in range(M, mlen, M):
        A[i-1, i] = 0
        A[i, i-1] = 0
    for i in range(M-1, mlen, M):
        if (i+M) < mlen:
            A[i,i-1] -= lam

    while time < stop_time and not stabalised(plate, machine_epsilon, track=True):
        time += dt
        count += 1

        # Construct matrix b
        # Note that b is dependent on boundary conditions and equation used.
        b = np.zeros(shape)
        b += plate[-1, 1:-1, 1:]    # Update from previous iteration
        # Apply boundary conditions
        b[-1,:] += lam * plate[0, -1, 1:]
        b = b.flatten()

        # Solve system of equations
        if method == 'jacobi':
            x = jacobi(A, b)    # jacobi method
        else:
            x = numpy_solver(A, b)   # np linear matrix eq solver
        plate = update_plate_2(plate, x)

    return plate

def update_contour(idx=-1, levels=100):
##    ax.clear()
    cont = ax.contourf(X, Y, plate[idx], levels, cmap=plt.cm.plasma)
    return cont

def label_axes(ax, show_title=True):
    """Labels the x- and y- axis and title"""
    ax.set_xlabel('x')
    ax.xaxis.set_label_coords(1.07, -0.05)
    ylabel = ax.set_ylabel('y')
    ylabel.set_rotation(0)
    ax.yaxis.set_label_coords(-0.12, 1.05)

    # Label boundary conditions
    if ques == 1:
        fig.text(0.4, 0.9, u'T = 1\u00B0C')   # top
        fig.text(0.05, 0.5, u'T = 0\u00B0C', rotation='vertical')   # left
        fig.text(0.74, 0.5, u'T = 0\u00B0C', rotation='vertical')   # right
        fig.text(0.4, 0.025, u'T = 0\u00B0C')   # bottom
    elif ques == 2:
        fig.text(0.4, 0.9, u'T = 1\u00B0C')   # top
        fig.text(0.05, 0.5, u'T = 0\u00B0C', rotation='vertical')   # left
        fig.text(0.74, 0.5, r'$\frac{\partial T}{\partial t} = 0$',
                 size=13, rotation='vertical')   # right
        fig.text(0.4, 0.025, u'T = 0\u00B0C')   # bottom

    # Set title
    if show_title:
        ax.set_title('Temperature Distribution (time = {:1.4f}s)'.format(time), position=(0.5,1.07))
##        ax.set_title('Temperature Distribution (Steady State)')

    return ax

def snap(idx=-1): # still developing, issue with update_contour()
    """Display a snapshot of contourplot at a given count."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)

    # Label axes
    ax = label_axes(ax)

    X, Y = np.meshgrid(x, y)
    cont = update_contour(idx)
    cbar = plt.colorbar(cont,ax=ax)
    cbar.ax.set_ylabel(u'Temperature (\u00B0C)')

    plt.show()
    plt.close()
    return

############################################################
# MAIN

# Initialisation
machine_epsilon = calc_machine_epsilon(4)
k = 1           # thermal diffusivity of plate's material
L = 1
M = 20          # No. of divisions per row
N = M           # No. of divisions per column
nrows = N + 1   # No. of nodes per row
ncols = M + 1   # No. of nodes per column
dx = L/M
dy = L/N
dt = 0.0005      # May be changed by explicit method
stop_time = np.inf
if dx == dy:
    dh = dx
    lam = k * dt/(dx**2)
plate = np.zeros((1, nrows, ncols))
time = 0.0
count = 0

# Question 1(a)
# plate = solve_1a_explicit(plate, stop_time=stop_time); ques=1
# plate = solve_1a_implicit(plate, dt, stop_time=stop_time, method=None); ques=1

# Question 1(b)
plate = solve_1b_implicit(plate, method=None); ques=1

# Question 2
# plate = solve_2_implicit(plate, dt, stop_time=stop_time, method=None); ques=2

############################################################
# Plotting & Visualisation

# Set plotting parameters
plot_enable = True
animate_enable = True
showplot_enable = True
num_frames = 10
max_frame_idx = count

# Plot
if plot_enable:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)

    # Label axes
    ax = label_axes(ax)

    X, Y = np.meshgrid(x, y)
    cont = update_contour(levels=200)
    ax.set_aspect('equal')

    # Animate with Func
    if animate_enable:
        frames = range(0,max_frame_idx,int(np.ceil(max_frame_idx/num_frames)))
        ani = animation.FuncAnimation(fig, update_contour, frames=frames,
                                          interval=5, repeat=False)

    cbar = plt.colorbar(cont, ax=ax, aspect=30, format='%.2f', fraction=0.1, pad=0.13)
    cbar.ax.set_ylabel(u'Temperature (\u00B0C)')

    if showplot_enable:
        plt.show()
############################################################
