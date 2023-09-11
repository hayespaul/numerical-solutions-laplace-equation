import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.stats as sps


"""


CODE INSTRUCTIONS:

    -Just run the code and the main_menu() will allow testing of individual sections
    -Some functions may take a few minutes to run.


"""

def Gauss_seidel(tol,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y):
    """
    Gauss-seidel solver for the Laplace equation
    """

    v = np.zeros((resolution_x,resolution_y))   # potential array of nodes

    v,v_boundary = boundary_conditions(v,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)  # assign boundary conditions

    tic = time.time()
    max_v_diff = 1
    n=0

    while (max_v_diff > tol):
        v_old = v.copy()   # copy for convergence check
        for i in range(resolution_x):
            for j in range(resolution_y):

                if v[i,j]==abs(v_boundary):  # avoids computing boundary nodes
                    pass

                #corner nodes
                elif i == 0 and j==0:
                    v[i,j] = (1/2)*(v[i+1,j] + v[i,j+1])
                elif i ==0 and j == (resolution_y-1):
                    v[i,j] = (1/2)*(v[i+1,j] + v[i,j-1])
                elif i == (resolution_x-1) and j == 0:
                    v[i,j] = (1/2)*(v[i-1,j] + v[i,j+1])
                elif i == (resolution_x-1) and j == (resolution_y-1):
                    v[i,j] = (1/2)*(v[i-1,j] + v[i,j-1])

                #edge nodes
                elif j == 0:
                    v[i,j] = (1/3)*(v[i-1,j] + v[i+1,j] + v[i,j+1])
                elif j == (resolution_y-1):
                    v[i,j] = (1/3)*(v[i-1,j] + v[i+1,j] + v[i,j-1])
                elif i ==  0:
                    v[i,j] = (1/3)*(v[i+1,j] + v[i,j-1] + v[i,j+1])
                elif i == (resolution_x-1):
                    v[i,j] = (1/3)*(v[i-1,j] + v[i,j-1] + v[i,j+1])

                #intermediate nodes
                else:
                    v,_=boundary_conditions(v,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)

                    v[i,j] = (1/4)*(v[i-1,j] + v[i+1,j] + v[i,j-1] + v[i,j+1])

        v,_=boundary_conditions(v,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)

        v_diff = v - v_old # difference array
        max_v_diff = np.absolute(v_diff).max()  # convergence condition, when max value in difference array is less than tolerance, stop
        n+=1
    toc = time.time() - tic
    return v,n,toc


def jacobi(tol,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y):
    """
    Jacobi solver for the Laplace equation
    """
    v = np.zeros((resolution_x,resolution_y))
    v2= np.zeros((resolution_x,resolution_y))  # jacobi second array

    v,v_boundary = boundary_conditions(v,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)


    tic = time.time()
    max_v_diff = 1
    n=0

    while (max_v_diff > tol):
        for i in range(resolution_x):
            for j in range(resolution_y):
                if v[i,j]==abs(v_boundary):
                    pass

                elif i == 0 and j==0:
                    v2[i,j] = (1/2)*(v[i+1,j] + v[i,j+1])
                elif i ==0 and j == (resolution_y-1):
                    v2[i,j] = (1/2)*(v[i+1,j] + v[i,j-1])
                elif i == (resolution_x-1) and j == 0:
                    v2[i,j] = (1/2)*(v[i-1,j] + v[i,j+1])
                elif i == (resolution_x-1) and j == (resolution_y-1):
                    v2[i,j] = (1/2)*(v[i-1,j] + v[i,j-1])

                elif j == 0:
                    v2[i,j] = (1/3)*(v[i-1,j] + v[i+1,j] + v[i,j+1])
                elif j == (resolution_y-1):
                    v2[i,j] = (1/3)*(v[i-1,j] + v[i+1,j] + v[i,j-1])
                elif i ==  0:
                    v2[i,j] = (1/3)*(v[i+1,j] + v[i,j-1] + v[i,j+1])
                elif i == (resolution_x-1):
                    v2[i,j] = (1/3)*(v[i-1,j] + v[i,j-1] + v[i,j+1])

                else:
                    v2[i,j] = (1/4)*(v[i-1,j] + v[i+1,j] + v[i,j-1] + v[i,j+1])

        boundary_conditions(v2,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)
        v_diff = v2 - v
        max_v_diff = np.absolute(v_diff).max()
        v=v2.copy()
        n+=1
    toc = time.time() - tic
    return v2,n,toc



def boundary_conditions(v,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y):
    """
    Boundary condition function for test cases and parallel plates,
    arguments are different options to test.
    """
    if edge==1: # zero at edges
        v[0,:]=0
        v[-1,:]=0
        v[:,0]=0
        v[:,-1]=0

    if pp == 1: # parallel plate, finite
        v_boundary=250
        v[resolution_x/2-3,8:-8] = v_boundary
        v[resolution_x/2+3,8:-8] = -v_boundary

    if infpp ==1: #parallel plate, infinite
        v_boundary=250
        v[resolution_x/2-3,:] = v_boundary
        v[resolution_x/2+3,:] = -v_boundary

    if inf == 1: # further conditions for infinite plates
        v[resolution_x/2+4,:] = 0
        v[resolution_x/2-4,:] = 0

    if single_plate==1: # single plate capacitor
        v_boundary=250
        v[resolution_x/2,3:-3] = v_boundary

    if square_plate==1: # square capacitor
        v_boundary=250
        v[resolution_x/2-10:resolution_x/2+10,20:-20] = v_boundary

    if point_charge==1: # 'point charge'
        v_boundary=250
        v[resolution_x/2,resolution_x/2]=v_boundary

    return v,v_boundary


#####################################



def plot_infpp():
    """
    Plots the infinite plate capacitor problem
    """

    resolution_x=50
    resolution_y=50

    grid_width=resolution_x
    grid_height=resolution_y

    width_mesh=np.linspace(0,grid_width,resolution_x)
    height_mesh=np.linspace(0,grid_height,resolution_y)

    xx,yy = np.meshgrid(width_mesh,height_mesh)

    vg,_,_ = Gauss_seidel(1e-2,0,1,1,0,0,0,0,resolution_x,resolution_y) # infinte plate
    E=np.gradient(vg)    # electric field
    Emag=np.sqrt(E[0]**2 + E[1]**2) # magnitude


    plt.imshow(vg,cmap=cm.coolwarm)
    plt.colorbar(label='V')
    #plt.streamplot(xx, yy, E[1], E[0],density=[0.5, 1])
    plt.title('Potential Difference (V) as a function of x and y.')
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()

    ax = Axes3D(plt.figure())
    ax.plot_surface(xx,yy,vg, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    ax.set_zlabel('V')
    ax.view_init(elev=20, azim=-20)
    plt.title('3D plot of the Potential Difference (V) as a function of x and y.')
    plt.show()

    vg2,_,_ = Gauss_seidel(1e-2,0,0,1,0,0,0,0,resolution_x,resolution_y)
    E=np.gradient(vg2)
    Emag=np.sqrt(E[0]**2 + E[1]**2)

    plt.imshow(Emag,cmap='hot')
    plt.colorbar()
    plt.title('Electric Field (V/m) as a function of x and y.')
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()

    ax = Axes3D(plt.figure())
    ax.plot_surface(xx,yy,Emag, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    ax.set_zlabel('E')
    ax.view_init(elev=20, azim=-20)
    plt.title('3D plot of the Electric field (V/m) as a function of x and y.')
    plt.show()

    print('The max Electric field (V/d) ==',np.max(Emag),'V/m')

    return

def plot_pp():
    """
    Plots the parallel plate of finite extent
    """

    resolution_x=20
    resolution_y=20

    grid_width=resolution_x
    grid_height=resolution_y

    width_mesh=np.linspace(0,grid_width,resolution_x)
    height_mesh=np.linspace(0,grid_height,resolution_y)

    xx,yy = np.meshgrid(width_mesh,height_mesh)

    vg,_,_ = Gauss_seidel(1e-2,1,0,0,1,0,0,0,resolution_x,resolution_y) # finite parallel plate
    E=np.gradient(vg)
    Emag=np.sqrt(E[0]**2 + E[1]**2)

    plt.imshow(vg,cmap=cm.coolwarm)
    plt.colorbar(label='V')
    plt.title('Potential Difference (V) as a function of x and y.')
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()

    ax = Axes3D(plt.figure())
    ax.plot_surface(xx,yy,vg, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    ax.set_zlabel('V')
    ax.view_init(elev=20, azim=-20)
    plt.title('3D plot of the Potential Difference (V) as a function of x and y.')
    plt.show()

    E=np.gradient(vg)
    Emag=np.sqrt(E[0]**2 + E[1]**2)

    plt.imshow(Emag,cmap='hot')
    #plt.quiver(xx, yy, E[1], E[0])
    plt.colorbar()
    plt.title('Electric Field (V/m) as a function of x and y.')
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()

    ax = Axes3D(plt.figure())
    ax.plot_surface(xx,yy,Emag, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    ax.set_zlabel('E')
    ax.view_init(elev=20, azim=-20)
    plt.title('3D plot of the Electric field (V/m) as a function of x and y.')
    plt.show()

    return

#############################################################################

def compare_j_g(resolution_x,resolution_y,n):
    """
    Used to compare jacobi and gauss seidel
    """
    nj=np.zeros(n) # number of iterations of jacobi
    ng=np.zeros(n) # Gauss-Seidel

    tj=np.zeros(n) # time for jacobi
    tg=np.zeros(n) # gauss-seidel

    for i in range(n): # repeats

        _,nj[i],tj[i]=jacobi(1e-2,1,0,0,0,0,0,1,resolution_x,resolution_y)
        _,ng[i],tg[i]=Gauss_seidel(1e-2,1,0,0,0,0,0,1,resolution_x,resolution_y)

    return nj,ng,tj,tg

def plot_compare_j_g(n):
    """
    Plots number of iterations against grid density for jacobi and gauss seidel.
    """

    resolution=[10,20,30,40,50,60,70,80] # range of grid densities

    nj=np.zeros(len(resolution))
    ng=np.zeros(len(resolution))

    for i in range(len(resolution)):

        nj[i],ng[i],_,_ = compare_j_g(resolution[i],resolution[i],n)

    plt.plot(resolution, nj, color='r', linestyle='-', marker='.',label=("Jacobi Method"))
    plt.plot(resolution, ng, color='b', linestyle='-', marker='.',label=("Gauss-Seidel Method"))
    plt.title("Number of iterations for solution to converge as a function of grid density.")
    plt.xlabel("Grid Density")
    plt.ylabel("Number of iterations to reach solution (N)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()

    return

def plot_compare_j_g_relation(n):
    """
    Plots log base 2 of iterations against grid density.
    """

    resolution=[10,20,30,40,50,60,70,80] # range of grid densities

    nj=np.zeros(len(resolution))
    ng=np.zeros(len(resolution))

    for i in range(len(resolution)):

        nj[i],ng[i],_,_ = compare_j_g(resolution[i],resolution[i],n)

    log_j=np.log2(nj)
    log_g=np.log2(ng)

    plt.plot(resolution, log_j, color='r', linestyle='-', marker='.',label=("Jacobi Method"))
    plt.plot(resolution, log_g, color='b', linestyle='-', marker='.',label=("Gauss-Seidel Method"))
    plt.title("Number of iterations for solution to converge as a function of grid density.")
    plt.xlabel("Grid Density")
    plt.ylabel("Number of iterations to reach solution (N)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()

    return

def test_times(n):
    """
    Plots comparision of jacobi and gauss seidel for time to reach convergence against grid density
    """

    resolution=[10,20,30,40,50,60]

    times_j=np.zeros(len(resolution))
    times_g=np.zeros(len(resolution)) #time arrays

    error_tj=np.zeros(len(resolution)) # error arrays
    error_tg=np.zeros(len(resolution))

    for i in range(len(resolution)):
        _,_,tj,tg=compare_j_g(resolution[i],resolution[i],n)
        times_j[i]=np.sum(tj)/n
        times_g[i]=np.sum(tg)/n # average of gauss seidel times for n repeats
        error_tj[i]=np.max(tj)-np.min(tj) # error bars
        error_tg[i]=np.max(tg)-np.min(tg)

    plt.errorbar(resolution, times_j, error_tj, color='r', linestyle='-', marker='.',capsize=4,label=("Jacobi Method"))
    plt.errorbar(resolution, times_g, error_tg, color='b', linestyle='-', marker='.',capsize=4,label=("Gauss-Seidel Method"))
    plt.title("Time to reach convergence as a function of grid density.")
    plt.xlabel("Number of iterations to reach convergence (N) ")
    plt.ylabel("Time (s)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()
    return


def test_times_relation(n):
    """
    Plots log base 4 of time to reach convergence for jacobi and gauss seidel

    """

    resolution=[40,50,60,70,80]

    times_j=np.zeros(len(resolution))
    times_g=np.zeros(len(resolution)) #time arrays


    for i in range(len(resolution)):
        _,_,tj,tg=compare_j_g(resolution[i],resolution[i],n)
        times_j[i]=np.sum(tj)/n
        times_g[i]=np.sum(tg)/n # average of gauss seidel times for n repeats

    log1=np.log2(times_j)
    log2=np.log2(log1)

    log1_g=np.log2(times_g)
    log2_g=np.log2(log1_g)

    expected=resolution**4
    expected_log=np.log2(np.log2(expected))

    print(sps.chisquare(expected_log,log2))
    plt.plot(resolution,log2, color='r', linestyle='-', marker='.',label=("Jacobi Method"))
    plt.plot(resolution,log2_g, color='b', linestyle='-', marker='.',label=("Gauss-Seidel Method"))
    plt.title("Time to reach convergence as a function of grid density, log4(N).")
    plt.xlabel("Number of iterations to reach convergence (N) ")
    plt.ylabel("Time (s)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()

    return

def test_tolerance(resolution_x,resolution_y):
    """
    Tests number of iterations for convergence against tolerance in convergence condition
    """
    lowerLim=1e-10 # boundaries for tolerance
    upperLim=9e-1

    diff=upperLim-lowerLim
    spacing=diff/100 # spacing between plots
    tolerance=np.zeros(100)

    for i in range(100):
        tolerance[i]=lowerLim + i*spacing # generates array of tolerances

    nj=np.zeros(len(tolerance))
    ng=np.zeros(len(tolerance))

    for i in range(len(tolerance)-1):

        _,nj[i],_=jacobi(tolerance[i],1,0,0,0,0,0,1,resolution_x,resolution_y)
        _,ng[i],_=Gauss_seidel(tolerance[i],1,0,0,0,0,0,1,resolution_x,resolution_y)

    plt.plot(tolerance[1:-1],nj[1:-1], label=("Jacobi method"))
    plt.plot(tolerance[1:-1],ng[1:-1], label=("Gauss Seidel method"))
    plt.title("Number of iterations for convergance as a function of the tolerance.")
    plt.xlabel("Tolerance")
    plt.ylabel("Number of iterations to reach solution (N)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()
    return


def test_tolerance_relation(resolution_x,resolution_y):
    """
    Plots inverse relationship between tolerance and number of iterations
    """
    lowerLim=1e-10 # boundaries for tolerance
    upperLim=9e-1

    diff=upperLim-lowerLim
    spacing=diff/100 # spacing between plots
    tolerance=np.zeros(100)

    for i in range(100):
        tolerance[i]=lowerLim + i*spacing # generates array of tolerances

    nj=np.zeros(len(tolerance))
    ng=np.zeros(len(tolerance))

    for i in range(len(tolerance)-1):

        _,nj[i],_=jacobi(tolerance[i],1,0,0,0,0,0,1,resolution_x,resolution_y)
        _,ng[i],_=Gauss_seidel(tolerance[i],1,0,0,0,0,0,1,resolution_x,resolution_y)
    inverse_tol_j=1/nj
    inverse_ng=1/ng

    plt.plot(tolerance[1:-1],inverse_tol_j[1:-1], label=("Jacobi method"))
    plt.plot(tolerance[1:-1],inverse_ng[1:-1], label=("Gauss Seidel method"))
    plt.title("Number of iterations for convergance as a function of the tolerance.")
    plt.xlabel("Tolerance")
    plt.ylabel("Number of iterations to reach solution (N)")
    plt.legend(loc='best') #puts legend to best location
    plt.show()
    return


############################ PART 2 ######################################################


def case1(max,standard):
    """
    Finds time until equilibrium reached or line graphs for the first case of
    no heat loss at one end of the rod
    """
    h=0.001 # step size
    delta_t = 1 # time step
    alpha = 1.6596e-5 # thermal diffusivity

    x = np.arange(0,0.5,h)    # length of poker array

    theta = np.zeros(len(x)) # assigns initial temperature values
    theta[:]=20
    theta[0]=1000

    #assign A matrix of coefficients

    A = np.zeros((len(x),len(x)))

    a = -alpha*delta_t/(h**2) # coefficients
    b = 1 - 2*a

    diag_1=np.arange(len(x)) # array used to assign diagonals
    diag_2=np.arange(len(x)-1)

    A[diag_1,diag_1] = b
    A[diag_2+1, diag_2] = a
    A[diag_2,diag_2+1] = a

    A[0,0]=1 # assign corner and certain values to right values
    A[-1,-1]=1
    A[0,1]=0
    A[-1,len(x)-2]=0

    if standard == 1: # line graphs

        times=[0,100,500,1000,2000,30000] # range of times

        for i in range(len(times)):
            theta_prime=theta
            for j in range(times[i]):
                LU=sc.lu_factor(A)    # LU solver for sim equation
                theta_prime=sc.lu_solve(LU,theta)
                theta_prime[-1]=theta_prime[-2]  # assigns previous array value to last array value - no heat loss
                theta=theta_prime

            plt.plot(x[:-2],theta_prime[:-2], label=('Time={}s'.format(times[i])))
            plt.legend(loc='best') #puts legend to best location
            plt.xlabel('Distance along rod (m)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature distribution along rod as a function of distance (m)')
        plt.show()

    if max == 1: # finds time for poker to reach equilibiium to with 1 degree

        diff=1
        tolerance = 1
        n=0

        while diff > tolerance: # keep iterating until solution
                LU=sc.lu_factor(A)
                theta_prime=sc.lu_solve(LU,theta)
                theta_prime[-1]=theta_prime[-2]
                theta=theta_prime

                diff = theta_prime[0]-theta_prime[-1]
                n+=1
        print('Seconds for whole rod to reach 1000K, within 1K=',n)

    return


def case1_image():
    """
    Plots a heat map for the first case
    """

    h=0.001
    delta_t = 1
    alpha = 1.6596e-5

    x = np.arange(0,0.5,h)

    theta = np.zeros([len(x)]) # makes theta an array of size 1 x len(x)
    theta[:]=20
    theta[0]=1000

    #assign A

    A = np.zeros((len(x),len(x)))

    a = -alpha*delta_t/(h**2)
    b = 1 - 2*a

    diag_1=np.arange(len(x))
    diag_2=np.arange(len(x)-1)

    A[diag_1,diag_1] = b
    A[diag_2+1, diag_2] = a
    A[diag_2,diag_2+1] = a

    A[0,0]=1
    A[-1,-1]=1
    A[0,1]=0
    A[-1,len(x)-2]=0


    theta_full=np.zeros((len(x),1000)) # image array
    theta_full[:,0]=theta # initial values zero on far left hand side

    for i in range(10,20000):
        LU=sc.lu_factor(A)
        theta_prime=sc.lu_solve(LU,theta)
        theta_prime[-1]=theta_prime[-2]
        theta=theta_prime
        if i % 20 == 0:
            interval=int(i/20)
            theta_full[:,interval]=theta # add values to full array every interval of 10

    plt.imshow(theta_full,cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('Time (seconds x 20)')
    plt.ylabel('Temperature (K)')
    plt.show()
    return


def case2(standard,max):
    """
    Finds time to reach equilibirum or plots line graph for second case of
    one end in an ice bath.
    """

    h=0.001
    delta_t = 1
    alpha = 1.6596e-5

    x = np.arange(0,0.5,h)

    theta = np.zeros(len(x))
    theta[:]=20
    theta[0]=1000
    theta[-1]=0 # end of rod temp = 0 this time

    print(theta)

    #assign A

    A = np.zeros((len(x),len(x)))

    a = -alpha*delta_t/(h**2)
    b = 1 - 2*a

    diag_1=np.arange(len(x))
    diag_2=np.arange(len(x)-1)

    A[diag_1,diag_1] = b
    A[diag_2+1, diag_2] = a
    A[diag_2,diag_2+1] = a

    A[0,0]=1
    A[-1,-1]=1
    A[0,1]=0
    A[-1,len(x)-2]=0

    if standard == 1: # line graphs

        times=[0,100,500,1000,2000,10000]

        for i in range(len(times)):
            theta_prime=theta
            for j in range(times[i]):
                LU=sc.lu_factor(A)
                theta_prime=sc.lu_solve(LU,theta)
                theta_prime=theta_prime
                theta=theta_prime
            plt.plot(x,theta_prime, label=('Time={}s'.format(times[i])))
            plt.legend(loc='best') #puts legend to best location
            plt.xlabel('Distance along rod (m)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature distribution along rod as a function of distance (m)')
        plt.show()
    if max == 1: # finds time to reach equilibirum to within 1.01 of gradient

        diff=2
        tolerance = 1.01
        n=0

        while diff > tolerance:
                LU=sc.lu_factor(A)
                theta_prime=sc.lu_solve(LU,theta)
                theta=theta_prime

                diff = (theta_prime[0]-theta_prime[1])/2
                n+=1
        print('Seconds for rod to reach equilibrium, within 1K=',n)

    return


def case2_image():
    """
    PLots heat map of second case.
    """

    h=0.001
    delta_t = 1
    alpha = 1.6596e-5

    x = np.arange(0,0.5,h)

    theta = np.zeros([len(x)])
    theta[:]=20
    theta[0]=1000
    theta[-1]=0

    #assign A

    A = np.zeros((len(x),len(x)))

    a = -alpha*delta_t/(h**2)
    b = 1 - 2*a

    diag_1=np.arange(len(x))
    diag_2=np.arange(len(x)-1)

    A[diag_1,diag_1] = b
    A[diag_2+1, diag_2] = a
    A[diag_2,diag_2+1] = a

    A[0,0]=1
    A[-1,-1]=1
    A[0,1]=0
    A[-1,len(x)-2]=0

    ###

    theta_full=np.zeros((len(x),1000))
    theta_full[:,0]=theta

    for i in range(10,10000):
        LU=sc.lu_factor(A)
        theta_prime=sc.lu_solve(LU,theta)
        theta=theta_prime
        if i % 10 == 0:
            interval=int(i/10)
            theta_full[:,interval]=theta

    plt.imshow(theta_full,cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('Time (seconds x 10)')
    plt.ylabel('Temperature (K)')
    plt.show()
    return

############################################################################

def plot_test_cases(gs,j,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y):
    """
    Plots test cases, used in menu system
    """
    grid_width=resolution_x
    grid_height=resolution_y

    width_mesh=np.linspace(0,grid_width,resolution_x)
    height_mesh=np.linspace(0,grid_height,resolution_y)

    xx,yy = np.meshgrid(width_mesh,height_mesh)

    if gs == 1:         # user selects gauss seidel
        vg,_,_= Gauss_seidel(1e-2,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)
        plt.imshow(vg,cmap=cm.coolwarm)
        plt.colorbar()
        plt.title('Potential Difference (V) as a function of x and y.')
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.show()

        ax = Axes3D(plt.figure())
        ax.plot_surface(xx,yy,vg, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
        ax.view_init(elev=20, azim=-20)
        plt.title('3D plot of the Potential Difference (V) as a function of x and y.r')
        plt.show()

    elif j == 1: #jacobi selected
        vj,_,_ = jacobi(1e-2,edge,inf,infpp,pp,single_plate,square_plate,point_charge,resolution_x,resolution_y)
        plt.imshow(vj,cmap=cm.coolwarm)
        plt.colorbar()
        plt.title('Potential Difference (V) as a function of x and y.')
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.show()

        ax = Axes3D(plt.figure())
        ax.plot_surface(xx,yy,vj, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
        ax.view_init(elev=20, azim=-20)
        plt.title('3D plot of the Potential Difference (V) as a function of x and y.')
        plt.show()

    return


def test_cases_menu(gs,j):
    """
    Menu system test cases
    """

    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------TEST CASES MENU--------------------------------------------------------------------------')
        print("[1]Test 1: Plots graphs of the potential of a single line plate capacitor.")
        print("[2]Test 2: Plots graphs of the potential of a square plate capacitor.")
        print("[3]Test 3: Plots graphs of a 'point charge'.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2, 3 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")
            print("Please wait.....")
            plot_test_cases(gs,j,1,0,0,0,1,0,0,50,50)

            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")
            print("Please wait.....")
            plot_test_cases(gs,j,1,0,0,0,0,1,0,60,60)

            print("")
            print("\n##########################")
        elif MyInput == '3':
            print("\n######## TEST 3 #########")
            print("Please wait.....")
            plot_test_cases(gs,j,1,0,0,0,0,0,1,50,50)

            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')

    return

def compare_jg_menu():
    """
    Menu system used for comparing gauss seidel and jacobi methods
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------COMPARE GAUSS-SEIDEL AND JACOBI MENU--------------------------------------------')
        print("[1]Test 1: Compares the number of iterations required to reach convergence as a function of grid density.")
        print("[2]Test 2: Compares the number of iterations required to reach convergence as a function of tolerance.")
        print("[3]Test 3: Compares the number of iterations required to reach convergence as a function of time.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2, 3 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")
            print("Please wait.....")
            plot_compare_j_g(1)

            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")
            print("Please wait.....")
            test_tolerance(20,20)

            print("")
            print("\n##########################")
        elif MyInput == '3':
            print("\n######## TEST 3 #########")
            print("Please wait.....")
            test_times(3)

            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')

    return


def plot_parallel_plate_menu():
    """
    menu system for plotting parallel plate case
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------PARALLEL PLATE MENU--------------------------------------------------------------')
        print("[1]Test 1: Investigates the case of a finite length parallel plate capacitor.")
        print("[2]Test 2: Investigates the case of an infinite length parallel plate capacitor.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")
            print("Please wait.....")
            plot_pp()

            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")
            print("Please wait.....")
            plot_infpp()

            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')

    return



def test_task1():
    """
    Menu system for task 1
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------TASK 1 MENU--------------------------------------------------------------------------')
        print("[1]Test 1: Tests the Jacobi or Gauss-Seidel method in solving Laplace's equation for a number of known cases.")
        print("[2]Test 2: Compares the Jacobi and Gauss-Seidel method for solving Laplace's equation.")
        print("[3]Test 3: Investigates the case of a parallel plate capictor using the Jacobi or Gauss-Seidel methods.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2, 3 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")

            MyInput= input('Would you like to test the Gauss-Seidel or Jacobi method? *case sensitive, use caps* (J/G).')
            if MyInput=='J':
                test_cases_menu(0,1)
            elif MyInput=='G':
                test_cases_menu(1,0)


            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")

            compare_jg_menu()

            print("")
            print("\n##########################")
        elif MyInput == '3':
            print("\n######## TEST 3 #########")

            plot_parallel_plate_menu()

            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')
    return

def test_task2():
    """
    Menu system for task 2.
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------TASK 2 MENU--------------------------------------------------------------------------')
        print("[1]Test 1: Investigates the case of no heat loss at one end of the poker.")
        print("[2]Test 2: Investigates the case of one end of the poker in an ice bath.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")

            MyInput= input('Would you like to display the temperature as a heat map or line graph?  *use caps* (H/L).')
            if MyInput=='H':
                print('Heat map selected, please wait, will take a few minutes.....')
                case1_image()
            elif MyInput=='L':
                print('Line graph selected, please wait, may take a few minutes.....')
                case1(0,1)

            print("\n##########################")
        elif MyInput == '2':
            MyInput= input('Would you like to display the temperature as a heat map or line graph?  *use caps* (H/L).')
            if MyInput=='H':
                print('Heat map selected, please wait.....')
                case2_image()
            elif MyInput=='L':
                print('Line graph selected please wait.....')
                case2(1,0)

            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')
    return


############################################################################################################################

def main_menu():
    """
    Menu system for overall code.
    """
    MyInput = '0'
    while MyInput != 'q':
        print("")
        print('------------------------------- MAIN MENU --------------------------------------------------------------------------')
        print("[1]Option 1: Performs task 1.")
        print("[2]Option 2: Performs task 2.")
        print("--------------------------------------------------------------------------------------------------------------------")
        print("")
        MyInput = input('Select 1, 2 or q to quit:')
        if MyInput == '1':
            print("\n######## OPTION 1 #########")
            test_task1()

        elif MyInput == '2':
            print("\n######## OPTION 2 #########")
            test_task2()

        else:
            print("")
    print('Goodbye')

    return

main_menu()
