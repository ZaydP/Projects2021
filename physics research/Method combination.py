import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import special
import time

# N = 2000
# ============ DEFINING THE TISE PARAMETERS: psi'' = C * (E - V(x))psi
# === where C = (2*m*c^2)/(hbar*c)^2 
m = 1
omega = 1
c = 2.99e23                     # fermi per second
hbar =1 #6.5821196e-16          #eV s
hbar_c = 197.326980             # MeV fermi
def V(x):
    return 0.5*m*(omega**2)*(x**2)

final_scaling = 1
C = 2#(2*m*c**2)/(hbar_c)**2

#===================================================================================================================

#====================================== BISECTION METHOD ==============================================================

def numerov(E, n, u): #find the wavefunction for a given energy
    h = np.abs(u[1] - u[0])
    f = C*(E-V(u))

    psi = np.empty(len(u))
    if (n%2 == 0):
        psi[0] = 1
        psi[1] = 1- 1e-10 #psi[0]*(12 - 10*(1 + f[0]*(h**2/12)))/(2*(1 + f[1]*(h**2/12)))
    elif (n%2==1):
        psi[0] = 0
        psi[1] = 1e-3
    else:
        print('number of nodes not positive integer')
    
    # implicit Numerov update
    for i in range(2,N):
        psi[i] = (psi[i-1]*(2-(5*h**2/6)*f[i-1]) - psi[i-2]*(1+ (h**2/12)*f[i-2]))/(1 + (h**2/12)*f[i])
    return psi


def HighOrLow(E, n, u): #we just use this to determine the sign of the last value of the wavefunction for a specific energy
    ulen = len(u)
    h = np.abs(u[1] - u[0])
    f = C*(E-V(u))

    psi = np.empty(ulen)
    if (n%2 == 0):
        psi[0] = 1
        psi[1] = 1- 1e-10 #psi[0]*(12 - 10*(1 + f[0]*(h**2/12)))/(2*(1 + f[1]*(h**2/12)))
    elif (n%2==1):
        psi[0] = 0
        psi[1] = 1e-3
    else:
        print('number of nodes not positive integer')
    
    # implicit Numerov update
    for i in range(2,ulen):
        psi[i] = (psi[i-1]*(2-(5*h**2/6)*f[i-1]) - psi[i-2]*(1+ (h**2/12)*f[i-2]))/(1 + (h**2/12)*f[i])
        if psi[i] > 1e12:
            return 1.0
        elif psi[i] < - 1e12:
            return -1.0

    return psi[-1] #if the function hasnt blown up, return the last value


def bisection(E_lw, E_up, n, u, mids):
    # If difference between bounds is lower than a threshold, return them
    # if np.abs(E_lw - E_up) < 1e-12: 
    #     return E_lw, E_up
    
    # determine whether the sign of the wavefunctions at are positive or negative at the end point.
    psi_lw = HighOrLow(E_lw, n, u)
    psi_up = HighOrLow(E_up, n, u)

    E_mid = (E_lw+E_up)/2
    psi_mid = HighOrLow(E_mid, n, u)
    print("bisection value", E_mid)

    mids.append(E_mid) 
    i = len(mids) 
    if len(mids) > 3 and np.abs(mids[i-1] - mids[i-2])<1.49e-8 :
        return E_lw, E_up

    if np.sign(psi_lw) == np.sign(psi_up):
        raise Exception("Initial energies do not bound an eigenstate, psi_lw and psi_up are: ",psi_lw, psi_up, E_lw, E_up)

    elif np.sign(psi_lw) == np.sign(psi_mid):
        return bisection(E_mid, E_up, n, u, mids)

    elif np.sign(psi_up) == np.sign(psi_mid):
        return bisection(E_lw, E_mid, n, u, mids)

def normalize(xarray, yarray): #Normalizes the half wavefunction (note the factor of 2 before the integral) over the half domain
    integral = integrate.trapz(np.square(yarray), xarray, dx = 1)
    normed_yarray = yarray * np.sqrt(1/(integral))
    normed_integral = integrate.trapz(np.square(normed_yarray), xarray, dx = 1)

    print("Integral before normalization: ",integral, ". Integral after normalization: ", normed_integral)
    return normed_yarray

def find_En(E_lw, E_up, n, x): # finds and prints our final, optimized energy eigenvalue
    mids = []
    Ebounds = bisection(E_lw, E_up, n, x, mids)
    Eavg = (Ebounds[1] + Ebounds[0])/2
    print("\n", "BISECTION METHOD: Final eigenvalue of eigenstate %i: " %n, Eavg )
    return Eavg


def waveFn_BisecM(E_lw, E_up, n, x): # Returns wavefunction by BISECTION METHOD
    # x = np.linspace(0, b, N)
    E_final = find_En(E_lw, E_up, n, x)
    wavefunction = numerov(E_final, n, x)
    full_x = np.concatenate((-np.flip(x), x))

    if (n%2 == 0):
        wvfn_even = normalize(full_x, np.concatenate((np.flip(wavefunction), wavefunction)) )
        return wvfn_even
    if (n%2 == 1):
        wvfn_odd = normalize(full_x, np.concatenate((-np.flip(wavefunction), wavefunction)) )
        return wvfn_odd

def plot_waveFn_BisecM(E_lw, E_up, n, x):
    wvfn = waveFn_BisecM(E_lw, E_up, n, x)
    full_x = np.concatenate((-np.flip(x), x))
    plt.plot(full_x, wvfn, label = 'Bisection Method')

#====================================================================================================================================

# ================================ METHOD OF MATCHING DERIVATIVES ==================================================================

def TP_index(E, u): # find index of classical turning point (first index such that V>E)
    return np.argmin(np.abs(E-V(u)))

def LR(E, u): # defines the domains left and right of the turning point found by TP_index
    b = u[-1]
    L = np.linspace(0, u[TP_index(E,u) + 1], N)
    R = np.linspace(u[TP_index(E,u) - 1], b, N)
    return L, R


def numerov_L(E, n, u): #find the wavefunction for a given energy left of TP
    # L = np.linspace(0, u[TP_index(E)+1], N)
    h = u[1] - u[0]
    L = u[:TP_index(E,u)+1]
    size_L = len(L)

    f = C*(E-V(L))
    #n=3
    psi = np.empty(size_L)
    if (n%2 == 0):
        psi[0] = 1
        psi[1] = psi[0]*(12 - 10*(1 + f[0]*(h**2/12)))/(2*(1 + f[1]*(h**2/12)))
    elif (n%2==1):
        psi[0] = 0
        psi[1] = 1e-1
    else:
        print('number of nodes not positive integer')
    
    # implicit Numerov update
    for i in range(2,size_L):
        psi[i] = (psi[i-1]*(2-(5*h**2/6)*f[i-1]) - psi[i-2]*(1+ (h**2/12)*f[i-2]))/(1 + (h**2/12)*f[i])
    return psi

def numerov_R(E, n, u): #find the wavefunction for a given energy right of TP
    # R = np.linspace(u[TP_index(E)-1], b, N)
    h = u[1] - u[0]
    R = u[TP_index(E,u)-1:]
    size_R = len(R)
    
    f = C*(E-V(R))
    #n=3
    psi = np.empty(size_R)
    if (n%2 == 0):
        psi[-1] = 1
        psi[-2] = psi[-1]*(12 - 10*(1 + f[-1]*(h**2/12))) / (2*(1 + f[-2]*(h**2/12)))
    elif (n%2==1):
        psi[-1] = 1e-5
        psi[-2] = 1e-2
    else:
        print('number of nodes not positive integer')
    
    # implicit Numerov update
    for i in range(size_R-3,-1, -1):
        psi[i] = (psi[i+1]*(2-(5*h**2/6)*f[i+1]) - psi[i+2]*(1+ (h**2/12)*f[i+2]))/(1 + (h**2/12)*f[i])
    return psi


def deriv_diff(E,n,u):
    h = u[1]-u[0]
    R = numerov_R(E,n,u)
    L = numerov_L(E,n,u)
    R = R * (L[-2]/R[1])
    return np.abs(((L[-1] - L[-3]) - (R[2] - R[0])) / (2*h*R[1]))

def SW_update(E,n,u):
    E = optimize.fsolve(lambda x: deriv_diff(x, n,u), E)
    print("\n", "METHOD OF MATCHING DERIVS: Final eigenvalue of eigenstate %i: " %n, E[0] )
    return E

def waveFn_MoMD(E, n, u):         # Returns wavefunction by METHOD of MATCHING DERIVATIVES
    E = SW_update(E, n, u)

    L = numerov_L(E, n, u)
    R = numerov_R(E, n, u)
    R = R * (L[-2]/R[1])
    wavefn = np.concatenate((L, R[2:]))
    # norm_wavefn = normalize(x, np.concatenate((np.flip(wavefn), wavefn))) # Normalize the entire wavefunction over the array x
    x = np.concatenate((-np.flip(u), u))
    if (n%2 == 0):
        return normalize(x, np.concatenate((np.flip(wavefn), wavefn)) )
    elif (n%2 == 1):
        return normalize(x, np.concatenate((-np.flip(wavefn), wavefn)) )
    else:
        print("Error in waveFn. n is not an integer")

def plot_wavefn_MoMD(E, n, u):
    x = np.concatenate((-np.flip(u), u))
    plt.plot(x, waveFn_MoMD(E, n, u), color = 'red', label = 'Numerical eigenstate %i' %n)

def analytic(xarray, n):
    G = m*omega/hbar 
    coeff = (1/np.sqrt(2**n * special.factorial(n))) * (G/np.pi)**(1/4)
    gauss = np.exp(-(G/2)*np.square(xarray))
    herm = special.hermite(n, monic= False)(np.sqrt(G)*xarray)
    analytic_wavefunction = coeff*gauss*herm
    if (n%2 == 0):
        if (n%4 == 0):
            return final_scaling*analytic_wavefunction
        elif (n%4 == 2):
            return -final_scaling*analytic_wavefunction
    elif (n%2 == 1):
        if ((n-1) % 4 == 0):
            return final_scaling*analytic_wavefunction
        elif((n-1) % 4 == 2):
            return -final_scaling*analytic_wavefunction

def plot_analytic(x, n):
    plt.plot(x, final_scaling*analytic(x, n), label = 'analytic solution')
    print("Integral of analytic solution: ", integrate.simps(np.square(analytic(x, n)), x, dx = x[1] - x[0]))


#=============================================================================================================================

#========================= ANALYSIS ======================================

N=2000

b=6
a=-b
halfdomain = np.linspace(0, b, N)
domain = np.linspace(a, b, 2*N)

# plot_wavefn_MoMD(2.46, 2, halfdomain)
# plot_waveFn_BisecM(2.4, 2.6, 2, halfdomain)
# plot_analytic(domain, 2)

def BISECTION_ANALYSIS(E_lw, E_up, n, N):
    u = np.linspace(0 ,b ,N)
    x = np.linspace(a, b, 2*N)

    tic = time.perf_counter()
    wvfn = waveFn_BisecM(E_lw, E_up, n, u)
    toc = time.perf_counter()
    analytic_wvfn = analytic(x, n)

    diff = wvfn - analytic_wvfn
    sum_diff = np.sum(np.abs(diff))/N
    run_time = toc-tic
    return wvfn, diff, sum_diff, run_time

def MoMD_ANALYSIS(E0, n, N):
    u = np.linspace(0, b, N)
    x = np.linspace(a, b, 2*N)

    tic = time.perf_counter()
    wvfn = waveFn_MoMD(E0, n, u)
    toc = time.perf_counter()
    analytic_wvfn = analytic(x, n)

    diff = wvfn - analytic_wvfn
    sum_diff = np.sum(np.abs(diff))/N
    run_time = toc-tic
    return wvfn, diff, sum_diff, run_time

bisec = BISECTION_ANALYSIS(3.45, 3.55, 3, 2000)
MoMD = MoMD_ANALYSIS(3.4, 3, 2000)
print("\n", "BISECTION METHOD: Total residuals divided by the number of points: ", bisec[2])
print("METHOD OF MATCHING DERIVS: Total residuals divided by the number of points: ", MoMD[2], "\n")
print("BISECTION METHOD: Total running time: ", bisec[3])
print("METHOD OF MATCHING DERIVS: Total running time: ", MoMD[3])

plt.plot(domain, bisec[0], label = 'Bisection method')
plt.plot(domain, MoMD[0], label = 'Method of Matching Derivs')
plt.show()
plt.plot(domain, bisec[1], label = 'Bisection residuals')
plt.plot(domain, MoMD[1], label = 'MoMD residuals')
plt.grid()
plt.minorticks_on()
plt.legend(fontsize = 'small')
plt.show()