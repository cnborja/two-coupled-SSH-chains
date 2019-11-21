"""
Plot the energy spectra for the Hamiltonian of two coupled SSH chains system defined in the article:

Li, C., Lin, S., Zhang, G., & Song, Z. (2017). Topological nodal points in two coupled Su-Schrieffer-Heeger chains. Physical Review B, 96(12), 125418.

as a function of the hoppings v/t on a 2xN ladder with open boundary condition. The script is able to reproduce FIG.3. from Li, et al. (2017) 

Author: Carla Borja Espinosa
Date: October 2019

Example command line in terminal to run the program: 
$ python3 ZeroPoints.py 10 1 figure.png

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from sympy.printing.str import StrPrinter
from sys import argv
from sympy.abc import v, w

class Hamiltonian(object):
    """Diagonalization of the system hamiltonian"""
    def __init__(self):
        """Variables""" 
        self.N = 2*int(argv[1])   # Number of ladders
        self.t = float(argv[2])   # Value of the interchain hopping
        self.filePNG = argv[3]    # Name of the file to store the plot
        """Functions"""
        self.HamiltonianMatrix()
        self.EigenvalsPlot()

    def HamiltonianMatrix(self):
        """Construct the matrix representation of the system 
           hamiltonian up to N ladders"""
        self.Inter = sp.Matrix([[0,self.t],[self.t,0]])
        self.Intra1 = sp.Matrix([[0,v],[w,0]])
        self.Intra2 = sp.Matrix([[0,w],[v,0]])
        H = sp.Matrix([])
        for i in range(1, self.N+1):
            fila = sp.Matrix([])
            for j in range(1, self.N+1):
                if j==i:
                    fila = fila.row_join(self.Inter)
                elif j==i+1:
                    fila = fila.row_join(self.Intra1)
                elif j==i-1:
                    fila = fila.row_join(self.Intra2)
                else:
                    fila = fila.row_join(sp.Matrix([[0,0],[0,0]]))
            H = H.col_join(fila) 
        H.simplify()
        #printer = StrPrinter()
        #print(H.table(printer,align='center'))
        self.H = H

    def EigenvalsPlot(self):
        """Plot the eigenvalues as a function of v/t for different 
           relations of the hopping parameters 
           (uncomment the desired case)"""
        Hfv = sp.lambdify(('v','w'), self.H)
        v_vals = np.arange(-1.5, 1.5+0.03 ,0.03)
        w_vals = self.t + v_vals     # case a 
        #w_vals = v_vals             # case b 
        #w_vals = 2*self.t - v_vals  # case c 
        #w_vals = -v_vals            # case d 
        size_v = np.size(v_vals)
        size_aut = self.H.shape[1]
        graph = np.empty([size_v,size_aut]) 
        for i in range(0,size_v):
            energies,vectors = la.eig(Hfv(v_vals[i],w_vals[i]))
            energies = np.sort(np.real(energies))
            graph[i] = energies
        graph = np.insert(graph, [0], v_vals.reshape(-1,1), axis = 1) 
        graph = np.matrix(graph)
        #with open(self.filetxt, "w+") as f:
        #    for line in graph:
        #        np.savetxt(f,line)
        for i in range(0, size_aut):
            plt.plot(v_vals,graph[:,i+1],'g')
        plt.xlim(-1.5,0.5)
        plt.ylim(-3,3)
        plt.savefig(self.filePNG)
        #plt.savefig(self.filePNG, format='eps', dpi=1200)
        #plt.show()

Hamiltonian()
