# system imports

# third party imports
import numpy as np
import scipy as sp
from pyscf import gto, scf, ao2mo

# local imports


class check_integral():
    """ check integral class test the input on one-electron and two-electron of PySCF input by comparing
    the converged Fock matrix, density matrix and GS energy between a SCF procedure
    writing by myself the the HF result of PySCF"""

    def __init__(self, E_HF, H_core, Fock, V_eri, n_occ, S, occupation_number, NRE, molecule, MO=False):
        """
        E_HF: energy expectation value (Hartree Fock GS energy)
        H_core: core electron Hamiltonian
        Fock: Fock matrix
        V_eri: 2-electron integral (chemist's notation)
        M: dimension of MO basis
        MO: Boolean to determine if the integral is implemented in MO basis or not
        molecule: name of the molecule for testing
        (all electron integrals are represented in AO basis)
        """

        print("***<start of input parameters>***")
        self.E_HF = E_HF
        self.V = V_eri

        # convert 2e integral to physist's notation
        # self.V = np.rollaxis(self.V, 1, 3)
        self.n_occ = n_occ
        self.molecule = molecule

        # core electron Hamiltonian
        self.H_core = H_core

        # construct Fock matrix (from 1e and 2e integral)
        self.F = Fock

        # overlap matrix (in AO basis)
        self.S = S

        self.occ = occupation_number

        # number of MOs
        self.M = self.F.shape[0]

        # NuclearRepulsionEnergy
        self.NRE = NRE

        # MO / AO integral?
        self.MO = MO
        print("***<end of input parameters>***")

    def my_SCF(self, max_iteration=100):
        """write my own SCF procedure"""

        def Cal_Fock(D):
            """Calculate Fock matrix from 1-RDM"""

            def Cal_sym_2e_int(V, DM):
                """Calculate symmetrized 2 electron integral"""

                E_mf = np.zeros([self.M, self.M])

                for a in range(self.M):
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                E_mf[a, b] += (2 * self.V[a, b, c, d] - self.V[a, c, b, d]) * DM[c, d]

                return E_mf

            # initialize
            Fock_Matrix = np.zeros([self.M, self.M])

            # add the core contributions
            Fock_Matrix += self.H_core

            # we choose the effective potential to be
            # the symmetrized two electron integral
            sym_V = Cal_sym_2e_int(self.V, D)

            # add the potential contributions
            Fock_Matrix += sym_V

            return Fock_Matrix

        def Cal_Density_Matrix(Fock_Matrix, Overlap_Matrix):
            """Calculate density matrix from Fock matrix and overlap matrix"""

            if self.MO:
                e, val = np.linalg.eigh(Fock_Matrix)
            else:
                e, val = sp.linalg.eigh(Fock_Matrix, Overlap_Matrix)

            Density_Matrix = np.einsum('ui,i,vi->uv', val, self.occ, val)

            return Density_Matrix

        def Cal_HF_energy(Fock_Matrix, Density_Matrix):
            """Calculate Hartree Fock ground state energy"""
            E_HF = np.einsum('pq,qp->', self.H_core + Fock_Matrix, Density_Matrix)
            return E_HF

        # initially set Fock matrix to be core electron Hamiltonian
        Fock = self.H_core.copy()

        # attempt to calculate the energy with a Hartree Fock SCF procedure
        for i in range(max_iteration):

            # calculate density matrix before the update of Fock matrix
            RDM_1_in = Cal_Density_Matrix(Fock, self.S)

            # calculate Fock matrix from density matrix
            Fock = Cal_Fock(RDM_1_in)

            # calculate density matrix after the update of Fock matrix
            RDM_1_out = Cal_Density_Matrix(Fock, self.S)

            # evaluate HF energy after update of Fock matrix and Density matrix
            E_new = Cal_HF_energy(Fock, RDM_1_out)

            if np.allclose(RDM_1_in, RDM_1_out):
                # break the SCF procedure when density matrix converge
                break

            print("SCF procedure at {:d} iteration".format(i))
            print("HF energy:{:.5f}".format(E_new))

        # if we don't break out of the loop that means
        # that our density matrices did NOT converge
        else:
            string = (
                'Reached max iterations of {:d}\n'
                'The density matrices did not converge?!'
            )
            print(string.format(max_iteration))

        # add the final contribution to the energy
        # (which is the NuclearRepulsionEnergy)
        self.E_HF_my_answer = E_new + self.NRE

        print("Converged HF energy (my answer): {:.5f}".format(self.E_HF_my_answer))
        print("Converged HF energy (PySCF result): {:.5f}".format(self.E_HF))
        print("Difference to PySCF result (E_pyscf - E_myhf):{:.5f}".format(self.E_HF - self.E_HF_my_answer))
