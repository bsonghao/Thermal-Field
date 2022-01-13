import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import factorial, isclose
from scipy.linalg import eigh
from CC_residue import *
from scipy.integrate import solve_ivp
import itertools as it


class two_body_model():
    """ Define a object that implement thermal field coupled_cluster
        method and thermal NOE method
        for GS and thermal property calculations for two-electron Hamiltonian """
    def __init__(self, E_HF, H_core, Fock, V_eri, n_occ, HF_occupation_number, molecule):
        """
        E_HF: energy expectation value (Hartree Fock GS energy)
        H_core: core electron Hamiltonian
        Fock: fock matrix
        V_eri: 2-electron integral (chemist's notation)
        M: dimension of MO basis
        molecule: name of the molecule for testing
        (all electron integrals are represented in MO basis)
        """
        print("***<start of input parameters>***")
        self.E_HF = E_HF
        self.V = V_eri
        self.n_occ = n_occ
        self.molecule = molecule
        self.HF_occupation_number = HF_occupation_number

        # HF occupation number
        print("HF occupation number: {:}".format(self.HF_occupation_number))
        # core electron Hamiltonian
        self.H_core = H_core

        # construct Fock matrix (from 1e and 2e integral)
        self.F = Fock

        # number of MOs
        self.M = self.F.shape[0]

        # f and f_bar defines contraction in thermal NOE
        self.f = self.n_occ / self.M
        self.f_bar = 1 - self.f

        # factorize prefactors
        self.one_half = 1. / 2.
        self.one_fourth = 1. / 4.
        # initialize Hamiltonian
        # constant part of the full Hamiltonian (H_0)
        print("HF ground state energy:{:.5f}".format(self.E_HF))
        # one-electron Hamiltonian   (H_1)
        print('One-electron part of the Hamiltonian (H_1):\n{:}'.format(self.F.shape))
        # two-electron Hamiltnoain (H_2)
        print("Two-electron part of the Hamiltonian (H_2):\n{:}".format(self.V.shape))

        # Boltzmann constant(Hartree T-1)
        self.kb = 3.1668152e-06

        print("Boltzmann constant(Hartree T-1):{:.9f}".format(self.kb))
        print("number of electrons: {:}".format(self.n_occ))
        print("number of orbitals: {:}".format(self.M))
        print("***<end of input parameters>***")

    def _mapping_HF_density_matrix_to_t_1_amplitude(self):
        """compute t_1 amplitude from occupation number of HF calculation"""
        # take occupation number as diagonal element of the 1-RDM
        RDM_HF = np.diag(self.HF_occupation_number)
        t_1 = np.zeros_like(RDM_HF)
        t_1 += RDM_HF
        t_1 -= np.einsum('pq,p,q->pq', np.eye(self.M), self.sin_theta, self.sin_theta)
        t_1 /= np.einsum('q,p->pq', self.cos_theta, self.sin_theta)

        return t_1

    def check_CC_residue(self):
        """subsitute HF mapped t_1 amplitude to CC residue equation and check if the CC residue to be zero"""
        # initialize T amplitude
        t_1 = self._mapping_HF_density_matrix_to_t_1_amplitude()
        t_2 = np.zeros([self.M, self.M, self.M, self.M])
        T_dic = {"t_1": t_1, "t_2": t_2}
        # substitute HF mapped T amplitude into the CC residue equation
        R_1, R_2 = residue(T_dic['t_1'], T_dic['t_2'], self.F_tilde, self.V_tilde, ERI_flag=True)

        CC_energy = energy(T_dic['t_1'], T_dic['t_2'], self.F_tilde, self.V_tilde, ERI_flag=True)

        # check if correlation energy + constant energy returns HF energy
        print("HF energy:{:.5f}".format(self.E_HF))
        print("E_0 + E_corr:{:.5f}".format(CC_energy + self.E_0))
        print("correlation energy:{:.5f}".format(CC_energy))


        # check if the CC reside get from HF mapped T amplitude to be zeros
        print("R_1:{:}".format(abs(R_1).max()))
        print("R_2:{:}".format(abs(R_2).max()))

        assert np.allclose(R_1, np.zeros_like(R_1), atol=1e-6)
        assert np.isclose(CC_energy + self.E_0, self.E_HF)

        return

    def Cal_mu_FD(self, E, beta):
        """calculate chemical that correct the total number of electrons using Newton's method"""
        # set initial guess to be zero
        mu = 0
        i = 0
        n_p = 1. / (1. + np.exp(beta * E - mu))
        while not isclose(sum(n_p), self.n_occ):
            n_p = 1. / (1. + np.exp(beta * E - mu))
            n_mu = sum(n_p) - self.n_occ
            dn_mu = sum(n_p**2 * np.exp(beta * E - mu))
            mu -= n_mu / dn_mu
            i += 1
            print("NEWTON PROCEDURE STEP {:d}:".format(i))
            print("chemical potential mu:{:.3f}".format(mu))
            print("n(mu):{:.3f}".format(n_mu + self.n_occ))
            if i > 100:
                print("Warning: Newtonain procedure does not converge with in 100 iterations! return zero beta reference state")
                n_p = self.f * np.ones(self.M)
                mu = - np.log(1./self.f - 1)
        return n_p, mu

    def _construct_fock_marix_from_BV_transformed_Hamiltonian(self):
        """construct fock matrix from 1-electron, 2-electron integral in quasi-particle representation"""
        return

    def _construct_fock_matrix_from_physical_Hamiltonian(self, RDM_1):
        """construct fock matrix from 1-electron, 2-electron integral in physical space and then BV
        transform to quasi-particle space
        """
        M = self.M
        # initialize
        Fock_Matrix = np.zeros([M, M])
        # add core electron part
        Fock_Matrix += self.H_core
        # add effective two electron part
        for a, b, c, d in it.product(range(M), repeat=4):
            Fock_Matrix[a, b] += RDM_1[c, d] * (2 * self.V[a, b, c, d] - self.V[a, c, b, d])
        # Bogoliubov transform the Fock_matrix
        Fock_Matrix_tilde = {"ij": np.einsum('i,j,ij->ij', self.sin_theta, self.sin_theta, Fock_Matrix),
                             "ai": np.einsum('a,i,ai->ai', self.cos_theta, self.sin_theta, Fock_Matrix),
                             "ia": np.einsum('i,a,ia->ia', self.sin_theta, self.cos_theta, Fock_Matrix),
                             "ab": np.einsum('a,b,ab->ab', self.cos_theta, self.cos_theta, Fock_Matrix)}
        return Fock_Matrix_tilde, Fock_Matrix

    def thermal_field_transform(self, T, debug=False):
        """conduct Bogoliubov transform on physical Hamiltonian"""
        # calculation recprical termperature
        beta = 1. / (self.kb * T)
        print("beta:{:.3f}".format(beta))
        # print("Fock matrix:{:}".format(self.F))
        # check if Fock matrix is diagonal
        if debug:
            assert np.allclose(self.F, np.diag(np.diag(self.F)), atol=1e-5)

        # calculate Femi-Dirac occupation number with chemical potential that fixex total number of electron
        n_p, mu = self.Cal_mu_FD(np.diag(self.F), beta)

        # construct 1-RDM in canonical basis
        RDM_1 = np.diag(n_p)

        print("Fermi-Dirac Occupation number:\n{:}".format(n_p))
        print("initial chemical potential:{:.3f}".format(mu))

        # define matrix elements in Bogoliubov transformation
        sin_theta = np.sqrt(n_p)
        cos_theta = np.sqrt(1 - n_p)

        # store sin_theta and cos_theta as object as instance
        self.sin_theta = sin_theta
        self.cos_theta = cos_theta
        # Bogoliubov transform 1-electron Hamiltonian (Fock Marix)
        self.H_core_tilde = {"ij": np.einsum('i,j,ij->ij', sin_theta, sin_theta, self.H_core),
                             "ai": np.einsum('a,i,ai->ai', cos_theta, sin_theta, self.H_core),
                             "ia": np.einsum('i,a,ia->ia', sin_theta, cos_theta, self.H_core),
                             "ab": np.einsum('a,b,ab->ab', cos_theta, cos_theta, self.H_core)}
        print("1-electron Hamiltonian in quasi-particle representation:")
        print("H_core_tilde_ij:\n{:}".format(self.H_core_tilde['ij'].shape))
        print("H_core_tilde_ai:\n{:}".format(self.H_core_tilde['ai'].shape))
        print("H_core_tilde_ia:\n{:}".format(self.H_core_tilde['ia'].shape))
        print("H_core_tilde_ab:\n{:}".format(self.H_core_tilde['ab'].shape))

        self.n_tilde = {"ij": np.einsum('i,j,ij->ij', self.sin_theta, self.sin_theta, np.eye(self.M)),
                        "ai": np.einsum('a,i,ai->ai', self.cos_theta, self.sin_theta, np.eye(self.M)),
                        "ia": np.einsum('i,a,ia->ia', self.sin_theta, self.cos_theta, np.eye(self.M)),
                        "ab": np.einsum('a,b,ab->ab', self.cos_theta, self.cos_theta, np.eye(self.M))}

        # Bogoliubov transfrom ERI (16 terms)
        self.V_tilde = {"ijkl": np.einsum('i,j,k,l,ijkl->ijkl', sin_theta, sin_theta, sin_theta, sin_theta, self.V),
                        "abcd": np.einsum('a,b,c,d,abcd->abcd', cos_theta, cos_theta, cos_theta, cos_theta, self.V),
                        "ijab": np.einsum('i,j,a,b,ijab->ijab', sin_theta, sin_theta, cos_theta, cos_theta, self.V),
                        "aibc": np.einsum('a,i,b,c,aibc->aibc', cos_theta, sin_theta, cos_theta, cos_theta, self.V),
                        "ijka": np.einsum('i,j,k,a,ijka->ijka', sin_theta, sin_theta, sin_theta, cos_theta, self.V),
                        "aijb": np.einsum('a,i,j,b,aijb->aijb', cos_theta, sin_theta, sin_theta, cos_theta, self.V),
                        "abci": np.einsum('a,b,c,i,abci->abci', cos_theta, cos_theta, cos_theta, sin_theta, self.V),
                        "iajk": np.einsum('i,a,j,k,iajk->iajk', sin_theta, cos_theta, sin_theta, sin_theta, self.V),
                        "iabc": np.einsum('i,a,b,c,iabc->iabc', sin_theta, cos_theta, cos_theta, cos_theta, self.V),
                        "ijak": np.einsum('i,j,a,k,ijak->ijak', sin_theta, sin_theta, cos_theta, sin_theta, self.V),
                        "iabj": np.einsum('i,a,b,j,iabj->iabj', sin_theta, cos_theta, cos_theta, sin_theta, self.V),
                        "abij": np.einsum('a,b,i,j,abij->abij', cos_theta, cos_theta, sin_theta, sin_theta, self.V),
                        "abic": np.einsum('a,b,i,c,abic->abic', cos_theta, cos_theta, sin_theta, cos_theta, self.V),
                        "aibj": np.einsum('a,i,b,j,aibj->aibj', cos_theta, sin_theta, cos_theta, sin_theta, self.V),
                        "aijk": np.einsum('a,i,j,k,aijk->aijk', cos_theta, sin_theta, sin_theta, sin_theta, self.V),
                        "iajb": np.einsum('i,a,j,b,iajb->iajb', sin_theta, cos_theta, sin_theta, cos_theta, self.V)}

        print("2-electron Hamiltonian in quasi-particle representation")
        print("V_tilde_ijkl:\n{:}".format(self.V_tilde['ijkl'].shape))
        print("V_tilde_abcd:\n{:}".format(self.V_tilde['abcd'].shape))
        print("V_tilde_ijab:\n{:}".format(self.V_tilde['ijab'].shape))
        print("V_tilde_aibc:\n{:}".format(self.V_tilde['aibc'].shape))
        print("V_tilde_ijka:\n{:}".format(self.V_tilde['ijka'].shape))
        print("V_tilde_aijb:\n{:}".format(self.V_tilde['aijb'].shape))
        print("V_tilde_abci:\n{:}".format(self.V_tilde['abci'].shape))
        print("V_tilde_iajk:\n{:}".format(self.V_tilde['iajk'].shape))
        print("V_tilde_iabc:\n{:}".format(self.V_tilde['iabc'].shape))
        print("V_tilde_ijak:\n{:}".format(self.V_tilde['ijak'].shape))
        print("V_tilde_iabj:\n{:}".format(self.V_tilde['iabj'].shape))
        print("V_tilde_abij:\n{:}".format(self.V_tilde['abij'].shape))
        print("V_tilde_abic:\n{:}".format(self.V_tilde['abic'].shape))
        print("V_tilde_aibj:\n{:}".format(self.V_tilde['aibj'].shape))
        print("V_tilde_aijk:\n{:}".format(self.V_tilde['aijk'].shape))
        print("V_tilde_iajb:\n{:}".format(self.V_tilde['iajb'].shape))

        # construct Fock matrix from 1-electron and 2-electron integrals
        self.F_tilde, self.F_physical = self._construct_fock_matrix_from_physical_Hamiltonian(RDM_1)

        print("Fock_Matrix in quasi-particle representation:")
        print("F_tilde_ij:\n{:}".format(self.F_tilde['ij'].shape))
        print("F_tilde_ai:\n{:}".format(self.F_tilde['ai'].shape))
        print("F_tilde_ia:\n{:}".format(self.F_tilde['ia'].shape))
        print("F_tilde_ab:\n{:}".format(self.F_tilde['ab'].shape))
        # determine the constant shift
        self.E_0 = np.trace(np.einsum('ij,jk->jk', 0.5 * (self.H_core + self.F_physical), RDM_1)) * 2

        print("constant term:{:}".format(self.E_0))

        return
