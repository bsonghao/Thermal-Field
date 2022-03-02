import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import factorial, isclose
from scipy.linalg import eigh
from CC_residue import *
import itertools as it


class two_body_model():
    """ Define a object that implement thermal field coupled_cluster
        method and thermal NOE method
        for GS and thermal property calculations for two-electron Hamiltonian """
    def __init__(self, E_core, H_core, Fock, V_eri, n_occ, molecule, E_NN, T_2_flag=True, chemical_potential=True, partial_trace_condition=True):
        """
        E_core: core electron contribution of the Hamiltonian
        Fock: ground_state_Fock_matrix
        H_core: core electron Hamiltonian
        V_eri: 2-electron integral (chemist's notation)
        n_occ: total number of electrons
        M: dimension of MO basis
        molecule: name of the molecule for testing
        E_NN: nuclear repulsion energy
        T_2_flag: boolean to determine whether to update T_2 amplitude
        chemical_potential: boolean to determine whether to introduce chemical potential in the integration
        partial_trace_condition: boolean to determine whether to introcude constraint on partial trace
        (all electron integrals are represented in MO basis)
        """
        print("***<start of input parameters>***")
        self.E_core = E_core
        self.V = V_eri
        self.n_occ = n_occ
        self.molecule = molecule
        # core electron Hamiltonian
        self.H_core = H_core
        # construct Fock matrix (from 1e and 2e integral)
        self.F_ground_state = Fock

        self.T_2_flag = T_2_flag
        self.chemical_potential = chemical_potential
        self.partial_trace_condition = partial_trace_condition

        # nuclear repulsion energy
        self.E_NN = E_NN

        # number of MOs
        self.M = self.F_ground_state.shape[0]

        # f and f_bar defines contraction in thermal NOE
        self.f = self.n_occ / self.M
        self.f_bar = 1 - self.f

        # factorize prefactors
        self.one_half = 1. / 2.
        self.one_fourth = 1. / 4.
        # initialize Hamiltonian
        # constant part of the full Hamiltonian (H_0)
        print("core electron energy:{:.5f}".format(self.E_core))

        print("nuclear repulsion enegy:{:}".format(self.E_NN))
        # one-electron Hamiltonian   (H_1)
        print('One-electron integrals:\n{:}'.format(self.H_core.shape))
        # two-electron Hamiltnoain (H_2)
        print("Two-electron integrals:\n{:}".format(self.V.shape))

        # Boltzmann constant(Hartree T-1)
        self.kb = 3.1668152e-06

        print("Boltzmann constant(Hartree T-1):{:.9f}".format(self.kb))
        print("number of electrons: {:}".format(self.n_occ))
        print("number of orbitals: {:}".format(self.M))
        print("***<end of input parameters>***")

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

    def thermal_field_transform(self, T):
        """conduct Bogoliubov transform on physical Hamiltonian"""
        # calculation recprical termperature
        beta = 1. / (self.kb * T)
        print("beta:{:.3f}".format(beta))

        # calculate Femi-Dirac occupation number with chemical potential that fixex total number of electron
        FD_occupation_number, mu = self.Cal_mu_FD(np.diag(self.F_ground_state), beta)

        # construct 1-RDM in canonical basis
        RDM_1 = np.diag(FD_occupation_number)

        print("Fermi-Dirac Occupation number:\n{:}".format(FD_occupation_number))
        print("initial chemical potential:{:.3f}".format(mu))

        # Bogoliubov transform the Hamiltonian from physical space to quasi-particle representation
        sin_theta = np.sqrt(FD_occupation_number)
        cos_theta = np.sqrt(1 - FD_occupation_number)

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

        # Bogoliubov transform the one body density matrix at zero beta
        self.n_tilde = {"ij": np.einsum('i,j,ij->ij', self.sin_theta, self.sin_theta, np.eye(self.M)),
                        "ai": np.einsum('a,i,ai->ai', self.cos_theta, self.sin_theta, np.eye(self.M)),
                        "ia": np.einsum('i,a,ia->ia', self.sin_theta, self.cos_theta, np.eye(self.M)),
                        "ab": np.einsum('a,b,ab->ab', self.cos_theta, self.cos_theta, np.eye(self.M))}

        # 2-electron Hamiltonian (16 terms)
        self.V_tilde = {
              "ijkl": np.einsum('i,j,k,l,ijkl->ijkl', sin_theta, sin_theta, sin_theta, sin_theta, self.V),
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
              "iajb": np.einsum('i,a,j,b,iajb->iajb', sin_theta, cos_theta, sin_theta, cos_theta, self.V)
              }

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

        # Bogoliubov transform the two body density matrix at zero beta
        # 2-electron density matrix at zero beta (16 terms)
        two_body_DM = 0.5 * np.einsum('pr,qs->pqrs', np.eye(self.M), np.eye(self.M))

        self.O_tilde = {
              "ijkl": np.einsum('i,j,k,l,ijkl->ijkl', sin_theta, sin_theta, sin_theta, sin_theta, two_body_DM),
              "abcd": np.einsum('a,b,c,d,abcd->abcd', cos_theta, cos_theta, cos_theta, cos_theta, two_body_DM),
              "ijab": np.einsum('i,j,a,b,ijab->ijab', sin_theta, sin_theta, cos_theta, cos_theta, two_body_DM),
              "aibc": np.einsum('a,i,b,c,aibc->aibc', cos_theta, sin_theta, cos_theta, cos_theta, two_body_DM),
              "ijka": np.einsum('i,j,k,a,ijka->ijka', sin_theta, sin_theta, sin_theta, cos_theta, two_body_DM),
              "aijb": np.einsum('a,i,j,b,aijb->aijb', cos_theta, sin_theta, sin_theta, cos_theta, two_body_DM),
              "abci": np.einsum('a,b,c,i,abci->abci', cos_theta, cos_theta, cos_theta, sin_theta, two_body_DM),
              "iajk": np.einsum('i,a,j,k,iajk->iajk', sin_theta, cos_theta, sin_theta, sin_theta, two_body_DM),
              "iabc": np.einsum('i,a,b,c,iabc->iabc', sin_theta, cos_theta, cos_theta, cos_theta, two_body_DM),
              "ijak": np.einsum('i,j,a,k,ijak->ijak', sin_theta, sin_theta, cos_theta, sin_theta, two_body_DM),
              "iabj": np.einsum('i,a,b,j,iabj->iabj', sin_theta, cos_theta, cos_theta, sin_theta, two_body_DM),
              "abij": np.einsum('a,b,i,j,abij->abij', cos_theta, cos_theta, sin_theta, sin_theta, two_body_DM),
              "abic": np.einsum('a,b,i,c,abic->abic', cos_theta, cos_theta, sin_theta, cos_theta, two_body_DM),
              "aibj": np.einsum('a,i,b,j,aibj->aibj', cos_theta, sin_theta, cos_theta, sin_theta, two_body_DM),
              "aijk": np.einsum('a,i,j,k,aijk->aijk', cos_theta, sin_theta, sin_theta, sin_theta, two_body_DM),
              "iajb": np.einsum('i,a,j,b,iajb->iajb', sin_theta, cos_theta, sin_theta, cos_theta, two_body_DM)
              }

        # construct Fock matrix from 1-electron and 2-electron integrals
        self.F_tilde, self.F_physical = self._construct_fock_matrix_from_physical_Hamiltonian(RDM_1)

        # determine the constant shift
        self.E_0 = np.trace(np.dot((self.H_core + self.F_physical), RDM_1))

        print("constant term:{:}".format(self.E_0))

        return

    def _symmetrize_density_matrix(self, RDM_1, RDM_2):
        """symmetrize 1 and 2-RDM and mapping them to T amplitude"""
        def symmetrize_RDM_1(RDM_1):
            """symmetrize 1-RDM"""
            RDM_1_sym = np.zeros_like(RDM_1)
            RDM_1_sym += RDM_1
            RDM_1_sym += RDM_1.transpose()
            RDM_1_sym *= 0.5
            return RDM_1_sym

        def symmetrize_RDM_2(RDM_2):
            """symmetrize 2-RDM"""
            RDM_2_sym = np.zeros_like(RDM_2)
            RDM_2_sym += RDM_2
            RDM_2_sym += RDM_2.transpose(2, 3, 0, 1)
            RDM_2_sym *= 0.5
            return RDM_2_sym

        def map_t_1_from_RDM(RDM_1):
            """map T_1 amplitude from symmetrized RDM_1"""
            T_1 = np.zeros_like(RDM_1)
            T_1 += RDM_1
            T_1 -= np.einsum('p,q,pq->pq', self.sin_theta, self.sin_theta, np.eye(self.M))
            T_1 /= np.einsum('q,p->pq', self.cos_theta, self.sin_theta)
            return T_1

        def map_t_2_from_RDM(RDM_1, RDM_2):
            """map T_2 amplitude from symmetrized RDM_1 and RDM_1"""
            T_2 = np.zeros_like(RDM_2)
            T_2 += RDM_2
            T_2 -= np.einsum('pr,qs->pqrs', RDM_1, RDM_1)
            T_2 += np.einsum('ps,qr->pqrs', RDM_1, RDM_1)
            T_2 /= np.einsum('r,s,p,q->pqrs', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta)
            return T_2

        # symmetrize density matrix
        RDM_1_sym = symmetrize_RDM_1(RDM_1)
        RDM_2_sym = symmetrize_RDM_2(RDM_2)

        # map T amplitude from symmetrize 1-RDM and 2-RDM
        T_1_sym = map_t_1_from_RDM(RDM_1_sym)
        T_2_sym = map_t_2_from_RDM(RDM_1_sym, RDM_2_sym)

        # check if the quantities are symmetrized as expected
        assert np.allclose(RDM_1_sym, RDM_1_sym.transpose())
        assert np.allclose(RDM_2_sym, RDM_2_sym.transpose(2, 3, 0, 1))
        # assert np.allclose(T_1_sym, T_1_sym.transpose())
        # assert np.allclose(T_2_sym, T_2_sym.transpose(2, 3, 0, 1))

        return RDM_1_sym, RDM_2_sym, T_1_sym, T_2_sym

    def _correct_occupation_number(self, RDM_1):
        """correct occupation number from 1-RDM"""
        # diagonalize 1-RDM to get occupation number and natural orbitals
        occupation_number, natural_orbital = np.linalg.eigh(RDM_1)
        for index, N_occ in enumerate(occupation_number):
            if N_occ > 1:
                occupation_number[index] = 1
            if N_occ < 0:
                occupation_number[index] = 0

        # correct total number of electron
        tmp = occupation_number * (np.ones_like(occupation_number) - occupation_number)
        X = sum(occupation_number)
        Y = sum(tmp)
        if Y == 0:
            L_lambda = 0
        else:
            L_lambda = (self.n_occ - X) / Y
        occupation_number_correct = occupation_number + L_lambda * tmp

        # transfrom the corrected occupation number to the corrected 1-RDM through orbital rotation
        RDM_1_correct = np.dot(natural_orbital, np.dot(np.diag(occupation_number_correct), natural_orbital.transpose()))

        # reverse mapping the correct 1-RDM to T_1 amplitude
        T_1_correct = np.zeros_like(RDM_1_correct)
        T_1_correct += RDM_1_correct
        T_1_correct -= np.einsum('p,q,pq->pq', self.sin_theta, self.sin_theta, np.eye(self.M))
        T_1_correct /= np.einsum('q,p->pq', self.cos_theta, self.sin_theta)

        return occupation_number_correct, RDM_1_correct, T_1_correct

    def _calculate_full_two_body_cumulant(self, T_2):
        """calculation full (four index) two body cumulant"""
        C_2 = np.einsum('p,q,r,s,psqr->pqrs', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, T_2)
        return C_2

    def _calculate_two_body_direct_cumulant(self, T_2):
        """calculate diagonal two-body cumulant from T_2 amplitude"""
        C_2_direct = np.einsum('p,q,p,q,ppqq->pq', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, T_2)
        return C_2_direct

    def _calculate_two_body_exchange_cumulant(self, T_2):
        """calculate exchange two-body cumulant from T_2 amplitude"""
        C_2_exchange = np.einsum('p,q,p,q,pqqp->pq', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, T_2)
        return C_2_exchange

    def _calculate_P_cumulant(self, RDM_1, C_2):
        """calulate P cumulant from 1-RDM and 2-body cumulant"""
        P_cumulant = C_2 + np.einsum('pp,qq->pq', RDM_1, RDM_1)
        return P_cumulant

    def _calculate_Q_cumulant(self, RDM_1, C_2):
        """calulate Q cumulant from 1-RDM and 2-body cumulant"""
        one_sub_RDM_1 = np.eye(self.M) - RDM_1
        Q_cumulant = C_2 + np.einsum('pp,qq->pq', one_sub_RDM_1, one_sub_RDM_1)
        return Q_cumulant

    def _calculate_G_cumulant(self, RDM_1, C_2):
        """calulate G cumulant from 1-RDM and 2-body cumulant"""
        one_sub_RDM_1 = np.eye(self.M) - RDM_1
        G_cumulant = -C_2 + np.einsum('pp,qq->pq', RDM_1, one_sub_RDM_1)
        return G_cumulant

    def _check_P_Q_G_condition(self, RDM_1, T_2):
        """exam N-representability condition P, Q, G"""
        # calucate diagonal two body cumulant
        C_2 = self._calculate_two_body_direct_cumulant(T_2)

        # calculate P cumulant
        P_cumulant = self._calculate_P_cumulant(RDM_1, C_2)

        # calculate Q cumulant
        Q_cumulant = self._calculate_Q_cumulant(RDM_1, C_2)

        # calculate G cumulant
        G_cumulant = self._calculate_G_cumulant(RDM_1, C_2)

        # exam P condition
        # assert P_cumulant.min() > 0

        # exam Q condition
        # assert Q_cumulant.min() > 0

        # exam G condition
        # assert G_cumulant.min() > 0

        return P_cumulant, Q_cumulant, G_cumulant

    def _correct_two_body_density_matrix(self, RDM_1, T_2):
        """implement correction to 2-RDM that satisfy N-representability condition"""
        def cal_upper_bound(diag_1_RDM, diag_1_RDM_bar):
            """calculate upper bound of two body cumulant"""
            upper_bound = np.zeros([self.M, self.M])

            for p, q in it.product(range(self.M), repeat=2):
                upper_bound[p, q] = min(diag_1_RDM[p] * diag_1_RDM_bar[q], diag_1_RDM[q] * diag_1_RDM_bar[p])

            return upper_bound

        def cal_lower_bound(diag_1_RDM, diag_1_RDM_bar):
            """calculate upper bound of two body cumulant"""
            lower_bound = np.zeros([self.M, self.M])

            for p, q in it.product(range(self.M), repeat=2):
                lower_bound[p, q] = -min(diag_1_RDM[p] * diag_1_RDM[q], diag_1_RDM_bar[p] * diag_1_RDM_bar[q])

            return lower_bound

        # evaluate n_p and 1 - n_p
        diag_1_RDM = np.diag(RDM_1)
        diag_1_RDM_bar = np.ones_like(diag_1_RDM) - diag_1_RDM

        # evaluate two body cumulant
        C_2 = self._calculate_two_body_direct_cumulant(T_2)

        # evaulate lower bound
        lower_bound = cal_lower_bound(diag_1_RDM, diag_1_RDM_bar)

        # evaluate upper bound
        upper_bound = cal_upper_bound(diag_1_RDM, diag_1_RDM_bar)

        # correct two body cumulant if it exceed the physical boundary
        for p, q in it.product(range(self.M), repeat=2):
            if C_2[p, q] > upper_bound[p, q]:
                C_2[p, q] = upper_bound[p, q]
            elif C_2[p, q] < lower_bound[p, q]:
                C_2[p, q] = lower_bound[p, q]
            else:
                pass

        # map matrix element of C_2 to T_2
        T_2_correct = C_2 / np.einsum('p,q,p,q->pq', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta)

        return T_2_correct

    def _correct_exchange_two_body_cumulant(self, RDM_1, T_2):
        """correct exchange part of the two body cumulant when it exceed the physical boundary"""
        def cal_upper_bound(C_2_direct, C_2_exchange, RDM_1):
            """calculate upper bound of the exchange cumulant"""
            def cal_exchange_part_of_upper_bound(C_2_direct, C_2_exchange, RDM_1):
                """calculate exchange part of upper bound by looping over internal label q"""
                upper_bound_exchange = np.zeros([self.M, self.M])
                # find min diagonal element of 1-RDM
                min_1_RDM = min(np.diag(RDM_1))
                # find min element of direct two body cumulant
                min_C_2_direct = np.zeros(self.M)
                for p in range(self.M):
                    min_C_2_direct[p] = min(C_2_direct[p, :])

                for p, r in it.product(range(self.M), repeat=2):
                    upper_bound_exchange[p, r] += min_1_RDM
                    upper_bound_exchange[p, r] -= min_C_2_direct[p]
                    upper_bound_exchange[p, r] -= min_C_2_direct[r]
                    upper_bound_exchange[p, r] -= min_1_RDM * RDM_1[r, r]
                    upper_bound_exchange[p, r] -= RDM_1[r, r] * RDM_1[p, p]

                return upper_bound_exchange

            upper_bound = np.zeros([self.M, self.M])
            upper_bound += C_2_direct
            upper_bound += np.einsum('pp,rr->pr', RDM_1, RDM_1)
            upper_bound -= np.einsum('pr,rp->pr', RDM_1, RDM_1)

            # calculate exchange part of the upper bound
            upper_bound_exchange = cal_exchange_part_of_upper_bound(C_2_direct, C_2_exchange, RDM_1)
            upper_bound += upper_bound_exchange

            return upper_bound

        def cal_lower_bound(C_2_direct, C_2_exchange, RDM_1):
            """calculate lower bound of the exchange cumulant"""
            lower_bound = np.zeros([self.M, self.M])
            lower_bound += C_2_direct
            lower_bound += np.einsum('pp,rr->pr', RDM_1, RDM_1)
            lower_bound -= np.einsum('pr,rp->pr', RDM_1, RDM_1)

            for p, r in it.product(range(self.M), repeat=2):
                lower_bound[p, r] -= min(RDM_1[p, p], RDM_1[r, r])

            return lower_bound

        # calculate two body cumulant
        C_2_direct = self._calculate_two_body_direct_cumulant(T_2)
        C_2_exchange = self._calculate_two_body_exchange_cumulant(T_2)

        # calculate upper bound
        upper_bound = cal_upper_bound(C_2_direct, C_2_exchange, RDM_1)

        # calculate lower bound
        lower_bound = cal_lower_bound(C_2_direct, C_2_exchange, RDM_1)

        # set constraint on two body exchange cumulant when it exceed the physical boundary
        for p, r in it.product(range(self.M), repeat=2):
            if p != r:
                if C_2_exchange[p, r] > upper_bound[p, r]:
                    C_2_exchange[p, r] = upper_bound[p, r]
                elif C_2_exchange[p, r] < lower_bound[p, r]:
                    C_2_exchange[p, r] = lower_bound[p, r]
                else:
                    pass
            else:
                pass

        # reverse map C_2_exchange to T_2
        T_2_exchange = C_2_exchange / np.einsum('p,r,r,p->pr', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta)

        return T_2_exchange

    def _calculate_chemical_potential(self, R_1, T):
        """calculate chemical potential to fix total number of electrons"""
        delta_1, delta_2 = \
                        update_amps(T['t_1'], T['t_2'], self.n_tilde, np.zeros([self.M, self.M, self.M, self.M]), flag=True)
        chemical_potential = np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, R_1)
        chemical_potential /= np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, delta_1)

        return chemical_potential, delta_1, delta_2

    def _calculate_partial_trace_residue(self, RDM_1, T_2):
        """calculate partial trace residue of 2-RDM"""
        # calculate two body cumulant
        C_2 = self._calculate_full_two_body_cumulant(T_2)
        # evlulate trace condition
        residue = 2 * np.einsum('pqrq->pr', C_2)
        residue -= np.einsum('pqqr->pr', C_2)
        residue += RDM_1
        residue -= np.dot(RDM_1, RDM_1)
        return residue

    def _constraint_partial_trace(self, R_2, T):
        """add constraint for partial trace residue once it hits the physical boundary"""
        zero_one_int = {
           "ij": np.zeros([self.M, self.M]),
           "ab": np.zeros([self.M, self.M]),
           "ia": np.zeros([self.M, self.M]),
           "ai": np.zeros([self.M, self.M]),
        }
        delta_1, delta_2 = \
                        update_amps(T['t_1'], T['t_2'], zero_one_int, self.O_tilde, flag=False)

        langrange_multiplier = np.einsum('p,p,p,p,pppp->', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, R_2)
        langrange_multiplier /= np.einsum('p,p,p,p,pppp->', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, delta_2)

        return langrange_multiplier, delta_1, delta_2

    def TFCC_integration(self, T_final, N, direct_flag=True, exchange_flag=True):
        """conduct imaginary time integration (first order Euler scheme) to calculate thermal properties"""
        # map initial T amplitude from reduced density matrix at zero beta
        ## 1-RDM
        RDM_1 = np.eye(self.M) * self.f

        # mapping initial T amplitudes from RDMs
        ## mapping T_2
        t_2 = np.zeros([self.M, self.M, self.M, self.M])

        ## mapping T_1
        t_1 = np.zeros([self.M, self.M])
        t_1 += RDM_1.transpose()
        t_1 -= np.einsum('p,q,pq->qp', self.sin_theta, self.sin_theta, np.eye(self.M))
        t_1 /= np.einsum('q,p->qp', self.cos_theta, self.sin_theta)

        # map initial constant amplitude (at zero beta)
        f = self.f
        t_0 = self.M * np.log(1 + f / (1 - f))

        # store T amplitude in a dictionary
        T = {"t_2": t_2, "t_1": t_1, "t_0": t_0}

        # compute final beta
        beta_final = 1. / (self.kb * T_final)

        # initial beta (starting point of the integration)
        beta_tmp = 0.

        dtau = (beta_final - beta_tmp) / N

        beta_grid = np.linspace(beta_tmp + dtau, beta_final, N-1)
        self.T_grid = 1. / (self.kb * beta_grid)
        self.E_th = []
        self.Z_th = []
        self.n_el_th = []
        self.mu_th = []
        self.occ = []
        # initial P, Q, G plots
        self.P_Q_G_condition = {"T(K)": self.T_grid, "P": [], "Q": [], "G": []}

        # initial trace residue plot
        self.trace_condition = {"T(K)": self.T_grid, "R": []}

        # imaginary propagation
        for i in range(N):
            # amplitude equation
            R_1, R_2 = update_amps(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)
            # energy equation
            E = energy(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)

            # shift core electron energy
            E += self.E_core

            # apply constant shift to energy equation
            E += self.E_0

            # apply nuclear repulsion energy to energy equation
            E += self.E_NN

            if self.chemical_potential:
                # if beta_tmp < 1. / (self.kb * 5e4):
                # compute chemical potential
                mu, delta_1, delta_2 = self._calculate_chemical_potential(R_1, T)
                # apply chemical potential to CC residue
                R_1 -= mu * delta_1
                R_2 -= mu * delta_2

                # E -= mu * self.n_occ
            if self.partial_trace_condition:
                # if beta_tmp < 1. / (self.kb * 5e4):
                # correct partial trace residue once it hits the physical boundary
                trace_residue = self._calculate_partial_trace_residue(RDM_1, T['t_2'])
                langrange_multiplier, delta_1, delta_2 = self._constraint_partial_trace(R_2, T)
                R_1 -= langrange_multiplier * delta_1
                R_2 -= langrange_multiplier * delta_2

            # update CC amplitude
            if self.T_2_flag:
                T['t_2'] -= R_2 * dtau

            T['t_1'] -= R_1 * dtau
            T['t_0'] -= E * dtau
            # (E - mu * self.n_occ) * dtau

            # compute RDM

            # 1-RDM
            RDM_1 = np.einsum('p,q,pq->pq', self.sin_theta, self.sin_theta, np.eye(self.M)) +\
                    np.einsum('q,p,qp->pq', self.cos_theta, self.sin_theta, T['t_1'])
            # 2-RDM (chemist's notation)
            RDM_2 = np.einsum('pr,qs->pqrs', RDM_1, RDM_1)
            RDM_2 -= np.einsum('ps,qr->pqrs', RDM_1, RDM_1)
            RDM_2 += np.einsum('r,s,p,q,rspq->pqrs', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta, T['t_2'])

            # symmetrize density matrix and T amplitude
            RDM_1_sym, RDM_2_sym, T_1_sym, T_2_sym = self._symmetrize_density_matrix(RDM_1, RDM_2)
            if self.T_2_flag:
                T['t_2'] = T_2_sym

            # correct the occupation number
            occupation_number_correct, RDM_1_correct, T_1_correct = self._correct_occupation_number(RDM_1_sym)
            T['t_1'] = T_1_correct

            if direct_flag:
                # correct direct two body density matrix
                T_2_correct_direct = self._correct_two_body_density_matrix(RDM_1, T['t_2'])
                for p, q in it.product(range(self.M), repeat=2):
                    T['t_2'][p, p, q, q] = T_2_correct_direct[p, q]

            if exchange_flag:
                # correct exchange two body density matrix
                T_2_correct_exchange = self._correct_exchange_two_body_cumulant(RDM_1, T['t_2'])
                for p, q in it.product(range(self.M), repeat=2):
                    if p != q:
                        T['t_2'][p, q, q, p] = T_2_correct_exchange[p, q]
                    else:
                        pass

            # check N representability condition
            P_cumulant, Q_cumulant, G_cumulant = self._check_P_Q_G_condition(RDM_1, T['t_2'])
            # number of electron
            n_el = sum(occupation_number_correct)
            # check trace condition
            trace_residue = self._calculate_partial_trace_residue(RDM_1, T['t_2'])

            # print and store properties along the propagation
            if i != 0:
                print("Temperature: {:.3f} K".format(1. / (self.kb * beta_tmp)))
                print("max 1-RDM:\n{:.3f}".format(abs(RDM_1_correct).max()))
                print("max 2-RDM:\n{:.3f}".format(abs(RDM_2_sym).max()))
                print("number of electron:{:.3f}".format(n_el))
                if self.chemical_potential:
                    print("chemical potential:{:} cm-1".format(mu))
                if self.partial_trace_condition:
                    print("langrange multilpier for partial trace constraint:{:} cm-1".format(langrange_multiplier))
                print("occupation number:\n{:}".format(occupation_number_correct))
                print("thermal internal energy:{:.3f}".format(E))
                print("*** N representability condition")
                print("P condition:{:.5f}".format(P_cumulant.min()))
                print("Q condition:{:.5f}".format(Q_cumulant.min()))
                print("G condition:{:.5f}".format(G_cumulant.min()))
                print("Trace condition:")
                print("trace of two body density matrix:{:.5f}".format(np.einsum('pppp->', RDM_2)))
                print("trace of two body cumulant residue:{:.5f}".format(np.trace(trace_residue)))

                # store thermal internal energy
                self.E_th.append(E)
                # store chemical potential
                if self.chemical_potential:
                    self.mu_th.append(mu)
                else:
                    self.mu_th.append(0)
                # store total number of electrons
                self.n_el_th.append(n_el)
                # store partition function
                self.Z_th.append(np.exp(T['t_0']))
                # store occupation number
                self.occ.append(occupation_number_correct)

                # store data for P Q G condition
                self.P_Q_G_condition['P'].append(P_cumulant.min())
                self.P_Q_G_condition['Q'].append(Q_cumulant.min())
                self.P_Q_G_condition['G'].append(G_cumulant.min())

                # store data for trace residue
                self.trace_condition['R'].append(np.trace(trace_residue))
                # break

            beta_tmp += dtau
        # store plot for trace condition
        df = pd.DataFrame(self.trace_condition)
        if self.T_2_flag:
            df.to_csv("trace_condition_TFCC_CCSD_sym.csv", index=False)
        else:
            df.to_csv("trace_condition_TFCC_CCS_sym.csv", index=False)
        # store plot for P Q G condition
        df = pd.DataFrame(self.P_Q_G_condition)
        if self.T_2_flag:
            df.to_csv("P_Q_G_condition_TFCC_CCSD_sym.csv", index=False)
        else:
            df.to_csv("P_Q_G_condition_TFCC_CCS_sym.csv", index=False)


        # store thermal property data
        thermal_prop = {"T": self.T_grid, "Z": self.Z_th, "mu": self.mu_th,
                        "U": self.E_th, "n_el": self.n_el_th}

        df = pd.DataFrame(thermal_prop)
        if self.T_2_flag:
            df.to_csv("thermal_properties_TFCC_CCSD_sym.csv", index=False)
        else:
            df.to_csv("thermal_properties_TFCC_CCS_sym.csv", index=False)

        # store occupation number data (from CC)
        occ_dic = {"T": self.T_grid}
        for i in range(self.M):
            occ_data = []
            for j in range(len(self.T_grid)):
                occ_data.append(self.occ[j][i])
            occ_dic[i] = occ_data

        df = pd.DataFrame(occ_dic)
        if self.T_2_flag:
            df.to_csv("occupation_number_TFCC_CCSD_sym.csv")
        else:
            df.to_csv("occupation_number_TFCC_CCS_sym.csv", index=False)
        return

    def Plot_thermal(self, compare=False):
        """plot thermal properties as function of temperature"""
        # read therma property data
        thermal_prop = pd.read_csv("thermal_properties_TFCC.csv")
        if compare:
            thermal_prop_FD = pd.read_csv("thermal_properties_FD.csv")
            thermal_prop_NOE = pd.read_csv("thermal_properties_NOE.csv")
            thermal_prop_NOE_quasi = pd.read_csv("thermal_properties_NOE_quasi.csv")
        # plot thermal properties
        plt.plot(thermal_prop["T"], thermal_prop["U"], label="TFCC", linestyle="-", alpha=.5)
        if compare:
            plt.plot(thermal_prop_FD["T"], thermal_prop_FD["U"], label="FD", linestyle="dotted", alpha=.8)
            plt.plot(thermal_prop_NOE["T"], thermal_prop_NOE["U"], label="NOE", linestyle="-.", alpha=.8)
            plt.plot(thermal_prop_NOE_quasi["T"], thermal_prop_NOE_quasi["U"], label="NOE quasi", linestyle="--", alpha=.8)
        plt.xlabel("Temp (K)")
        plt.ylabel("Energy(cm-1)")
        plt.title("Plot of thermal internal Energy")
        # plt.xlim(10, 1000)
        plt.legend()
        plt.show()

        plt.plot(thermal_prop["T"], thermal_prop["n_el"])
        plt.xlabel("Temp (K)")
        plt.ylabel("total number of electrons")
        plt.title("Plot of # electrons vs T")
        # plt.ylim(0, 10)
        plt.show()

        # plt.plot(thermal_prop["T"], thermal_prop["mu"], label="TFCC", linestyle="-", alpha=.5)
        if compare:
            plt.plot(thermal_prop_FD["T"], thermal_prop_FD["mu"], label="FD", linestyle="dotted", alpha=.8)
            plt.plot(thermal_prop_NOE["T"], thermal_prop_NOE["mu"], label="NOE", linestyle="-.", alpha=.8)
            plt.plot(thermal_prop_NOE_quasi["T"], thermal_prop_NOE_quasi["mu"], label="NOE quasi", linestyle="--", alpha=.8)

        # plt.xlabel("Temp (K)")
        # plt.ylabel("chemical potential (cm-1)")
        # plt.title("Plot of chemical potential vs T")
        # plt.xlim(10, 1000)
        # plt.ylim(-10, 10)
        # plt.legend()
        # plt.show()

        plt.plot(thermal_prop["T"], np.log(thermal_prop["Z"]), label="TFCC", linestyle="-", alpha=1.)
        if compare:
            plt.plot(thermal_prop_FD["T"], np.log(thermal_prop_FD["Z"]), label="FD", linestyle="dotted", alpha=.8)
            plt.plot(thermal_prop_NOE["T"], np.log(thermal_prop_NOE["Z"]), label="NOE", linestyle="-.", alpha=.8)
            plt.plot(thermal_prop_NOE_quasi["T"], np.log(thermal_prop_NOE_quasi["Z"]), label="NOE quasi", linestyle="--", alpha=.8)

        plt.xlabel("Temp (K)")
        plt.ylabel("partition function (in log scale)")
        plt.title("Plot of partition function vs T")
        # plt.xlim(10, 1000)
        plt.legend()
        plt.show()

        # read in occupation number data
        occ_TFCC = pd.read_csv("occupation_number_TFCC.csv")
        if compare:
            occ_FD = pd.read_csv("occupation_number_FD.csv")
            occ_NOE = pd.read_csv("occupation_number_NOE.csv")
        # plot occupation number
        for i in range(self.M):
            plt.plot(occ_TFCC["T"], occ_TFCC[str(i)], label="orbital {:} TFCC".format(i), linestyle='-', alpha=.5)
            if compare:
                plt.plot(occ_FD["T"], occ_FD[str(i)], label="orbital {:} FD".format(i), linestyle='dotted', alpha=.8)
                plt.plot(occ_NOE["T"], occ_NOE[str(i)], label="orbital {:} NOE".format(i), linestyle='-.', alpha=.8)

        plt.xlabel("Temp (K)")
        plt.ylabel("occupation number")
        plt.title("Plot of occupation number vs T")
        # plt.xlim(10, 1000)
        plt.legend()
        plt.show()

        return
