import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import factorial, isclose
from scipy.linalg import eigh
from CC_residue import *


class two_body_model():
    """ Define a object that implement thermal field coupled_cluster
        method and thermal NOE method
        for GS and thermal property calculations for two-electron Hamiltonian """
    def __init__(self, E_0, Fock, S, V_eri, n_occ):
        """
        E_0: energy expectation value
        Fock: Fock matrix
        S: overlap matrix
        V_eri: 2-electron integral
        M: dimension of MO basis
        """
        print("***<start of input parameters>***")
        self.E_0 = E_0
        self.F = Fock
        self.V = V_eri
        self.n_occ = n_occ
        self.M = self.F.shape[0]
        self.S = S

        # f and f_bar defines contraction in thermal NOE
        self.f = self.n_occ / self.M
        self.f_bar = 1 - self.f

        # factorize prefactors
        self.one_half = 1. / 2.
        self.one_fourth = 1. / 4.
        # initialize Hamiltonian
        # constant part of the full Hamiltonian (H_0)
        print("constant part of Hamiltonian (H_0):\n{:}".format(self.E_0))
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

        return n_p, mu

    def thermal_field_transform(self, T):
        """conduct Bogoliubov transform on physical Hamiltonian"""
        # calculation recprical termperature
        beta = 1. / (self.kb * T)
        print("beta:{:.3f}".format(beta))

        # Diagonalize one-electron Hamiltonian (FC = SCE)
        E_mf, V_mf = eigh(self.F, self.S)
        print("Eigenvalue of Fock matrix\n:{:}".format(E_mf))

        # calculate Femi-Dirac occupation number with chemical potential that fixex total number of electron
        n_p, mu = self.Cal_mu_FD(E_mf, beta)
        # n_p = self.f * np.ones(self.M) # assume zero beta occupation number
        print("Fermi-Dirac Occupation number:\n{:}".format(n_p))
        print("initial chemical potential:{:.3f}".format(mu))

        # Bogoliubov transform the Hamiltonian from physical space to quasi-particle representation
        sin_theta = np.sqrt(n_p)
        cos_theta = np.sqrt(1 - n_p)

        # store sin_theta and cos_theta as object as instance
        self.sin_theta = sin_theta
        self.cos_theta = cos_theta

        # 1-electron Hamiltonian
        self.F_tilde = {"ij": np.einsum('i,j,ij->ij', sin_theta, sin_theta, self.F),
                        "ai": np.einsum('a,i,ai->ai', cos_theta, sin_theta, self.F),
                        "ia": np.einsum('i,a,ia->ia', sin_theta, cos_theta, self.F),
                        "ab": np.einsum('a,b,ab->ab', cos_theta, cos_theta, self.F)}
        print("1-electron Hamiltonian in quasi-particle representation:")
        print("F_tilde_ij:\n{:}".format(self.F_tilde['ij'].shape))
        print("F_tilde_ai:\n{:}".format(self.F_tilde['ai'].shape))
        print("F_tilde_ia:\n{:}".format(self.F_tilde['ia'].shape))
        print("F_tilde_ab:\n{:}".format(self.F_tilde['ab'].shape))

        # 2-electron Hamiltonian (16 terms)
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

        return

    def R_0(self, T):
        """ constant residue (energy equation) """
        R = 0.
        R += self.one_fourth * np.einsum('ijab,abij->', self.V_tilde['ijab'], T['abij'])
        R += np.einsum('ia,ai->', self.F_tilde['ia'], T['ai'])
        R += self.one_half * np.einsum('ijab,ai,bj->', self.V_tilde['ijab'], T['ai'], T['ai'])
        return R_0

    def R_1(self, T):
        """ singles residue """
        R = np.zeros([self.M, self.M])
        R += self.F_tilde['ai']
        R += np.einsum('kc,acik->ai', self.F_tilde['ia'], T['abij'])
        R += self.one_half * np.einsum('akcd,cdik->ai', self.V_tilde['aibc'], T['abij'])
        R -= self.one_half * np.einsum('klic,ackl->ai', self.V_tilde['ijka'], T['abij'])
        R += np.einsum('ac,ci->ai', self.F_tilde['ab'], T['ai'])
        R -= np.einsum('ki,ak->ai', self.F_tilde['ij'], T['ai'])
        R += np.einsum('akic,ck->ai', self.V_tilde['aijb'], T['ai'])
        R -= self.one_half * np.einsum('klcd,ci,adkl->ai', self.V_tilde['ijab'], T['ai'], T['abij'])
        R -= self.one_half * np.einsum('klcd,ak,cdil->ai', self.V_tilde['ijab'], T['ai'], T['abij'])
        R += np.einsum('klcd,ck,dali->ai', self.V_tilde['ijab'], T['ai'], T['iabij'])
        R -= np.einsum('kc,ci,ak->ai', self.F_tilde['ia'], T['ai'], T['ai'])
        R += np.einsum('akcd,ci,dk->ai', self.V_tilde['aibc'], T['ai'], T['ai'])
        R -= np.einsum('klic,ak,cl->ai', self.V_tilde['ijka'], T['ai'], T['ai'])
        R += np.einsum('klcd,ci,ak,dl->ai', self.V_tilde['ijab'], T['ai'], T['ai'], T['ai'])

        return R

    def R_2(self, T):
        """ double residue"""
        R = np.zeros([self.M, self.M, self.M, self.M])

        def Cal_CCD(T):
            """calculate CCD residue"""
            def Cal_perm(T):
                """calculate the four fold permutation term in CCD contribution"""
                R += np.einsum('kbcj,acik->abij', self.V_tilde['iabj'], T['abij'])
                R += np.einsum('kbci,acjk->abij', self.V_tilde['iabj'], T['abij'])
                R += np.einsum('kacj,bcik->abij', self.V_tilde['iabj'], T['abij'])
                R += np.einsum('kaci,bcjk->abij', self.V_tilde['iabj'], T['abij'])

                return

            Cal_perm(T)

            R += self.V_tilde['abij']

            R += np.einsum('bc,acij->abij', self.F_tilde['ab'], T['abij'])
            R -= np.einsum('ac,bcij->abij', self.F_tilde['ab'], T['abij'])

            R -= np.einsum('kj,abik->abij', self.F_tilde['ij'], T['abij'])
            R += np.einsum('ki,abjk->abij', self.F_tilde['ij'], T['abij'])

            R += self.one_half * np.einsum('abcd,cdij->abij', self.V_tilde['abcd'], T['abij'])

            R += self.one_half * np.einsum('ijkl,abkl->abij', self.V_tilde['ijkl'], T['abij'])

            R += self.one_fourth * np.einsum('klcd,cdij,abkl->abij', self.V_tilde['ijab'], T['abij'], T['abij'])

            R += np.einsum('klcd,acik,bdjl->abij', self.V_tilde['ijab'], T['abij'], T['abij'])
            R -= np.einsum('klcd,acjk,bdil->abij', self.V_tilde['ijab'], T['abij'], T['abij'])

            R -= self.one_half * np.einsum('klcd,dcik,ablj->abij', self.V_tilde['ijab'], T['abij'], T['abij'])
            R += self.one_half * np.einsum('klcd,dcjk,abli->abij', self.V_tilde['ijab'], T['abij'], T['abij'])

            R -= self.one_half * np.einsum('klcd,aclk,dbij->abij', self.V_tilde['ijab'], T['abij'], T['abij'])
            R += self.one_half * np.einsum('klcd,bclk,daij->abij', self.V_tilde['ijab'], T['abij'], T['abij'])

            return

        def Cal_perm(T):
            """calculate the term that has four fold permutations"""
            R += np.einsum('akcd,ci,dbkj->abij', self.V_tilde['aibc'], T['ai'], T['abij'])
            R += np.einsum('akcd,cj,dbki->abij', self.V_tilde['aibc'], T['ai'], T['abij'])
            R += np.einsum('bkcd,ci,dakj->abij', self.V_tilde['aibc'], T['ai'], T['abij'])
            R += np.einsum('bkcd,cj,daki->abij', self.V_tilde['aibc'], T['ai'], T['abij'])

            R -= np.einsum('klic,ak,cblj->abij', self.V_tilde['ijka'], T['ai'], T['abij'])
            R -= np.einsum('kljc,ak,cbli->abij', self.V_tilde['ijka'], T['ai'], T['abij'])
            R -= np.einsum('klic,bk,calj->abij', self.V_tilde['ijka'], T['ai'], T['abij'])
            R -= np.einsum('kljc,bk,cali->abij', self.V_tilde['ijka'], T['ai'], T['abij'])

            R -= np.einsum('kbcj,ci,ak->abij', self.V_tilde['iabj'], T['ai'], T['ai'])
            R -= np.einsum('kbci,cj,ak->abij', self.V_tilde['iabj'], T['ai'], T['ai'])
            R -= np.einsum('kacj,ci,bk->abij', self.V_tilde['iabj'], T['ai'], T['ai'])
            R -= np.einsum('kaci,cj,bk->abij', self.V_tilde['iabj'], T['ai'], T['ai'])

            R -= np.einsum('klcj,ci,ak,bl->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'], T['ai'])
            R -= np.einsum('klci,cj,ak,bl->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'], T['ai'])
            R -= np.einsum('klcj,ci,bk,al->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'], T['ai'])
            R -= np.einsum('klci,cj,bk,al->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'], T['ai'])

            return
        # calculate CCD residue
        Cal_CCD(T)
        Cal_perm(T)

        R += np.einsum('abcj,ci->abij', self.V_tilde['abci'], T['ai'])
        R -= np.einsum('abci,cj->abij', self.V_tilde['abci'], T['ai'])

        R -= np.einsum('kbij,ak->abij', self.V_tilde['iajk'], T['ai'])
        R += np.einsum('kaij,bk->abij', self.V_tilde['iajk'], T['ai'])

        R -= np.einsum('kc,ci,abkj->abij', self.F_tilde['ia'], T['ai'], T['abij'])
        R += np.einsum('kc,cj,abki->abij', self.F_tilde['ia'], T['ai'], T['abij'])

        R -= np.einsum('kc,ak,cbij->abij', self.F_tilde['ia'], T['ai'], T['abij'])
        R += np.einsum('kc,bk,caij->abij', self.F_tilde['ia'], T['ai'], T['abij'])

        R -= self.one_half * np.einsum('kbcd,ak,cdij->abij', self.V_tilde['iabc'], T['ai'], T['ai'], T['ai'])
        R += self.one_half * np.einsum('kacd,bk,cdij->abij', self.V_tilde['iabc'], T['ai'], T['ai'], T['ai'])

        R += self.one_half * np.einsum('klcj,aci,abkl->abij', self.V_tilde['ijak'], T['ai'], T['abij'])
        R -= self.one_half * np.einsum('klci,acj,abkl->abij', self.V_tilde['ijak'], T['ai'], T['abij'])

        R += np.einsum('kacd,ck,dbij->abij', self.V_tilde['iabc'], T['ai'], T['abij'])
        R -= np.einsum('kbcd,ck,daij->abij', self.V_tilde['iabc'], T['ai'], T['abij'])

        R -= np.einsum('klci,ck,ablj->abij', self.V_tilde['ijak'], T['ai'], T['abij'])
        R += np.einsum('klcj,ck,abli->abij', self.V_tilde['ijak'], T['ai'], T['abij'])

        R += np.einsum('abcd,ci,dj->abij', self.V_tilde['abcd'], T['ai'], T['ai'])

        R += np.einsum('klij,ak,bl->abij', self.V_tilde['ijkl'], T['ai'], T['ai'])

        R += self.one_half * np.einsum('klcd,ci,dj,abkl->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])

        R += self.one_half * np.einsum('klcd,ak,bl,cdij->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])

        R -= np.einsum('klcd,ck,di,ablj->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])
        R += np.einsum('klcd,ck,dj,abli->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])

        R -= np.einsum('klcd,ck,al,dbij->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])
        R += np.einsum('klcd,ck,bl,daij->abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['abij'])

        R += np.einsum('kbcd,ci,ak,dj->abij', self.V_tilde['iabc'], T['ai'], T['ai'], T['ai'])
        R -= np.einsum('kacd,ci,bk,dj->abij', self.V_tilde['iabc'], T['ai'], T['ai'], T['ai'])

        R += np.einsum('klcj,ci,ak,bl->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'])
        R -= np.einsum('klci,cj,ak,bl->abij', self.V_tilde['ijak'], T['ai'], T['ai'], T['ai'])

        R += np.einsum('klcd,ci,dj,ak,bl-abij', self.V_tilde['ijab'], T['ai'], T['ai'], T['ai'], T['ai'])

    def thermal_field_coupled_cluster(self, T_final, N):
        """conduct imaginary time integration to calculate thermal properties"""
        # calculation initial T amplitude

        # compute reduced density matrix at zero beta
        ## 1-RDM
        RDM_1 = np.eye(self.M) * self.f

        ## 2-RDM (in chemist's notation)
        RDM_2 = np.zeros([self.M, self.M, self.M, self.M])
        # for p in range(self.M):
            # for q in range(self.M):
                # RDM_2[p, p, q, q] = (self.n_occ - 1) / self.M * self.f

        # mapping initial T amplitudes from RDMs

        ## mapping T_2
        t_2 = np.zeros([self.M, self.M, self.M, self.M])
        t_2 += RDM_2.transpose(2, 3, 0, 1)
        t_2 -= np.einsum('pr,qs->rspq', RDM_1, RDM_1)
        t_2 += np.einsum('ps,qr->rspq', RDM_1, RDM_1)
        t_2 /= np.einsum('r,s,p,q->rspq', self.cos_theta, self.cos_theta, self.sin_theta, self.sin_theta)

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

        # imaginary propagation
        for i in range(N):
            # amplitude equation
            R_1, R_2 = update_amps(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)
            # energy equation
            E = energy(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)


            # compute RDM
            RDM = np.einsum('p,q,pq->pq', self.sin_theta, self.sin_theta, np.eye(self.M)) +\
                  np.einsum('q,p,qp->pq', self.cos_theta, self.sin_theta, T['t_1'])

            # compute occupation number
            occ, nat = np.linalg.eigh(RDM)

            # number of electron
            n_el = np.trace(RDM)

            # introduce chemical potential to correct number of electrons
            # delta = self.Cal_R_1_TF(T, flag=True)
            # mu = np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, R_1)
            # mu /= np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, delta)

            # chemical protential corrections on CC residue
            # R_1 -= mu * delta

            # update CC amplitude
            T['t_2'] -= R_2 * dtau
            T['t_1'] -= R_1 * dtau
            T['t_0'] -= E * dtau
            # (E - mu * self.n_occ) * dtau

            # print and store properties along the propagation
            if i != 0:
                print("Temperature: {:.3f} K".format(1. / (self.kb * beta_tmp)))
                print("1-RDM:\n{:}".format(RDM))
                print("number of electron:{:.3f}".format(n_el))
                # print("chemical potential:{:} cm-1".format(mu))
                print("occupation number:\n{:}".format(occ))
                print("thermal internal energy:{:.3f}".format(E))

                # store thermal internal energy
                self.E_th.append(E)
                # store chemical potential
                # self.mu_th.append(mu)
                # store total number of electrons
                self.n_el_th.append(n_el)
                # store partition function
                self.Z_th.append(np.exp(T['t_0']))
                # store occupation number
                self.occ.append(occ)

                # break

            beta_tmp += dtau

        # store thermal property data
        thermal_prop = {"T": self.T_grid, "Z": self.Z_th,
                         #"mu": self.mu_th,
                        "U": self.E_th, "n_el": self.n_el_th}

        df = pd.DataFrame(thermal_prop)
        df.to_csv("thermal_properties_TFCC.csv", index=False)

        # store occupation number data (from CC)
        occ_dic = {"T": self.T_grid}
        for i in range(self.M):
            occ_data = []
            for j in range(len(self.T_grid)):
                occ_data.append(self.occ[j][i])
            occ_dic[i] = occ_data

        df = pd.DataFrame(occ_dic)
        df.to_csv("occupation_number_TFCC.csv", index=False)
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
