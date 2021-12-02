import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import factorial, isclose
from scipy.linalg import eigh
from CC_residue import *
from scipy.integrate import solve_ivp


class two_body_model():
    """ Define a object that implement thermal field coupled_cluster
        method and thermal NOE method
        for GS and thermal property calculations for two-electron Hamiltonian """
    def __init__(self, E_HF, Fock, V_eri, n_occ):
        """
        E_HF: energy expectation value (Hartree Fock GS energy)
        Fock: fock matrix
        V_eri: 2-electron integral (chemist's notation)
        M: dimension of MO basis
        (all electron integrals are represented in MO basis)
        """
        print("***<start of input parameters>***")
        self.E_HF = E_HF
        self.V = V_eri
        self.n_occ = n_occ

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

    def thermal_field_transform(self, T):
        """conduct Bogoliubov transform on physical Hamiltonian"""
        # calculation recprical termperature
        beta = 1. / (self.kb * T)
        print("beta:{:.3f}".format(beta))

        # Diagonalize Fock matrix
        E_mf, V_mf = np.linalg.eigh(self.F)
        # calculate Femi-Dirac occupation number with chemical potential that fixex total number of electron
        n_p, mu = self.Cal_mu_FD(E_mf, beta)

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

        self.n_tilde = {"ij": np.einsum('i,j,ij->ij', self.sin_theta, self.sin_theta, np.eye(self.M)),
                        "ai": np.einsum('a,i,ai->ai', self.cos_theta, self.sin_theta, np.eye(self.M)),
                        "ia": np.einsum('i,a,ia->ia', self.sin_theta, self.cos_theta, np.eye(self.M)),
                        "ab": np.einsum('a,b,ab->ab', self.cos_theta, self.cos_theta, np.eye(self.M))}

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

    def ravel_T_tensor(self, T):
        """Flatten the T tensor to feed into RK integrator"""
        y_tensor = np.concatenate(
            (
              np.array([T['t_0']]), T['t_1'].ravel(), T['t_2'].ravel()
            )
        )

        return y_tensor

    def unravel_T_tensor(self, y_tensor):
        """Restore the original shape of T tensor"""
        M = self.M
        # restore z tensor

        # constant term
        start_constant_slice_index = 0
        end_constant_slice_index = start_constant_slice_index + 1
        t_0 = y_tensor[0]

        # single term
        start_linear_slice_index = end_constant_slice_index
        end_linear_slice_index = start_linear_slice_index + M*M
        t_1 = np.reshape(
                    y_tensor[start_linear_slice_index: end_linear_slice_index],
                    newshape=(M, M)
        )

        # double term
        start_quadratic_slice_index = end_linear_slice_index
        end_quadratic_slice_index = start_quadratic_slice_index + M * M * M * M
        t_2 = np.reshape(
                    y_tensor[start_quadratic_slice_index: end_quadratic_slice_index],
                    newshape=(M, M, M, M)
        )

        T = {"t_0": t_0, "t_1": t_1, "t_2": t_2}

        return T

    def _print_integration_progress(self, tau, T_final, Z, E, mu, n_el, occ):
        """ Prints to stdout every 1e4 steps or if current fs value is a multiple of (0.1 * `t_final`). """
        if tau != 0:
            print(f"On integration step at {1./ (self.kb * tau):>9.4f} K")
            print("Z:{:.5f}".format(Z))
            print("E:{:.5f} hartree".format(E))
            print("mu:{:.5f} hartree".format(mu))
            print("Z:{:.5f}".format(n_el))
            print("occupation number:\n{:}".format(occ))

        return

    def _rk45_solve_ivp_integration_function(self, tau, y_tensor, T_final, chemical_potential=True):
        """ Integration function used by `solve_ivp` integrator inside `rk45_integration` method."""
        def correct_occupation_number(occ, nat):
            """correct occupation number is n_p <0 or n_p > 1"""
            for i, n in enumerate(occ):
                if n < 0:
                    occ[i] = 0.
                if n > 1:
                    occ[i] = 1.
            # correct occupation number
            c_p = occ * (np.ones_like(occ) - occ)
            lmd = (self.n_occ - sum(occ)) / sum(c_p)
            occ += lmd * c_p
            # print("occ:{:}".format(occ))
            return occ

        # restore the original shape of T tensor
        T = self.unravel_T_tensor(y_tensor)

        # compute properties for CC amplitude

        # 1-RDM
        RDM_1 = np.einsum('p,q,pq->pq', self.sin_theta, self.sin_theta, np.eye(self.M)) +\
                np.einsum('q,p,qp->pq', self.cos_theta, self.sin_theta, T['t_1'])
        # compute occupation number and store them
        occ, nat = np.linalg.eigh(RDM_1)
        if min(occ) < 0 or max(occ) > 1:
            # add correction when occupation number become abnormal
            occ = correct_occupation_number(occ, nat)
        self.n_p.append((tau, occ))

        # compute partition function
        self.Z.append((tau, np.exp(T['t_0'])))

        # update CC amplitudes
        # amplitude equation
        R_1, R_2 = update_amps(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)
        # energy equation
        R_0 = energy(T['t_1'], T['t_2'], self.F_tilde, self.V_tilde)
        # apply constant shift to energy equation
        R_0 += self.E_HF
        self.E.append((tau, R_0))

        # total number of electrons
        self.n_el.append((tau, np.sum(occ)))

        if chemical_potential:
            # compute chemical potential
            delta_1, delta_2 = \
                            update_amps(T['t_1'], T['t_2'], self.n_tilde, np.zeros([self.M, self.M, self.M, self.M]), flag=True)
            mu = np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, R_1)
            mu /= np.einsum('p,p,pp->', self.cos_theta, self.sin_theta, delta_1)

            # apply chemical potential to CC residue
            R_1 -= mu * delta_1
            R_2 -= mu * delta_2
            R_0 -= mu * self.n_occ
            self.mu.append((tau, mu))

        # print process
        self._print_integration_progress(tau, T_final, self.Z[-1][1], self.E[-1][1], mu, self.n_el[-1][1], occ)
        R = {"t_0": -R_0, "t_1": -R_1, "t_2": -R_2}

        delta_y_tensor = self.ravel_T_tensor(R)

        return delta_y_tensor

    def _postprocess_rk45_integration_results(self, sol):
        """process and store data for thermal properties"""
        # convert imaginary time to temperature
        self.T_cc = 1. / (self.kb * sol.t[1:])

        # partition function
        self.Z_cc = np.zeros_like(sol.t)
        Z_dic = {z[0]: z[1] for z in self.Z}
        for idx, t in enumerate(sol.t):
            self.Z_cc[idx] = Z_dic[t]

        # internal energy
        self.E_cc = np.zeros_like(sol.t)
        E_dic = {z[0]: z[1] for z in self.E}
        for idx, t in enumerate(sol.t):
            self.E_cc[idx] = E_dic[t]

        # chemical potential
        self.mu_cc = np.zeros_like(sol.t)
        mu_dic = {z[0]: z[1] for z in self.mu}
        for idx, t in enumerate(sol.t):
            self.mu_cc[idx] = mu_dic[t]

        # number of electrons
        self.n_el_cc = np.zeros_like(sol.t)
        n_el_dic = {z[0]: z[1] for z in self.n_el}
        for idx, t in enumerate(sol.t):
            self.n_el_cc[idx] = n_el_dic[t]

        # occupation number
        self.n_p_cc = np.zeros([len(sol.t), self.M])

        N_dic = {N[0]: N[1] for N in self.n_p}
        for idx, t in enumerate(sol.t):
            self.n_p_cc[idx, :] = N_dic[t]


        # store thermal property data in csv file
        thermal_prop = {"T": self.T_cc, "Z": self.Z_cc[1:], "mu": self.mu_cc[1:],
                        "U": self.E_cc[1:], "n_el": self.n_el_cc[1:]}

        df = pd.DataFrame(thermal_prop)
        df.to_csv("thermal_properties_TFCC.csv", index=False)

        # store occupation number data (from CC)
        occ_dic = {"T": self.T_cc}
        for i in range(self.M):
            occ_data = []
            for j in range(len(sol.t)):
                if j != 0:
                    occ_data.append(self.n_p_cc[j][i])
            occ_dic[i] = occ_data
        df = pd.DataFrame(occ_dic)
        df.to_csv("occupation_number_TFCC.csv", index=False)
        return

    def rk45_integration(self, T_final, N=10000):
        """Apply RK45 integrator of scipy library to conduct numerical integration"""
        # detemine initial and final value of integration
        tau_init = 0
        tau_final = 1. / (T_final * self.kb)
        # initial step step size
        step_size = tau_final / N
        # initialize quantities to be stored
        self.E = []
        self.Z = []
        self.n_p = []
        self.mu = []
        self.n_el = []
        # prepare the initial y_tensor
        # map initial constant amplitude (at zero beta)
        f = self.f
        t_0 = self.M * np.log(1 + f / (1 - f))
        ## 1-RDM
        RDM_1 = np.eye(self.M) * self.f
        ## Mapping T_1
        t_1 = np.zeros([self.M, self.M])
        t_1 += RDM_1.transpose()
        t_1 -= np.einsum('p,q,pq->qp', self.sin_theta, self.sin_theta, np.eye(self.M))
        t_1 /= np.einsum('q,p->qp', self.cos_theta, self.sin_theta)
        ## mapping T_2
        t_2 = np.zeros([self.M, self.M, self.M, self.M])

        initial_y_tensor = self.ravel_T_tensor({"t_0": t_0, "t_1": t_1, "t_2": t_2})

        relative_tolerance = 1e-3
        absolute_tolerance = 1e-6

        integration_function = self._rk45_solve_ivp_integration_function


        sol = solve_ivp(
            fun=integration_function,  # the function we are integrating
            method="RK45",  # the integration method we are using
            first_step=step_size,  # fix the initial step size
            t_span=(
                tau_init,  # initial time
                tau_final,  # boundary time, integration end point
            ),
            y0=initial_y_tensor,  # initial state - shape (n, )
            args=(tau_final, ),  # extra args to pass to `_new_5_rk45_solve_ivp_integration_function`
            # max_step=self.step_size,  # maximum allowed step size
            rtol=relative_tolerance,  # relative tolerance
            atol=absolute_tolerance,  # absolute tolerance
            dense_output=False,  # extra debug information
            # we do not need to vectorize
            # this means to process multiple time steps inside the function `_new_5_rk45_solve_ivp_integration_function`
            # it would be useful for a method which does some kind of block stepping
            vectorized=False,
        )

        self._postprocess_rk45_integration_results(sol)

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
