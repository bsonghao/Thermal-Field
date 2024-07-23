#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate the entire FCI Hamiltonian for small system

See also 36-determinants_basis_matrix.py
'''

# import numpy
from pyscf import fci
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
import itertools as it
import os
import math
from math import factorial
import pandas as pd

def extract_Hamiltonian_parameters(mo_flag, CAS_SCF, mol_HF):
    """
    extract 1-electron integral, overlap matrix, 2-electron integral and fock matrix from  Hamiltonian parameters
    """
    def Cal_core_1_RDM(CI_coefficient, core_orbitals, M):
        """calculate core 1-RDM"""
        size_core = len(core_orbitals)
        RDM_1_core = np.zeros([M, M])
        for alpha in range(M):
            for beta in range(M):
                for i in range(size_core):
                    RDM_1_core[alpha, beta] += CI_coefficient[alpha, i] * CI_coefficient[beta, i]
        # print("core 1-RDM:\n{:}".format(RDM_1_core))

        return RDM_1_core

    def Cal_one_electron_integral(h_core_AO, RDM_1_core, M, eri_AO):
        """calculate one electron integral in model CAS space"""
        h_core_AO_CAS = np.zeros([M, M])

        # add one-electron part
        h_core_AO_CAS += h_core_AO

        # add effective two electron part
        for a, b, c, d in it.product(range(M), repeat=4):
            h_core_AO_CAS[a, b] += RDM_1_core[c, d] * (2 * eri_AO[a, b, c, d] - eri_AO[a, c, b, d])

        return h_core_AO_CAS

    def Cal_two_electron_integral(active_orbitals, eri_AO):
        """calculate two electron integral in model CAS space"""
        size_cas = len(active_orbitals)
        eri_AO_CAS = np.zeros([size_cas, size_cas, size_cas, size_cas])

        # calculate two electron integral is CAS model active_space
        for a, orbital_label_a in enumerate(active_orbitals):
            for b, orbital_label_b in enumerate(active_orbitals):
                for c, orbital_label_c in enumerate(active_orbitals):
                    for d, orbital_label_d in enumerate(active_orbitals):
                        eri_AO_CAS[a, b, c, d] += eri_AO[orbital_label_a, orbital_label_b, orbital_label_c, orbital_label_d]

        return eri_AO_CAS

    def Cal_Fock_ground_state(active_orbitals, Fock):
        """calculate ground state Fock matrix in model active space"""
        size_cas = len(active_orbitals)
        Fock_ground_state = np.zeros([size_cas, size_cas])
        for a, orbital_label_a in enumerate(active_orbitals):
            for b, orbital_label_b in enumerate(active_orbitals):
                Fock_ground_state[a, b] += Fock[orbital_label_a, orbital_label_b]
        return Fock_ground_state

    # get occupation number for CASSCF
    occupation_number = CAS_SCF.mo_occ / 2

    # calcuation number of orbatals
    M = len(occupation_number)

    # get core orbitals and active orbitals
    core_orbitals = []
    active_orbitals = []
    for index, orbital in enumerate(occupation_number):
        if orbital == 1:
            core_orbitals.append(index)
        elif orbital > 0 and orbital < 1:
            active_orbitals.append(index)
        else:
            pass

    print("core orbital labels:{:}".format(core_orbitals))
    print("active orbital labels:{:}".format(active_orbitals))

    # get 1-electron, 2-electron integrals in AO basis from PySCF
    # 1-electron integral
    h_core_AO = mol_HF.intor('int1e_kin_sph') + mol_HF.intor('int1e_nuc_sph')
    print("1-electron integral (in AO basis):\n{:}".format(h_core_AO.shape))

    # Fock matrix
    fock_AO = CAS_SCF.get_fock()
    print("Fock matrix (in AO basis):\n{:}".format(fock_AO.shape))

    # 2-electron integral
    eri_AO = mol_HF.intor('int2e_sph', aosym=1)
    print("2-electron integral (in AO basis):\n{:}".format(eri_AO.shape))

    # overlap matrix
    S_AO = mol_HF.intor('int1e_ovlp_sph')


    # construct model Hamiltonian active space ( in AO basis)
    ## calculate core 1-RDM
    RDM_1_core = Cal_core_1_RDM(CAS_SCF.mo_coeff, core_orbitals, M)
    ## calculate one electron model Hamitonian in active space
    h_core_AO_CAS_full = Cal_one_electron_integral(h_core_AO, RDM_1_core, M, eri_AO)

    ## calculate ground state Fock matrix in active space
    Fock_AO_CAS = Cal_Fock_ground_state(active_orbitals, fock_AO)

    ## calculate term
    E_HF = np.trace(np.dot((h_core_AO_CAS_full + h_core_AO), RDM_1_core))

    print("constant term in CAS: {:}".format(E_HF))

    if not mo_flag:
        h_core_AO_CAS = Cal_Fock_ground_state(active_orbitals, h_core_AO_CAS_full)
        ## calculate two electron model Hamitonian in active space
        eri_AO_CAS = Cal_two_electron_integral(active_orbitals, eri_AO)
        return h_core_AO_CAS, eri_AO_CAS, Fock_AO_CAS, E_HF

    else:

        # construct model Hamiltonian in CAS in MO basis

        # Transfrom original integrals from AO basis to MO basis

        ## calculate nature orbital in active space
        nature_orbital = CAS_SCF.mo_coeff
        ## Fock matrix
        Fock_MO = np.einsum('pi,pq,qj->ij', nature_orbital, fock_AO, nature_orbital)
        print('Ground state Fock matrix (in MO basis):\n{:}'.format(Fock_MO.shape))

        ## 1-electron integral
        h_core_MO = np.einsum('pi,pq,qj->ij', nature_orbital, h_core_AO_CAS_full, nature_orbital)
        print('1-electron integral (in MO basis):\n{:}'.format(h_core_MO.shape))

        ## 2-electron integral
        eri_MO = ao2mo.incore.full(eri_AO, nature_orbital)
        print("2-electron integral (in MO basis):\n{:}".format(eri_MO.shape))

        # get CAS block from the full matrix
        Fock_MO_CAS = Cal_Fock_ground_state(active_orbitals, Fock_MO)
        h_core_MO_CAS = Cal_Fock_ground_state(active_orbitals, h_core_MO)
        eri_MO_CAS = Cal_two_electron_integral(active_orbitals, eri_MO)


        return h_core_MO_CAS, eri_MO_CAS, Fock_MO_CAS, E_HF

def run_FCI_calcuation(h1 ,h2, CAS, E_core, NR_energy):
    """run exact diagonaization calcuation based on CASSCF calculation
    h1: effective one electron integral obtained from the CASSCF calacuation
    h2: effective two electron integral obtained from the CASSCF calacuation
    CAS: active space settings
    E_core: core electron contribution to the total energy
    NR_energy: nuclear repulsion energy
    """
    # run exact diagonalization calcuation to get all energy eigenvalues
    # calcuate total number of configurations
    norb = CAS[0]
    nelec_alpha = CAS[1][0]
    nelec_beta = CAS[1][1]
    Alpha_config = factorial(norb) / (factorial(nelec_alpha) * factorial(norb - nelec_alpha))
    Beta_config =  factorial(norb) / (factorial(nelec_beta) * factorial(norb - nelec_beta))
    ndet = Alpha_config * Beta_config
    # nelec = CAS[1]
    # ndet = factorial(2*norb) / (factorial(nelec)*factorial(2*norb-nelec))

    # form FCI Hamiltonian an diagonalize it
    H_fci = fci.direct_spin1.pspace(h1, h2, norb, CAS[1], np=ndet)[1]
    e_all, v_all = np.linalg.eigh(H_fci)
    # add core electron contribution and the nuclear repulsion energy
    e_all = e_all + E_core + NR_energy
    print("GS energy:", e_all[0])
    print("Number of root:", len(e_all))

    return e_all

def cal_chemical_potential(initial_guess, beta, energy, total_nel, max_threshold=1000):
    """
    implement a Newtonian procedure to calculate the chemical potential
    """
    def cal_z():
        """
        calculate grand canonical partition function
        """
        Z = 0
        for key in energy.keys():
            n_el = key[1][0] + key[1][1] # total # of electron = alpha + beta
            Z += np.exp(X * n_el) * sum(np.exp(-beta * energy[key]))
        return Z

    def cal_n():
        """
        calculate <n>
        """
        Z = cal_z()
        n_avg = 0
        for key in energy.keys():
            n_el = key[1][0] + key[1][1] # total # of electron = alpha + beta
            n_avg += np.exp(X * n_el) * n_el * sum(np.exp(-beta * energy[key]))
        n_avg /= Z
        return Z, n_avg

    def cal_dn():
        """
        calculate dn/dmu
        """
        dn_temp = 0
        for key in energy.keys():
            n_el = key[1][0] + key[1][1]
            dn_temp += n_el**2 * beta * np.exp(X*n_el) * sum(np.exp(-beta * energy[key]))
            dn_temp -= n_temp * n_el * beta * np.exp(X * n_el) * sum(np.exp(-beta * energy[key]))
            dn_temp /= z_temp
        return dn_temp

    X = initial_guess # intialize the chemical potential (X = mu * beta) to be zero
    z_temp, n_temp = cal_n() # initialize partition function and <n>
    print("intial <n>:{:f}".format(n_temp))
    # iteratively update mu
    i = 0
    while( not (np.allclose(total_nel, n_temp, atol=1e-2, rtol=1e-3))):
        k = cal_dn()
        X = (total_nel - n_temp) / k + X
        z_temp, n_temp = cal_n() # update partition function and <n>
        i += 1
        # print("Iteration{:d}:".format(i))
        # print("beta*mu={:f}".format(X))
        # print("n_avg - n_el:", total_nel-n_temp)
        if np.allclose(total_nel, n_temp):
            print("Newtonian procedure converged in {:d} iterations!".format(i))

        if math.isnan(X):
            print("***Warning: Newtonian procedure break, return its initial value!")
            X = 0.
            z_temp, n_temp = cal_n()
            print("Terminate at iteration {:d}, n_avg:{:f}".format(i, n_temp))
            break

        if i > max_threshold:
            print("***Warning: Newtonian procedure do not converge within {:d} iteration".format(max_threshold))
            print("n_avg - n_el:", total_nel-n_temp)
            break

    return X, n_temp

def cal_canonical_thermal(beta, energy, total_nel):
    """
    calcuate thermal properties for canonical ensemble
    """
    temp = np.array([])
    for key in energy.keys():
        n_el = key[1][0] + key[1][1]
        if n_el == total_nel:
            temp = np.concatenate([temp, energy[key]])
    const = temp.min()
    temp -= const
    boltzman_factor = np.exp(-beta * temp)
    # calculate partition function
    Z = sum(boltzman_factor)
    E = sum(temp * boltzman_factor) / Z
    E += const
    return Z, E

def cal_grand_canonical_thermal(beta, energy, chemical_potential):
    """
    calcuate thermal properties for grand canonical ensemble
    """
    temp = np.array([])
    boltzmann_factor = np.array([])
    for key in energy.keys():
        n_el = key[1][0] + key[1][1]
        temp = np.concatenate([temp, energy[key]])
        boltz_temp = np.exp(-beta * energy[key])
        boltz_temp *= np.exp(chemical_potential * n_el)
        boltzmann_factor = np.concatenate([boltzmann_factor, boltz_temp])

    const = temp.min()
    temp -= const
    # calculate partition function
    Z = sum(boltzmann_factor)
    E = sum(temp * boltzmann_factor) / Z
    E += const
    return Z, E


def main():
    # perform CASSCF calcuations
    mo_flag = True

    # geometry of molecules (in Angstrom)
    HF = 'H 0 0 0; F 0 0 1.1'

    H2O = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''

    O2 = 'O 0 0 0; O 0 0 1.2'

    N2 = 'N 0 0 0; N 0 0 1.1'

    # active space of molecules
    CAS_N2 = (6, (3, 3))
    CAS_HF = (4, 6)
    CAS_O2 = (8, (4, 2))

    CAS = CAS_N2
    atom = N2
    molecule = "N2"
    s_mult = CAS[1][0]-CAS[1][1]
    nel_CAS = CAS[1][0] + CAS[1][1]

    molecular_HF = gto.M(
           atom=atom,  # in Angstrom
           basis='ccpvdz',
           # basis="6-31g",
           symmetry=False,
           spin= s_mult,
           charge = 0
    )

    # run RHF calculation
    mean_field = scf.RHF(molecular_HF).run()

    # run CASSCF calculation
    mycas = mean_field.CASSCF(CAS[0], CAS[1])
    mycas.natorb = True
    mycas.kernel()

    # get Nuclear Repusion Energy
    NR_energy = mycas.energy_nuc()
    # extract effective model Hamiltonian from the CASSCF calcuation
    h_core, eri_integral, Fock_ground_state, E_core = \
    extract_Hamiltonian_parameters(mo_flag, mycas, molecular_HF)




    # loop over all configurations and diagonalize
    num_orb = CAS[0]
    energy_dic = {}
    for num_elec in range(num_orb+1):
        for i in range(num_elec+1):
            alpha_elec = i
            beta_elec = num_elec - i
            CAS_FCI = (num_orb, (alpha_elec, beta_elec))
            print("Run calcuation with configuration:",CAS_FCI)
            if num_elec != 0:
                energy_level = run_FCI_calcuation(h_core, eri_integral, CAS_FCI, E_core, NR_energy)
                energy_dic[(CAS_FCI)] = energy_level
            else:
                print("GS energy:", E_core + NR_energy)
                energy_dic[(CAS_FCI)] = np.array([E_core + NR_energy])

    for key in energy_dic.keys():
        print("Configuration:", key)
        print("GS energy: ", energy_dic[key][0])

    # calculation chemical potential using the Newtonian procedure
    # Kb = 3.1668152e-06 # Boltzmann constant Hartree K-1
    # beta = 1. / (Kb * 5e3) # say at 300 K

    # renormalize the energy
    const = 0
    for key in energy_dic.keys():
        if energy_dic[key].min() < const:
            const = energy_dic[key].min()
    for key in energy_dic.keys():
        energy_dic[key] -= const

    # mu =cal_chemical_potential(beta, energy_dic, nel_CAS)
    # print("Converge chemical potential:", mu)

    # calcuate grand canonical partition function
    T = np.linspace(1e3, 1e7, int(1e3))
    data = {
      "T(K)": T,
       "Z":[],
       "E":[],
       "n_el":[],
          }
     # print(T.shape)
    Kb = 3.1668152e-06 # Boltzmann constant Hartree K-1
    initial_guess = 0
    for temperature in T:
         # calculate Boltzmann factor
         beta = 1. / (Kb * temperature)
         mu , n_avg= cal_chemical_potential(initial_guess, beta, energy_dic, nel_CAS)
         initial_guess = mu
         print("Converge chemical potential:", mu)
         part, inter_e = cal_grand_canonical_thermal(beta, energy_dic, mu)
         data["Z"].append(part)
         data["E"].append(inter_e+const)
         data["n_el"].append(n_avg)
     # store thermal data
    df = pd.DataFrame(data)
    df.to_csv("{:}_FCI_fix_grand_canonical_thermal_data.csv".format(molecule))

    # calcuate canonical partition function
    # T = np.linspace(1e3, 1e6, int(1e5))
    # data = {
    # "T(K)": T,
       # "Z":[],
       # "E":[],
          # }
     # print(T.shape)
    # Kb = 3.1668152e-06 # Boltzmann constant Hartree K-1

    # for temperature in T:
         # calculate Boltzmann factor
         # beta = 1. / (Kb * temperature)
         # part, inter_e = cal_canonical_thermal(beta, energy_dic, nel_CAS)
         # data["Z"].append(part)
         # data["E"].append(inter_e)
     # store thermal data
    # df = pd.DataFrame(data)
    # df.to_csv("{:}_FCI_fix_canonical_thermal_data.csv".format(molecule))

    return




if (__name__ == '__main__'):
    main()
