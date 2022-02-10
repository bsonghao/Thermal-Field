# import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
from two_body_modeling import two_body_model


def extract_Hamiltonian_parameters(one_body_hamitonian, V_couple, V_ontop, nelectron):
    """
    extract 1-electron integral, overlap matrix, 2-electron integral and fock matrix Hartree Fock calculation of
    magnetic model Hamiltonian
    """
    def construct_eri(V_couple, V_ontop, basis_size):
        """construct two electron integral from model parameters"""
        # initialize eri
        two_electron_integral = np.zeros([basis_size, basis_size, basis_size, basis_size])
        # enter model parameters
        for p in range(basis_size):
            for q in range(basis_size):
                if p != q:
                    two_electron_integral[p, p, q, q] = V_couple/abs(p-q)
                else:
                    two_electron_integral[p, p, q, q] = V_ontop

        return two_electron_integral

    basis_size = one_body_hamitonian.shape[0]
    mol = gto.M(verbose=4)
    mol.nelectron = nelectron
    # Setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
    # attribute) to be used in the post-HF calculations.  Without this parameter,
    # some post-HF method (particularly in the MO integral transformation) may
    # ignore the customized Hamiltonian if memory is not enough.
    mol.incore_anyway = True

    # construct two electron integral from model parameters
    eri = construct_eri(V_couple, V_ontop, basis_size)

    # overlap matrix
    S = np.eye(basis_size)

    # core Hamiltonian
    h_core = one_body_hamitonian

    # conduct Hartree-Fock Calculation on the model Hamiltonian
    mean_field = scf.RHF(mol)
    mean_field.get_hcore = lambda *args: h_core
    mean_field.get_ovlp = lambda *args: S
    mean_field._eri = ao2mo.restore(8, eri, basis_size)
    mean_field.kernel()


    print("one electron Hamiltonian:\n{:}".format(one_body_hamitonian))
    # get converged Fock matrix from HF calculation
    fock = mean_field.get_fock()
    print("Fock matrix:\n{:}".format(fock.shape))

    # get converge HF energy
    E_Hartree_Fock = mean_field.e_tot
    print("energy expectation value (HF energy) (in cm-1):{:.5f}".format(E_Hartree_Fock))

    # get Nuclear Repusion Energy
    NR_energy = mean_field.energy_nuc()
    print("nuclear repulsion energy (in cm-1):{:.5f}".format(NR_energy))

    return h_core, fock, eri, S, E_Hartree_Fock, NR_energy


def main():
    """main run TFCC approach that conduct imaginary time integration on thermal properties for molecular compounds"""
    # enter model parameters
    V_couple = 10
    V_ontop = 10
    one_body_hamiltonian = np.diag(np.array([-50., -30., -10., 10., 30., 50.]))
    nelectron = 6

    molecule = "magnetic_model_hamiltonian"

    # extract parameter from the input Hamitonian and HF calculation
    h_core, fock_matrix, eri_integral, S_matrix, E_HF, E_NR = extract_Hamiltonian_parameters(one_body_hamiltonian, V_couple, V_ontop, nelectron)

    # total number of electron
    nof_electron =  nelectron / 2
    print("total number of electrons:{:}".format(nof_electron))




    # run TFCC & thermal NOE calculation
    model = two_body_model(E_HF, h_core, fock_matrix, eri_integral, nof_electron, molecule=molecule,
                           E_NN=E_NR, T_2_flag=True, chemical_potential=True, partial_trace_condition=False)
    # thermal field transform
    model.thermal_field_transform(T=1e8)
    # TFCC imaginary time integration
    model.TFCC_integration(T_final=1e0, N=10000, direct_flag=True, exchange_flag=True)
    # plot thermal properties
    model.Plot_thermal()

    return

if (__name__ == '__main__'):
    main()
