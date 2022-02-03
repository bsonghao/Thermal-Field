# import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
from two_body_modeling import two_body_model


def extract_Hamiltonian_parameters(mo_flag, mean_field, mol_HF):
    """
    extract 1-electron integral, overlap matrix, 2-electron integral and fock matrix from  Hamiltonian parameters
    """
    # atomic orbitals

    # 1-electron integral
    h_core_AO = mol_HF.intor('int1e_kin_sph') + mol_HF.intor('int1e_nuc_sph')
    print("1-electron integral (in AO basis):\n{:}".format(h_core_AO.shape))

    # Fock matrix
    fock_AO = mean_field.get_fock()
    print("Fock matrix (in AO basis):\n{:}".format(fock_AO.shape))

    # 2-electron integral
    eri_AO = mol_HF.intor('int2e_sph', aosym=1)
    print("2-electron integral (in AO basis):\n{:}".format(eri_AO.shape))

    # overlap matrix
    S_AO = mol_HF.intor('int1e_ovlp_sph')

    if not mo_flag:
        return h_core_AO, fock_AO, eri_AO, S_AO

    # molecular orbitals

    # 1-electron integral
    h_core_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, h_core_AO, mean_field.mo_coeff)
    print('1-electron integral (in MO basis):\n{:}'.format(h_core_MO.shape))

    # Fock matrix
    fock_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, fock_AO, mean_field.mo_coeff)
    print("Fock matrix (in MO basis):\n{:}".format(fock_MO.shape))

    # 2-electron integral
    eri_MO = ao2mo.incore.full(eri_AO, mean_field.mo_coeff)
    print("2-electron integral (in MO basis):\n{:}".format(eri_MO.shape))

    # overlap matrix
    S_MO = np.einsum('pi,pq,qj->ij', mean_field.mo_coeff, S_AO, mean_field.mo_coeff)
    print("Overlap matrix (in MO basis):\n{:}".format(S_MO.shape))

    return h_core_MO, fock_MO, eri_MO, S_MO


def main():
    """main run TFCC approach that conduct imaginary time integration on thermal properties for molecular compounds"""

    mo_flag = True

    # geometry of molecules (in Angstrom)
    HF = 'H 0 0 0; F 0 0 1.1'

    H2O = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''

    atom = HF
    molecule = "HF"

    # setup model input using gaussian-type-orbitals
    molecular_HF = gto.M(
           atom=atom,  # in Angstrom
           # basis='ccpvdz',
           basis="6-31g",
           symmetry=1,
    )

    # run HF calculation
    mean_field = scf.HF(molecular_HF)
    mean_field.kernel()

    # extract parameter from the input Hamitonian and HF calculation
    h_core, fock_matrix, eri_integral, S_matrix = extract_Hamiltonian_parameters(mo_flag, mean_field, molecular_HF)

    # energy expectation value (HF energy)
    E_Hartree_Fock = mean_field.e_tot
    print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(E_Hartree_Fock))

    # total number of electron
    OccupationNumber = mean_field.mo_occ / 2
    nof_electron = sum(OccupationNumber)
    print("total number of electrons:{:}".format(nof_electron))
    print("occupation number:\n{:}".format(OccupationNumber))

    # get Nuclear Repusion Energy
    NR_energy = mean_field.energy_nuc()

    # run TFCC & thermal NOE calculation
    model = two_body_model(E_Hartree_Fock, h_core, fock_matrix, eri_integral, nof_electron, molecule=molecule,
                           E_NN=NR_energy, T_2_flag=True, chemical_potential=False, )
    # thermal field transform
    model.thermal_field_transform(T=1e8)
    # TFCC imaginary time integration
    model.TFCC_integration(T_final=2e3, N=10000, direct_flag=True, exchange_flag=True)
    # plot thermal properties
    model.Plot_thermal()

    return

if (__name__ == '__main__'):
    main()
