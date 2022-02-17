 # import pyscf
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
from two_body_modeling import two_body_model
import itertools as it


def extract_Hamiltonian_parameters(mo_flag, CAS_SCF, mol_HF):
    """
    extract 1-electron integral, overlap matrix, 2-electron integral and fock matrix from  Hamiltonian parameters
    """
    def Cal_core_1_RDM(CI_coefficient, core_orbitals, active_orbitals):
        """calculate core 1-RDM"""
        size_core = len(core_orbitals)
        size_cas = len(active_orbitals)
        RDM_1_core = np.zeros([size_cas, size_cas])

        for alpha, orbital_label_alpha in enumerate(active_orbitals):
            for beta, orbital_label_beta in enumerate(active_orbitals):
                for i in range(size_core):
                    RDM_1_core[alpha, beta] += CI_coefficient[orbital_label_alpha, i] * CI_coefficient[orbital_label_beta, i]

        return RDM_1_core

    def Cal_one_electron_integral(h_core_AO, RDM_1_core, active_orbitals, eri_AO_CAS):
        """calculate one electron integral in model CAS space"""
        size_cas = len(active_orbitals)
        h_core_AO_CAS = np.zeros([size_cas, size_cas])

        # add core electron part
        for alpha, orbital_label_alpha in enumerate(active_orbitals):
            for beta, orbital_label_beta in enumerate(active_orbitals):
                h_core_AO_CAS[alpha, beta] += h_core_AO[orbital_label_alpha, orbital_label_beta]

        # add effective two electron part
        for a, b, c, d in it.product(range(size_cas), repeat=4):
            h_core_AO_CAS[a, b] += RDM_1_core[c, d] * (2 * eri_AO_CAS[a, b, c, d] - eri_AO_CAS[a, c, b, d])

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

    if not mo_flag:
        # construct model Hamiltonian active space ( in AO basis)
        ## calculate core 1-RDM
        RDM_1_core = Cal_core_1_RDM(CAS_SCF.mo_coeff, core_orbitals, active_orbitals)

        ## calculate two electron model Hamitonian in active space
        eri_AO_CAS = Cal_two_electron_integral(active_orbitals, eri_AO)

        ## calculate one electron model Hamitonian in active space
        h_core_AO_CAS = Cal_one_electron_integral(h_core_AO, RDM_1_core, active_orbitals, eri_AO_CAS)

        ## calculate ground state Fock matrix in active space
        Fock_AO_CAS = Cal_Fock_ground_state(active_orbitals, fock_AO)


        return h_core_AO_CAS, eri_AO_CAS, Fock_AO_CAS

    else:
        # construct model Hamiltonian in CAS in MO basis

         # Transfrom original integrals from AO basis to MO basis

         # calculate nature orbital in active space
         nature_orbital = CAS_SCF.mo_coeff
         # Fock matrix
         Fock_MO = np.einsum('pi,pq,qj->ij', nature_orbital, fock_AO, nature_orbital)
         print('Ground state Fock matrix(in MO basis):\n{:}'.format(Fock_MO.shape))

         # 1-electron integral
         h_core_MO = np.einsum('pi,pq,qj->ij', nature_orbital, h_core_AO, nature_orbital)
         print('1-electron integral (in MO basis):\n{:}'.format(h_core_MO.shape))

         # 2-electron integral
         eri_MO = ao2mo.incore.full(eri_AO, nature_orbital)
         print("2-electron integral (in MO basis):\n{:}".format(eri_MO.shape))

         # construct model Hamiltonian active space ( in AO basis)
         ## calculate core 1-RDM
         RDM_1_core = Cal_core_1_RDM(CAS_SCF.mo_coeff, core_orbitals, active_orbitals)

         ## calculate two electron model Hamitonian in active space
         eri_MO_CAS = Cal_two_electron_integral(active_orbitals, eri_MO)
         print('1-electron integral (in CAS MO basis):\n{:}'.format(eri_MO_CAS.shape))

         ## calculate one electron model Hamitonian in active space
         h_core_MO_CAS = Cal_one_electron_integral(h_core_AO, RDM_1_core, active_orbitals, eri_MO_CAS)
         print("2-electron integral (in CAS MO basis):\n{:}".format(h_core_MO_CAS.shape))

         ## calculate ground state Fock matrix in active space
         Fock_MO_CAS = Cal_Fock_ground_state(active_orbitals, Fock_MO)
         print('Ground state Fock matrix(in CAS MO basis):\n{:}'.format(Fock_MO_CAS.shape))

    return h_core_MO_CAS, eri_MO_CAS, Fock_MO_CAS


def main():
    """main run TFCC approach that conduct imaginary time integration on thermal properties for molecular compounds"""

    mo_flag = True

    # geometry of molecules (in Angstrom)
    HF = 'H 0 0 0; F 0 0 1.1'

    H2O = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''

    O2 = 'O 0 0 0; O 0 0 1.2'

    N2 = 'N 0 0 0; N 0 0 2.0'

    atom = N2
    molecule = "N2"

    # setup model input using gaussian-type-orbitals
    molecular_HF = gto.M(
           atom=atom,  # in Angstrom
           basis='ccpvdz',
           # basis="6-31g",
           symmetry=True,
           spin=0
    )

    # run HF calculation
    mean_field = scf.RHF(molecular_HF).run()

    mycas = mcscf.CASSCF(mean_field, 6, 6)
    mycas.natorb = True
    # Here mycas.mo_coeff are natural orbitals because .natorb is on.
    # Note The active space orbitals have the same symmetry as the input HF
    # canonical orbitals.  They are not fully sorted wrt the occpancies.
    # The mcscf active orbitals are sorted only within each irreps.
    mycas.kernel()

    # extract parameter from the input Hamitonian and HF calculation
    h_core, eri_integral, Fock_ground_state = extract_Hamiltonian_parameters(mo_flag, mycas, molecular_HF)

    # energy expectation value (HF energy)
    E_Hartree_Fock = mycas.e_tot
    print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(E_Hartree_Fock))

    # total number of electron
    OccupationNumber = mycas.mo_occ / 2
    nof_electron = 3
    print("total number of electrons:{:}".format(nof_electron))
    print("occupation number:\n{:}".format(OccupationNumber))

    # get Nuclear Repusion Energy
    NR_energy = mycas.energy_nuc()

    # run TFCC & thermal NOE calculation
    model = two_body_model(E_Hartree_Fock, h_core, Fock_ground_state, eri_integral, nof_electron, molecule=molecule,
                           E_NN=NR_energy, T_2_flag=True, chemical_potential=True, partial_trace_condition=False)
    # thermal field transform
    model.thermal_field_transform(T=1e8)
    # TFCC imaginary time integration
    model.TFCC_integration(T_final=3.5e4, N=10000, direct_flag=True, exchange_flag=True)
    # plot thermal properties
    model.Plot_thermal()

    return

if (__name__ == '__main__'):
    main()
