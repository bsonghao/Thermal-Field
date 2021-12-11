# import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
from Hartree_Fock import check_integral

#geometry of molecules (in Angstrom)
HF = 'H 0 0 0; F 0 0 1.1'

H2O = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''

atom = HF
molecule = "HF"
# setup model input
mol_hf = gto.M(
    atom = atom,  # in Angstrom
    basis = 'ccpvdz',
    symmetry = 1,
)

# run HF calculate
mf = scf.HF(mol_hf)
mf.kernel()

# extract Hamiltonian parameters
# 1-electron integral
hcore_ao = mol_hf.intor('int1e_kin_sph') + mol_hf.intor('int1e_nuc_sph')
hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcore_ao, mf.mo_coeff)
print("1-electron integral (in AO basis):\n{:}".format(hcore_ao.shape))
print("1-electron integral (in MO basis):\n{:}".format(hcore_mo.shape))

# overlap matrix
S_ao = mol_hf.intor('int1e_ovlp_sph')
S_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, S_ao, mf.mo_coeff)

# 2-electron integral
eri_ao = mol_hf.intor('int2e_sph', aosym=1)
eri_mo = ao2mo.incore.full(eri_ao, mf.mo_coeff)
print("2-electron integral (in AO basis):\n{:}".format(eri_ao.shape))
print("2-electron integral (in MO basis):\n{:}".format(eri_mo.shape))

# Fock matrix
fock_ao = mf.get_fock()
fock_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, fock_ao, mf.mo_coeff)
print("Fock matrix (in AO basis):\n{:}".format(fock_ao.shape))
print("Fock matrix (in MO basis):\n{:}".format(fock_mo.shape))

# energy expectation value (HF energy)
E_HF = mf.e_tot

print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(E_HF))

# total number of electrons
occupation_number = mf.mo_occ / 2
n_el = sum(occupation_number)
print("total number of electrons: {:}".format(n_el))
print("occupation number:\n{:}".format(occupation_number))

# get NuclearRepulsionEnergy
NRE = mf.energy_nuc()

# run TFCC & thermal NOE calculation
model = check_integral(E_HF, hcore_ao, fock_ao, eri_ao, n_el, S_ao, occupation_number, NRE, molecule=molecule)
# thermal field transform
model.my_SCF()
