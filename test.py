# import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
from two_body_modeling import two_body_model

# setup model input
mol_hf = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = 1,
)

# run HF calculate
myhf = scf.HF(mol_hf)
myhf.kernel()

# extract Hamiltonian parameters
# 1-electron integral
hcore_ao = mol_hf.intor_symmetric('int1e_kin') + mol_hf.intor_symmetric('int1e_nuc')
hcore_mo = np.einsum('pi,pq,qj->ij', myhf.mo_coeff, hcore_ao, myhf.mo_coeff)
print("1-electron integral (in AO basis):\n{:}".format(hcore_ao.shape))
print("1-electron integral (in MO basis):\n{:}".format(hcore_mo.shape))

# 2-electron integral
eri_ao = mol_hf.intor('int2e_sph', aosym=1)
eri_mo = ao2mo.incore.full(eri_ao, myhf.mo_coeff)
print("2-electron integral (in AO basis):\n{:}".format(eri_ao.shape))
print("2-electron integral (in MO basis):\n{:}".format(eri_mo.shape))

# Overlap matrix
# S_ovlp_ao = myhf.get_ovlp()
# S_ovlp_mo = np.einsum('pi,pq,qj->ij', myhf.mo_coeff, S_ovlp_ao, myhf.mo_coeff)

# print("Overlap matrix (in AO basis):\n{:}".format(S_ovlp_ao.shape))
# print("Overlap matrix (in MO basis):\n{:}".format(S_ovlp_mo.shape))

# assert np.allclose(S_ovlp_mo, np.eye(fock_mo.shape[0]))

# energy expectation value (constant term)
E_0 = myhf.e_tot
print("energy expectation value (in Hartree):{:.5f}".format(E_0))

# total number of electrons
n_el = sum(myhf.mo_occ) / 2
print("total number of electrons: {:}".format(n_el))
print("occupation number:\n{:}".format(myhf.mo_occ))

# run TFCC & thermal NOE calculation
model = two_body_model(E_0, hcore_mo, eri_mo, n_el)
# thermal field transform
model.thermal_field_transform(T=100)
# TFCC imaginary time integration
model.thermal_field_coupled_cluster(T_final=1e6, N=10000, chemical_potential=True)
# plot thermal properties
model.Plot_thermal()
