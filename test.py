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
mf = scf.HF(mol_hf)
mf.kernel()

# extract Hamiltonian parameters
# 1-electron integral
hcore_ao = mol_hf.intor_symmetric('int1e_kin') + mol_hf.intor_symmetric('int1e_nuc')
hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcore_ao, mf.mo_coeff)
print("1-electron integral (in AO basis):\n{:}".format(hcore_ao.shape))
print("1-electron integral (in MO basis):\n{:}".format(hcore_mo.shape))

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
n_el = sum(mf.mo_occ) / 2
print("total number of electrons: {:}".format(n_el))
print("occupation number:\n{:}".format(mf.mo_occ))

E, V = np.linalg.eigh(fock_mo)
RDM_1 = np.dot(V, np.dot(np.diag(mf.mo_occ), V.transpose()))

# run TFCC & thermal NOE calculation
model = two_body_model(E_HF, fock_mo, eri_mo, n_el)
# thermal field transform
model.thermal_field_transform(T=3e5)
# TFCC imaginary time integration
model.thermal_field_coupled_cluster(T_final=5e4, N=10000, chemical_potential=True)
# plot thermal properties
model.Plot_thermal()
