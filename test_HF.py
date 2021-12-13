# import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
from Hartree_Fock import check_integral

# MO
MO = False

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
hcoreAO = mol_hf.intor('int1e_kin_sph') + mol_hf.intor('int1e_nuc_sph')
hcoreMO = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcoreAO, mf.mo_coeff)
print("1-electron integral (in AO basis):\n{:}".format(hcoreAO.shape))
print("1-electron integral (in MO basis):\n{:}".format(hcoreMO.shape))

# overlap matrix
SAO = mol_hf.intor('int1e_ovlp_sph')
SMO = np.einsum('pi,pq,qj->ij', mf.mo_coeff, SAO, mf.mo_coeff)

# 2-electron integral
eriAO = mol_hf.intor('int2e_sph', aosym=1)
eriMO = ao2mo.incore.full(eriAO, mf.mo_coeff)
print("2-electron integral (in AO basis):\n{:}".format(eriAO.shape))
print("2-electron integral (in MO basis):\n{:}".format(eriAO.shape))

# Fock matrix
fockAO = mf.get_fock()
fockMO = np.einsum('pi,pq,qj->ij', mf.mo_coeff, fockAO, mf.mo_coeff)
print("Fock matrix (in AO basis):\n{:}".format(fockAO.shape))
print("Fock matrix (in MO basis):\n{:}".format(fockMO.shape))

# energy expectation value (HF energy)
EHartreeFock = mf.e_tot

print("energy expectation value (HF energy) (in Hartree):{:.5f}".format(EHartreeFock))

# total number of electrons
OccupationNumber = mf.mo_occ / 2
nelectron = sum(OccupationNumber)
print("total number of electrons: {:}".format(nelectron))
print("occupation number:\n{:}".format(OccupationNumber))

# get NuclearRepulsionEnergy
NRE = mf.energy_nuc()

if MO:
    hcore = hcoreMO
    fock = fockMO
    eri = eriMO
    S = SMO
else:
    hcore = hcoreAO
    fock = fockAO
    eri = eriAO
    S = SAO

# run TFCC & thermal NOE calculation
model = check_integral(EHartreeFock, hcore, fock, eri, nelectron, S, OccupationNumber, NRE, molecule=molecule, MO=MO)
# thermal field transform
model.my_SCF()
