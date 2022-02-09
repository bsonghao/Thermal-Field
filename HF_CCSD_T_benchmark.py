#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation on model magnetic system.
'''
from pyscf import gto, scf, ao2mo, cc
import numpy as np

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

def main():
    """main function that run CCSD / CCSD(T) ground state calculation"""
    # enter model parameters
    V_couple = 10
    V_ontop = 10
    one_body_hamiltonian = np.diag(np.array([-50., -30., -10., 10., 30., 50.]))
    nelectron = 6

    basis_size = one_body_hamiltonian.shape[0]
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
    h_core = one_body_hamiltonian

    # conduct Hartree-Fock Calculation on the model Hamiltonian
    mean_field = scf.RHF(mol)
    mean_field.get_hcore = lambda *args: h_core
    mean_field.get_ovlp = lambda *args: S
    mean_field._eri = ao2mo.restore(8, eri, basis_size)
    mean_field.kernel()

    # CCSD calcuation
    mycc = cc.RCCSD(mean_field)
    mycc.kernel()


    print('HF ground state energy', mean_field.e_tot)
    print('CCSD ground state energy', mean_field.e_tot + mycc.e_corr)

    return

if (__name__ == '__main__'):
    main()
