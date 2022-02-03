#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = '6-31g')

mf = mol.RHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)
print('CCSD(T) ground state energy', mf.e_tot + mycc.e_corr + et)
mf = mol.UHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)
print('UCCSD(T) ground state energy', mf.e_tot + mycc.e_corr + et)
