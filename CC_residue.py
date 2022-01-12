import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.mp import mp2
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


# functions that calculate intermediate quantities

### Eqs. (37)-(39) "kappa"
def cc_Foo(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    foo = F_tilde['ij']
    Fki = foo.copy()
    if ERI_flag:
        eris_ovov = W_tilde['iajb']
        Fki += 2 * lib.einsum('kcld,ilcd->ki', eris_ovov, t2)
        Fki -= lib.einsum('kdlc,ilcd->ki', eris_ovov, t2)
        Fki += 2 * lib.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
        Fki -= lib.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    return Fki


def cc_Fvv(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    fvv = F_tilde['ab']
    Fac = fvv.copy()
    if ERI_flag:
        eris_ovov = W_tilde['iajb']
        Fac -= 2 * lib.einsum('kcld,klad->ac', eris_ovov, t2)
        Fac += lib.einsum('kdlc,klad->ac', eris_ovov, t2)
        Fac -= 2 * lib.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
        Fac += lib.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    return Fac


def cc_Fov(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    fov = F_tilde['ia']
    Fkc = fov.copy()
    if ERI_flag:
        eris_ovov = W_tilde['iajb']
        Fkc += 2 * np.einsum('kcld,ld->kc', eris_ovov, t1)
        Fkc -= np.einsum('kdlc,ld->kc', eris_ovov, t1)
    return Fkc

### Eqs. (40)-(41) "lambda"


def cc_Loo(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    fov = F_tilde['ia']
    Lki = cc_Foo(t1, t2, F_tilde, W_tilde, ERI_flag=ERI_flag) + np.einsum('kc,ic->ki', fov, t1)
    if ERI_flag:
        eris_ovoo = W_tilde['iajk']
        Lki += 2*np.einsum('lcki,lc->ki', eris_ovoo, t1)
        Lki -=   np.einsum('kcli,lc->ki', eris_ovoo, t1)
    return Lki


def cc_Lvv(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    fov = F_tilde['ia']
    Lac = cc_Fvv(t1, t2, F_tilde, W_tilde, ERI_flag=ERI_flag) - np.einsum('kc,ka->ac',fov, t1)
    if ERI_flag:
        eris_ovvv = W_tilde['iabc']
        Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
        Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"
def cc_Woooo(t1, t2, F_tilde, V_tilde):
    eris_ovoo = V_tilde['iajk']
    Wklij  = lib.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
    eris_ovov = V_tilde['iajb']
    Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += V_tilde['ijkl'].transpose(0,2,1,3)
    return Wklij


def cc_Wvvvv(t1, t2, F_tilde, V_tilde, flag=False):
    # Incore
    eris_ovvv = V_tilde['iabc']
    Wabcd  = lib.einsum('kdac,kb->abcd', eris_ovvv,-t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += V_tilde['abcd'].transpose(0,2,1,3)
    return Wabcd


def cc_Wvovo(t1, t2, F_tilde, V_tilde):
    eris_ovvv = V_tilde['iabc']
    eris_ovoo = V_tilde['iajk']
    Wakci  = lib.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += V_tilde['ijab'].transpose(2,0,3,1)
    eris_ovov = V_tilde['iajb']
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

def cc_Wvoov(t1, t2, F_tilde, V_tilde):
    eris_ovvv = V_tilde['iabc']
    eris_ovoo = V_tilde['iajk']
    Wakic  = lib.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += np.asarray(V_tilde['iabj']).transpose(2,0,3,1)
    eris_ovov = V_tilde['iajb']
    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def residue(t1, t2, F_tilde, V_tilde, W_tilde, cc2=False, ERI_flag=False):
    """Singles and Doubles residue equation"""
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    # assert(isinstance(eris, ccsd._ChemistsERIs))

    # notice that V_tilde is the original two electron integral and W_tilde is the spin-adapted two electron integrals

    fov = F_tilde['ia']
    foo = F_tilde['ij']
    fvv = F_tilde['ab']

    # mo_e_o = mo_energy[:nocc]
    # mo_e_v = mo_energy[nocc:] + cc.level_shift

    Foo = cc_Foo(t1, t2, F_tilde, W_tilde, ERI_flag=ERI_flag)
    Fvv = cc_Fvv(t1, t2, F_tilde, W_tilde, ERI_flag=ERI_flag)
    Fov = cc_Fov(t1, t2, F_tilde, W_tilde, ERI_flag=ERI_flag)

    M = t1.shape[0]
    # T1 equation
    t1new = np.zeros([M, M])
    t1new +=-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    if ERI_flag:
        t1new += 2*np.einsum('kcai,kc->ia', W_tilde['iabj'], t1)
        t1new +=  -np.einsum('kiac,kc->ia', W_tilde['ijab'], t1)
        t1new += 2*lib.einsum('kdac,ikcd->ia', W_tilde['iabc'], t2)
        t1new +=  -lib.einsum('kcad,ikcd->ia', W_tilde['iabc'], t2)
        t1new += 2*lib.einsum('kdac,kd,ic->ia', W_tilde['iabc'], t1, t1)
        t1new +=  -lib.einsum('kcad,kd,ic->ia', W_tilde['iabc'], t1, t1)
        t1new +=-2*lib.einsum('lcki,klac->ia', W_tilde['iajk'], t2)
        t1new +=   lib.einsum('kcli,klac->ia', W_tilde['iajk'], t2)
        t1new +=-2*lib.einsum('lcki,lc,ka->ia', W_tilde['iajk'], t1, t1)
        t1new +=   lib.einsum('kcli,lc,ka->ia', W_tilde['iajk'], t1, t1)

    # T2 equation
    t2new = np.zeros([M, M, M, M])

    # print("shape of V:{:}".format(V_tilde['aibj'].shape))
    # print("shape of t2new:{:}".format(t2new.shape))

    if ERI_flag:
        tmp2 = lib.einsum('kibc,ka->abic', V_tilde['ijab'], -t1)
        tmp2 += V_tilde['iabc'].conj().transpose(1,3,0,2)
        tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
        t2new = tmp + tmp.transpose(1,0,3,2)
        tmp2  = lib.einsum('kcai,jc->akij', V_tilde['iajb'], t1)
        tmp2 += V_tilde['iajk'].transpose(1,3,0,2).conj()
        tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
        t2new -= tmp + tmp.transpose(1,0,3,2)
        t2new += V_tilde['iajb'].conj().transpose(0,2,1,3)
    if cc2:
        if ERI_flag:
            Woooo2 = V_tilde['ijkl'].transpose(0,2,1,3).copy()
            Woooo2 += lib.einsum('lcki,jc->klij', V_tilde['iajk'], t1)
            Woooo2 += lib.einsum('kclj,ic->klij', V_tilde['iajk'], t1)
            Woooo2 += lib.einsum('kcld,ic,jd->klij', V_tilde['iajb'], t1, t1)
            t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
            Wvvvv = lib.einsum('kcbd,ka->abcd', V_tilde['iabc'], -t1)
            Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
            Wvvvv += V_tilde['abcd'].transpose(0,2,1,3)
            t2new += lib.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = lib.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = cc_Loo(t1, t2, F_tilde, V_tilde, ERI_flag=ERI_flag)
        Lvv = cc_Lvv(t1, t2, F_tilde, V_tilde, ERI_flag=ERI_flag)
        # Loo[np.diag_indices(nocc)] -= mo_e_o
        # Lvv[np.diag_indices(nvir)] -= mo_e_v
        if ERI_flag:
            Woooo = cc_Woooo(t1, t2, F_tilde, V_tilde)
            Wvoov = cc_Wvoov(t1, t2, F_tilde, V_tilde)
            Wvovo = cc_Wvovo(t1, t2, F_tilde, V_tilde)
            Wvvvv = cc_Wvvvv(t1, t2, F_tilde, V_tilde)

            tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
            t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
            t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        if ERI_flag:
            tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
            tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
            t2new += (tmp + tmp.transpose(1,0,3,2))
            tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
            t2new -= (tmp + tmp.transpose(1,0,3,2))
            tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
            t2new -= (tmp + tmp.transpose(1,0,3,2))

    return t1new, t2new


def energy(t1, t2, F_tilde, W_tilde, ERI_flag=False):
    '''RCCSD correlation energy'''
    # if t1 is None: t1 = cc.t1
    # if t2 is None: t2 = cc.t2
    # if eris is None: eris = cc.ao2mo()
    e = 2*np.einsum('ia,ia->', F_tilde['ia'], t1)
    if ERI_flag:
        tau = np.einsum('ia,jb->ijab',t1,t1)
        tau += t2
        eris_ovov = W_tilde['iajb'].copy()
        e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
        e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real
