import numpy as np
from scipy.linalg import eigh

lapack_driver='evd'

# function definitions.

def cartdist(x1, x2, xmax):
    """cartesian distance on 1d periodic line"""
    dx = np.abs(x1 - x2)
    dx = np.where(dx > 0.5 * xmax, xmax - dx, dx)
    return np.sqrt(dx ** 2)

def gaspcohn(r):
    """
    Gaspari-Cohn taper function.
    very close to exp(-(r/c)**2), where c = sqrt(0.15)
    r should be >0 and normalized so taper = 0 at r = 1
    """
    rr = 2.0 * r
    rr += 1.0e-13  # avoid divide by zero warnings from numpy
    taper = np.where(
        r <= 0.5,
        (((-0.25 * rr + 0.5) * rr + 0.625) * rr - 5.0 / 3.0) * rr ** 2 + 1.0,
        np.zeros(r.shape, r.dtype),
    )
    taper = np.where(
        np.logical_and(r > 0.5, r < 1.0),
        ((((rr / 12.0 - 0.5) * rr + 0.625) * rr + 5.0 / 3.0) * rr - 5.0) * rr
        + 4.0
        - 2.0 / (3.0 * rr),
        taper,
    )
    return taper

# ensemble modulator
def modens(enspert, sqrtcovlocal):
    neig = sqrtcovlocal.shape[0]
    nanals = enspert.shape[0]
    enspert2 = np.empty((neig*nanals,)+enspert.shape[1:],enspert.dtype)
    nanal2 = 0
    for j in range(neig):
        for nanal in range(nanals):
            enspert2[nanal2,...] =\
            enspert[nanal,...]*sqrtcovlocal[j,np.newaxis,...]
#           enspert[nanal,...]*sqrtcovlocal[neig-j-1,np.newaxis,...]
            nanal2 += 1
    return enspert2

def get_nanal_index(nanals, neig):
    nanal_index=np.empty(neig*nanals, np.int32)
    nanal2 = 0
    for j in range(neig):
        for nanal in range(nanals):
            nanal_index[nanal2]=nanal
            nanal2 += 1
    return nanal_index

def lgetkf(xens, hxens, obs, oberrs, covlocal, nerger=True, ngroups=None):

    """returns ensemble updated by LGETKF with cross-validation and single-scale R localization"""

    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    nanals = hxens.shape[0]
    ndim = covlocal.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime_b = xprime.copy()
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(hx, Rlocal, oberrvar, nerger=True):
        normfact = np.array(np.sqrt(hx.shape[0]-1),dtype=np.float64)
        if nerger:
            # Nerger regularization
            hpbht = (hx**2).sum(axis=0)/normfact**2
            Rinvsqrt = np.sqrt(Rlocal/(hpbht*(1.-Rlocal)+oberrvar))
            YbRinv = hx*Rinvsqrt**2/normfact
            YbsqrtRinv = hx*Rinvsqrt/normfact
        else:
            YbsqrtRinv = hx*np.sqrt(Rlocal/oberrvar)/normfact
            YbRinv = hx*(Rlocal/oberrvar)/normfact
        return YbsqrtRinv, YbRinv

    def calcwts_mean(ndgf, hx, Rlocal, oberrvar, ominusf, nerger=True):
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float64)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(ndgf, hx_orig, hx, Rlocal, oberrvar,nerger=True):
        # hx_orig contains the ensemble pert for the witheld member
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float64)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    for n in range(ndim):
        mask = covlocal[:,n] > 1.0e-10
        nobs_local = mask.sum()
        if nobs_local > 0:
            Rlocal = covlocal[mask, n]
            oberrvar_local = oberrs[mask]
            ominusf_local = (obs-hxmean)[mask]
            hxprime_local = hxprime[:,mask]
            wts_ensmean = calcwts_mean(nanals-1, hxprime_local, Rlocal, oberrvar_local, ominusf_local, nerger=nerger)
            xmean[n] += np.dot(wts_ensmean,xprime_b[:,n])
            # update sub-ensemble groups, using cross validation.
            for ngrp in range(ngroups):
                nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
                hxprime_cv = np.delete(hxprime_local,nanal_cv,axis=0); xprime_cv = np.delete(xprime_b[:,n],nanal_cv,axis=0)
                hxprime_cv_mean = hxprime_cv.mean(axis=0); xprime_cv_mean = xprime_cv.mean(axis=0)
                hxprime_cv -= hxprime_cv_mean; xprime_cv -= xprime_cv_mean
                wts_ensperts_cv = calcwts_perts((nanals-nanals//ngroups)-1, hxprime_local[nanal_cv], hxprime_cv, Rlocal, oberrvar_local, nerger=nerger)
                xprime[nanal_cv,n] += np.dot(wts_ensperts_cv,xprime_cv)
            xprime_mean = xprime[:,n].mean(axis=0) 
            xprime[:,n] -= xprime_mean # ensure zero mean
            xens[:,n] = xmean[n]+xprime[:,n]

    return xens

def lgetkf_ms(nlscales, xens, xprime, hxprime, hxprime_orig, omf, oberrs, covlocal, ngroups=None):

    """returns ensemble updated by LGETKF with cross-validation and multi-scale R localization"""

    nanals = hxprime_orig.shape[0]
    ndim = covlocal.shape[-1]
    xmean = xens.mean(axis=0)
    xprime_orig = xens - xmean
    nanal_index = np.empty(nanals*nlscales)
    nanal2 = 0
    for nl in range(nlscales):
        for nanal in range(nanals):
            nanal_index[nanal2]=nanal
            nanal2 += 1
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(ndgf,nlscales,hx,Rlocal,oberrvar):
        nanalstot, nobs = hx.shape
        nanals_orig = nanalstot//nlscales
        normfact = np.array(np.sqrt(ndgf),dtype=np.float64)
        YbsqrtRinv = np.empty((nanalstot,nobs),np.float64)
        YbRinv = np.empty((nanalstot,nobs),np.float64)
        hpbht = np.empty((nlscales,nobs),np.float64)
        Rinvsqrt_nerger = np.empty_like(hpbht)
        rij = np.empty(nobs,np.float64)
        for nl in range(nlscales):
            nanal1=nl*nanals_orig; nanal2=(nl+1)*nanals_orig
            hpbht[nl] = (hx[nanal1:nanal2]**2).sum(axis=0)/normfact**2
            if nl == 0:
               rij = hpbht[0]
            else:
               rij += Rlocal[nl]*hpbht[nl]
        hpbht_tot = hpbht.sum(axis=0)
        #hpbht_tot = ((hx/normfact)**2).sum(axis=0) # same as above for heaviside cutoff
        rij = rij/hpbht_tot
        for nl in range(nlscales):
            # CB's original version (no rij factor)
            Rinvsqrt_nerger[nl] = np.sqrt(Rlocal[nl]/(hpbht_tot*(1.-Rlocal[nl])+oberrvar))
            # Bo's suggested modification
            #Rinvsqrt_nerger[nl] = np.sqrt(Rlocal[nl]/(hpbht[nl]*(1.-Rlocal[nl])+oberrvar))
            # CB's adjusted version
            #Rinvsqrt_nerger[nl] = np.sqrt(Rlocal[nl]/(rij*(hpbht_tot*(1.-Rlocal[nl])+oberrvar)))
        Rdsqrt = (Rinvsqrt_nerger*hpbht).sum(axis=0)/hpbht_tot
        for nl in range(nlscales):
            nanal1=nl*nanals_orig; nanal2=(nl+1)*nanals_orig
            YbRinv[nanal1:nanal2] = hx[nanal1:nanal2]*Rinvsqrt_nerger[nl]*Rdsqrt/normfact
            YbsqrtRinv[nanal1:nanal2] = hx[nanal1:nanal2]*Rinvsqrt_nerger[nl]/normfact
        return YbsqrtRinv, YbRinv

    def calcwts_mean(ndgf, nlscales, hx, oberrvar, Rlocal, ominusf):
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float64)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,nlscales,hx,Rlocal,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        gamma_inv = 1./evals
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(ndgf, nlscales, hx_orig, hx, oberrvar, Rlocal):
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float64)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,nlscales,hx,Rlocal,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs, = eigh(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs, = eigh(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.; gamma_inv = 1./evals
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        pa=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        return -np.dot(pa, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    for n in range(ndim):
        mask = covlocal[0,:,n] > 1.0e-10
        nobs_local = mask.sum()
        if nobs_local > 0:
            oberrvar_local = oberrs[mask]
            Rlocal = np.empty((nlscales,nobs_local),np.float64)
            for nl in range(nlscales):
                Rlocal[nl] = covlocal[nl,mask,n].clip(min=np.finfo(np.float64).eps)
            ominusf_local = omf[mask]
            hxprime_local = hxprime[:,mask]
            hxprime_orig_local = hxprime_orig[:,mask]
            wts_ensmean = calcwts_mean(nanals-1, nlscales, hxprime_local, oberrvar_local, Rlocal, ominusf_local)
            xmean[n] += np.dot(wts_ensmean,xprime[:,n])
            # update one member at a time (one member for each scale), using cross validation.
            for ngrp in range(ngroups):
                nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
                nanals_sub = np.nonzero(np.isin(nanal_index,nanal_cv))
                hxprime_cv = np.delete(hxprime_local,nanals_sub,axis=0)
                xprime_cv = np.delete(xprime[:,n],nanals_sub,axis=0)
                hxprime_cv_mean = hxprime_cv.mean(axis=0); xprime_cv_mean = xprime_cv.mean(axis=0)
                hxprime_cv -= hxprime_cv_mean; xprime_cv -= xprime_cv_mean
                wts_ensperts_cv = calcwts_perts((nanals-nanals//ngroups)-1, nlscales, hxprime_orig_local[nanal_cv], hxprime_cv, oberrvar_local, Rlocal)
                xprime_orig[nanal_cv,n] += np.dot(wts_ensperts_cv,xprime_cv)
            xprime_mean = xprime_orig[:,n].mean(axis=0) 
            xprime_orig[:,n] -= xprime_mean # ensure zero mean
            xens[:,n] = xmean[n]+xprime_orig[:,n]

    return xens

def getkf_bloc(xens, ominusf, oberrvar, sqrtcovlocal, indxob, ngroups=None):

    """returns ensemble updated by GETKF with cross-validation and model-space localization"""

    nanals = xens.shape[0]
    ndim = xens.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime_b = xprime.copy()
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(ndgf, hx, oberrvar):
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        nens = hx.shape[0]
        YbsqrtRinv = (hx/normfact)*np.sqrt(1./oberrvar)
        YbRinv = (hx/normfact)*(1./oberrvar)
        return YbsqrtRinv, YbRinv

    def calcwts_mean(ndgf, hx, oberrvar, ominusf):
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf, hx,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(ndgf, hx_orig, hx, oberrvar):
        # hx_orig contains the ensemble for the witheld member
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,hx,oberrvar)
        if nobs >= ndgf:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    xprime2 = modens(xprime_b,sqrtcovlocal)
    nanals2 = xprime2.shape[0]
    nanal_index = get_nanal_index(nanals, nanals2//nanals)
    nobs = len(indxob)
    hxprime = np.empty((nanals,nobs),np.float32)
    hxprime2 = np.empty((nanals2,nobs),np.float32)
    for nanal in range(nanals):
        hxprime[nanal] = xprime[nanal,indxob]
    for nanal in range(nanals2):
        hxprime2[nanal] = xprime2[nanal,indxob]
    wts_ensmean = calcwts_mean(nanals-1, hxprime2, oberrvar, ominusf)
    xmean += np.dot(wts_ensmean,xprime2)
    # update sub-ensemble groups, using cross validation.
    for ngrp in range(ngroups):
        nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
        nanals_sub = np.nonzero(np.isin(nanal_index,nanal_cv))
        hxprime_cv = np.delete(hxprime2,nanals_sub,axis=0)
        xprime_cv = np.delete(xprime2,nanals_sub,axis=0)
        hxprime_cv_mean = hxprime_cv.mean(axis=0); xprime_cv_mean = xprime_cv.mean(axis=0)
        hxprime_cv -= hxprime_cv_mean; xprime_cv -= xprime_cv_mean
        wts_ensperts_cv = calcwts_perts((nanals-nanals//ngroups)-1, hxprime[nanal_cv], hxprime_cv, oberrvar)
        xprime[nanal_cv] += np.dot(wts_ensperts_cv,xprime_cv)
        xprime_mean = xprime.mean(axis=0) 
        xprime -= xprime_mean # ensure zero mean
    return xmean+xprime

def getkfms_bloc(xens, xprime, ominusf, oberrvar, sqrtcovlocal, indxob, ngroups=None):

    """returns ensemble updated by GETKF with cross-validation and multi-scale model-space localization"""

    # xens is original ensemble, xprime has scale-decomposed ensemble perturbations
    nanals = xprime.shape[0]
    nlscales = xprime.shape[1]
    nx = xprime.shape[-1]
    xmean = xens.mean(axis=0)
    xprime_orig = xens - xmean
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(ndgf, hx, oberrvar):
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        nens = hx.shape[0]
        YbsqrtRinv = (hx/normfact)*np.sqrt(1./oberrvar)
        YbRinv = (hx/normfact)*(1./oberrvar)
        return YbsqrtRinv, YbRinv

    def calcwts_mean(ndgf, hx, oberrvar, ominusf):
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,hx,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, = eigh(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, = eigh(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(ndgf, hx_orig, hx, oberrvar):
        # hx_orig contains the ensemble for the witheld member
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,hx,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    neig = sqrtcovlocal.shape[0]
    nanals2 = nanals*neig
    xprime2 = (modens(xprime.reshape((nanals,nlscales*nx)),sqrtcovlocal)).reshape((nanals2,nlscales,nx))
    xprime2 = xprime2.sum(axis=1) # sum over wavebands after modulation
    xprime = xprime.sum(axis=1) # sum over wavebands
    nanal_index = get_nanal_index(nanals, neig)
    nobs = len(indxob)
    hxprime = np.empty((nanals,nobs),np.float32)
    hxprime2 = np.empty((nanals2,nobs),np.float32)
    for nanal in range(nanals):
        hxprime[nanal] = xprime[nanal,indxob]
    for nanal in range(nanals2):
        hxprime2[nanal] = xprime2[nanal,indxob]
    wts_ensmean = calcwts_mean(nanals-1, hxprime2, oberrvar, ominusf)
    xmean += np.dot(wts_ensmean,xprime2)
    # update sub-ensemble groups, using cross validation.
    for ngrp in range(ngroups):
        nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
        nanals_sub = np.nonzero(np.isin(nanal_index,nanal_cv))
        hxprime_cv = np.delete(hxprime2,nanals_sub,axis=0)
        xprime_cv = np.delete(xprime2,nanals_sub,axis=0)
        hxprime_cv_mean = hxprime_cv.mean(axis=0); xprime_cv_mean = xprime_cv.mean(axis=0)
        hxprime_cv -= hxprime_cv_mean; xprime_cv -= xprime_cv_mean
        wts_ensperts_cv = calcwts_perts((nanals-nanals//ngroups)-1, hxprime[nanal_cv], hxprime_cv, oberrvar)
        xprime_orig[nanal_cv] += np.dot(wts_ensperts_cv,xprime_cv)
    xprime_mean = xprime_orig.mean(axis=0) 
    xprime_orig -= xprime_mean # ensure zero mean
    return xmean+xprime_orig

def lgetkfms_bloc(xens, xprime, omf, oberrs, sqrtcovlocal_local, covlocal_ob, indxob, covlocal_model, ngroups=None):

    """returns ensemble updated by LGETKF with cross-validation and multi-scale model-space localization"""

    # xens is original ensemble, xprime has scale-decomposed ensemble perturbations
    nanals = xprime.shape[0]
    nlscales = xprime.shape[1]
    ndim = xprime.shape[-1]
    xmean = xens.mean(axis=0)
    xprime_orig = xens - xmean
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(ndgf, hx, oberrvar):
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        nens = hx.shape[0]
        YbsqrtRinv = (hx/normfact)*np.sqrt(1./oberrvar)
        YbRinv = (hx/normfact)*(1./oberrvar)
        return YbsqrtRinv, YbRinv

    def calcwts_mean(ndgf, hx, oberrvar, ominusf):
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,hx,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(ndgf, hx_orig, hx, oberrvar):
        # hx_orig contains the ensemble for the witheld member
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(ndgf),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(ndgf,hx,oberrvar)
        if nobs >= hx.shape[0]:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs = eigh(a,driver=lapack_driver)
            #evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    nc = 0
    for n in range(ndim):
        mask = covlocal_ob[:,n] > np.finfo(covlocal_ob.dtype).eps
        mask_local = covlocal_model[:,n] > np.finfo(covlocal_model.dtype).eps
        # indices of model grid points in local volume on global grid
        indx_local = np.nonzero(mask_local)[0]
        # global indices of observations in local volume
        indxob_local = np.nonzero(mask)[0]
        # indices of observation grid points in local volume on local grid
        indxob_local_local = np.isin(indx_local,indxob[indxob_local])
        nobs_local = len(indxob_local); npts_local = len(indx_local)
        if nobs_local != indxob_local_local.sum():
            raise ValueError('not all obs in local volume accounted for')
        xprime_local = xprime[:,:,indx_local]
        neig = sqrtcovlocal_local[nc].shape[0]
        nanals2 = nanals*neig
        xprime2_local = (modens(xprime_local.reshape((nanals,nlscales*npts_local)),sqrtcovlocal_local[nc])).reshape((nanals2,nlscales,npts_local))
        xprime2_local = xprime2_local.sum(axis=1) # sum over wavebands after modulation
        xprime_local = xprime_local.sum(axis=1) # sum over wavebands
        nmindist = np.argmax(covlocal_model[indx_local,n])
        nanal_index = get_nanal_index(nanals, neig)
        if nobs_local > 0:
            hxprime_local = np.empty((nanals,nobs_local),np.float32)
            hxprime2_local = np.empty((nanals2,nobs_local),np.float32)
            for nanal in range(nanals):
                hxprime_local[nanal] = xprime_local[nanal,indxob_local_local]
            for nanal in range(nanals2):
                hxprime2_local[nanal] = xprime2_local[nanal,indxob_local_local]
            oberrvar_local = oberrs[mask]
            ominusf_local = omf[mask]
            wts_ensmean = calcwts_mean(nanals-1, hxprime2_local, oberrvar_local, ominusf_local)
            xmean[n] += np.dot(wts_ensmean,xprime2_local[:,nmindist])
            # update sub-ensemble groups, using cross validation.
            for ngrp in range(ngroups):
                nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
                nanals_sub = np.nonzero(np.isin(nanal_index,nanal_cv))
                hxprime_cv = np.delete(hxprime2_local,nanals_sub,axis=0)
                xprime_cv = np.delete(xprime2_local[:,nmindist],nanals_sub,axis=0)
                hxprime_cv_mean = hxprime_cv.mean(axis=0); xprime_cv_mean = xprime_cv.mean(axis=0)
                hxprime_cv -= hxprime_cv_mean; xprime_cv -= xprime_cv_mean
                wts_ensperts_cv = calcwts_perts((nanals-nanals//ngroups)-1, hxprime_local[nanal_cv], hxprime_cv, oberrvar_local)
                xprime_orig[nanal_cv,n] += np.dot(wts_ensperts_cv,xprime_cv)
            xprime_mean = xprime_orig[:,n].mean(axis=0) 
            xprime_orig[:,n] -= xprime_mean # ensure zero mean
        nc += 1

    return xmean+xprime_orig
