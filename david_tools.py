import numpy as np
import fnmatch
import os
from field import Field
import itertools

def computephi(x,y):
    # return np.atan2(x,y)
    if x>0:
        if y>=0:
            phi = np.atan(y/x)
        elif y<0:
            phi = np.atan(y/x) + 2*np.pi
    elif x<0:
        phi = phi = np.atan(y/x) + np.pi
    elif x==0:
        if y>0:
            phi = np.pi/2
        elif y<0:
            phi = 3*np.pi/2
        elif y==0:
            return 0
    return phi%(2*np.pi)
    
def compute_rh(q,a):
    '''
    return hill radius
    '''
    return q/3**(1/3)*a+1e-15

def mask_rh(R,PHI,xp,yp,rh):
    '''
    La grille est donnée en r/phi alors que la position de la planète en x/y
    On veux filtrer seulement les couples (r,phi) hors du rayon de hill, i.e
    |\vec r - \vec r_p| > r_h > D'où passage en cartésien.

    R et PHI doivent être passées au meshgrid avant
    '''
    X=R*np.cos(PHI)
    Y=R*np.sin(PHI)
    return ((X-xp)**2 + (Y-yp) > rh)

def extract_nb_outputs(par, directory):
    if par.fargo3d == 'No':
        nb_outputs = len(fnmatch.filter(os.listdir(
            directory), 'gasdens*.dat'))  # ! fnmatch ?
    else:
        raise NotImplementedError(
            "fargo3d not implemented for torque density")
    print('number of outputs for directory ',
            directory, ': ', nb_outputs)
    return nb_outputs

def get_pla(dens,directory,torque):
    '''
    return xpla, ypla, mpla, date, omega
    '''
    if dens.fargo3d == 'Yes':
        f1, xpla, ypla, f4, f5, f6, f7, mpla, date, omega = np.loadtxt(
            directory+"/planet0.dat", unpack=True, max_rows=torque.nb_outputs)
    else:
        f1, xpla, ypla, f4, f5, mpla, f7, date, omega, f10, f11 = np.loadtxt(
            directory+"/planet0.dat", unpack=True, max_rows=torque.nb_outputs)
    return xpla, ypla, mpla, date, omega

def get_apla(directory):
    with open(directory+"/orbit0.dat") as f_in:
        firstline_orbitfile = np.genfromtxt(
            itertools.islice(f_in, 0, 1, None), dtype=float)
    apla = firstline_orbitfile[2]
    return apla

def extract_masstaper(par, directory):
    import subprocess
    if par.fargo3d == 'Yes':
        command   = par.awk_command+' " /^MASSTAPER/ " '+directory+'/variables.par'
    else:
        command   = par.awk_command+' " /^MassTaper/ " '+directory+'/*.par'
    buf = subprocess.getoutput(command)
    masstaper = float(buf.split()[1]) 
    return masstaper

def compute_epsilon(eta, ar, xpla, ypla, fli):
    return eta*ar*np.sqrt(xpla**2 + ypla**2)**(1+fli)

def get_dens(on,k,par,directory):
    dens = Field(field='dens', fluid='gas', on=on[k], directory=directory, physical_units=par.physical_units,
                nodiff=par.nodiff, fieldofview=par.fieldofview, onedprofile='No', override_units=par.override_units)
    return dens

def get_vphi(on,k,par,directory):
    vtheta = Field(field='vtheta', fluid='gas', on=on[k], directory=directory, physical_units=par.physical_units,
                nodiff=par.nodiff, fieldofview=par.fieldofview, onedprofile='No', override_units=par.override_units)
    return vtheta

def integrate_acc(acc_cell, dens, surface, dr, dphi, rdphi, vecpospla, mask=None):
    if mask is not None:
        acc_cell = acc_cell[mask]
    acc_lin = acc_cell* rdphi # intégrande * dphi
    acc_az = acc_cell * dens.rmed[:,np.newaxis] * dr[:,np.newaxis]
    acc_radial_density = np.sum(acc_lin, axis=2) # intégrale sur phi
    acc_az_density = np.sum(acc_az, axis=1) # intégrale sur r
    acc_tot = np.sum(acc_cell*surface, axis=(1,2))# intégrale sur le disque

    return acc_lin,acc_az,acc_radial_density,acc_az_density,acc_tot

def integrate_torque(acc_cell, dens, surface, dr, dphi, rdphi, vecpospla, mask=None):
    acc_lin,acc_az,acc_radial_density,acc_az_density,acc_tot = integrate_acc(acc_cell, dens, surface, dr, dphi, rdphi, vecpospla, mask)

    torque_tot = np.cross(vecpospla, acc_tot)[2]
    torque_radial_density = np.cross(vecpospla[:,np.newaxis], acc_radial_density, axisa=0, axisb=0)[:,2]
    torque_az_density = np.cross(vecpospla[:,np.newaxis], acc_az_density, axisa=0, axisb=0)[:,2]
    torque_radial_cumu = np.cumsum(torque_radial_density*dr)
    torque_az_cumu = np.cumsum(torque_az_density*dphi)

    return torque_tot,torque_radial_density,torque_az_density,torque_radial_cumu,torque_az_cumu

def torque_of_each_cell(acc_cell,surface, vecpospla):
    return np.cross(vecpospla[:,np.newaxis,np.newaxis], acc_cell*surface, axisa=0, axisb=0)[:,:,2]

def integrate_torque_gap(acc_cell, dens, dr,iinf_gap,isup_gap, vecpospla):
    '''returns azimuthal torque density with radial integration over the gap only
    xs must be xs = 1.16*np.sqrt(q/h)
    '''
    acc_az_gap = acc_cell[:,iinf_gap:isup_gap,:] * dens.rmed[iinf_gap:isup_gap,np.newaxis] * dr[iinf_gap:isup_gap,np.newaxis]
    acc_az_density = np.sum(acc_az_gap, axis=1) # intégrale sur r
    torque_az_density_gap = np.cross(vecpospla[:,np.newaxis], acc_az_density, axisa=0, axisb=0)[:,2]
    return torque_az_density_gap

def get_grid(dens):
    surface = np.zeros((dens.nrad, dens.nsec)) 
    Rinf = dens.redge[0:len(dens.redge)-1]
    Rsup = dens.redge[1:len(dens.redge)]
    surf = np.pi * (Rsup*Rsup - Rinf*Rinf) / dens.nsec
    for th in range(dens.nsec):
        surface[:, th] = surf

    dr = Rsup - Rinf
    dphi = dens.pedge[1:] - dens.pedge[0:-1]
    rdphi = np.dot(dens.rmed[:,np.newaxis], dphi[np.newaxis,:])

    return surface, dr, dphi, rdphi

def get_vecpos_cells(dens):
    cosphi = np.reshape(np.cos(dens.pmed), (1, dens.nsec))
    sinphi = np.reshape(np.sin(dens.pmed), (1, dens.nsec))
    x = np.dot(dens.rmed[:, np.newaxis], cosphi) # Produit scalaire d'une colonne et d'une ligne.
    y = np.dot(dens.rmed[:, np.newaxis], sinphi)
    z = np.zeros((dens.nrad, dens.nsec))
    return np.array([x, y, z])

def compute_Gamma0(Mpla, Xpla, Ypla, directory):
    import subprocess
    import par
    q = Mpla[len(Mpla)-1]   # time-varying array
    if q==0:
        return 1
    
    summary0_file = directory+'/summary0.dat'
    if os.path.isfile(summary0_file) == True:
        fargo3d = 'Yes'
    else:
        fargo3d = 'No'
    # get planet's orbital radius, local disc's aspect ratio + check if energy equation was used
    if fargo3d == 'Yes':
        command  = par.awk_command+' " /^ASPECTRATIO/ " '+directory+'/*.par'
        command2 = par.awk_command+' " /^FLARINGINDEX/ " '+directory+'/*.par'
        if "ISOTHERMAL" in open(directory+'/summary0.dat',"r").read():
            energyequation = "No"
        else:
            energyequation = "Yes"
    else:
        command  = par.awk_command+' " /^AspectRatio/ " '+directory+'/*.par'
        command2 = par.awk_command+' " /^FlaringIndex/ " '+directory+'/*.par'
        command3 = par.awk_command+' " /^EnergyEquation/ " '+directory+'/*.par'
        buf3 = subprocess.getoutput(command3)
        energyequation = str(buf3.split()[1])

    buf = subprocess.getoutput(command)
    aspectratio = float(buf.split()[1])
    buf2 = subprocess.getoutput(command2)
    fli = float(buf2.split()[1])
    rpla0_normtq = np.sqrt( Xpla[0]*Xpla[0] + Ypla[0]*Ypla[0] )
    h = aspectratio*(rpla0_normtq**fli)  # 

        # get adiabatic index
    if energyequation == 'Yes':
        if fargo3d == 'Yes':
            command4 = par.awk_command+' " /^GAMMA/ " '+directory+'/*.par'
        else:
            command4 = par.awk_command+' " /^AdiabaticIndex/ " '+directory+'/*.par'
        buf4 = subprocess.getoutput(command4)
        adiabatic_index = float(buf4.split()[1])
    else:
        adiabatic_index = 1.0

    # get local azimuthally averaged surface density
    myfield0 = Field(field='dens', fluid='gas', on=0, directory=directory, physical_units='No', nodiff='Yes', fieldofview=par.fieldofview, slice=par.slice, onedprofile='Yes', override_units=par.override_units)
    dens = np.sum(myfield0.data,axis=1) / myfield0.nsec
    imin = np.argmin(np.abs(myfield0.rmed-rpla0_normtq))
    sigmap = dens[imin]
    Gamma_0 = (q/h/h)*sigmap*rpla0_normtq/adiabatic_index

    return Gamma_0


def extract_awk(name, par, directory):
    import subprocess
    command   = f'{par.awk_command} " /^{name}/ " {directory}/*.par'
    # command4 = par.awk_command+' " /^AdiabaticIndex/ " '+directory[j]+'/*.par'
    buf = subprocess.getoutput(command)
    if len(buf.split())>1:
        value = str(buf.split()[1])
    else:
        value="?"
    return value


