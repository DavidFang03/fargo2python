import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import fnmatch
from field import Field
import itertools

import scipy.special
import scipy.optimize
import sys

from BLIT import BlitManager
import david_tools as tools

import matplotlib.animation as animation

def compute_rh(q,a):
    '''
    return hill radius
    '''
    return np.pow(q/3,1/3)*a+1e-15

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

def formula_corot_torque(rmed, q,h,Sigma, rpla, omega, dr):
    der_term = np.diff(Sigma*rmed**(3/2))/dr[1:]
    i_rp = np.argmin(np.abs(rmed-rpla))
    der_term_rp =  der_term[i_rp,:]
    return 3*(1.1)**2*np.sqrt(q)*omega**2* der_term_rp[0] / (h*rpla**2) # en phi = 0 ?

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

def init_frn(orbit_number, ax, xytext=(10,-20), ha='left'):
    frn = ax.annotate(
        r"$t={}$ orbits".format(orbit_number),
        (0, 1),
        xycoords="axes fraction",
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="center",
        animated=True)
    return frn

def set_angle_xticks(ax):
    ax.set_xticks([i*2*np.pi/12 for i in range(12)],[f"{i*30}°" for i in range(12)], color='red')

def from_cart_to_rad(ux,uy):
    return np.sqrt(ux**2+uy**2), tools.computephi(ux,uy)

def compute_arrow_coords(pos,delta,norm=1):
    x, y = pos
    dx, dy = np.array(delta)/norm

    # Convert starting point to polar
    r_start, phi_start = from_cart_to_rad(x, y)
    # Convert end point to polar
    r_end, phi_end = from_cart_to_rad(x + dx, y + dy)

    xy=(phi_end, r_end)  # End point
    xytext=(phi_start, r_start)# Start point
    return xy, xytext

def plot_arrow_on_polar_ax(ax, pos, delta, norm=1, text="", **kwargs):
    '''
    pos : tuple (x, y) - starting point in Cartesian coordinates
    delta : tuple (dx, dy) - displacement vector in Cartesian coordinates
    '''
    xy, xytext = compute_arrow_coords(pos,delta,norm)

    r_tip = np.sqrt((pos[0]+delta[0]/norm)**2+(pos[1]+delta[1]/norm)**2) # if the tip of the arrow is outside the range of the plot, it will not be shown.
    print("coucou",kwargs)
    kwargs["arrowprops"]["arrowstyle"]="->"
    if "lw" not in kwargs["arrowprops"] or "linewidth" not in kwargs["arrowprops"]:
        kwargs["arrowprops"]["lw"]=2
    kwargs["xy"]=xy
    kwargs["xytext"]=xytext
    arrow = ax.annotate(
                "",
                **kwargs,
                animated=True,
                )

    text = ax.text(xy[0], 0.1+xy[1], s=text, va='center', ha='center', color=kwargs["arrowprops"]["color"])

    if r_tip>ax.get_ylim()[1]:
        ax.set_ylim(0,1.1*r_tip)

    # arrow = ax.quiver(phi_start, r_start, 
    #         phi_end - phi_start, r_end - r_start,
    #         angles='uv', scale_units='xy', scale=1, **kwargs)


    return arrow, text

class MyTorque():
    def __init__(self, **kwargs):
        '''
        Possible kwargs:
        on_start (int) and on_end (int) : range of output numbers (change it for debug so it loops faster). Default : first and last output of the directory.
        init_on (int) : initial time (if you want to initially show the data of output $n$ e.g.). Default : 'on_end'.
        '''
        import par
        self.directory = par.directory

        self.nb_outputs = extract_nb_outputs(par, self.directory)  # # Number of outputs

        # ! Plot infos<
        self.on_start   = kwargs.get("on_start",    0)
        self.on_end     = kwargs.get("on_end",      self.nb_outputs)
        self.init_on    = kwargs.get("init_on",     self.on_end - 1)
        self.curr_on    = self.init_on
        self.loop_length = self.on_end - self.on_start
        self.on = range(self.nb_outputs)

        self.do_movie = kwargs.get("do_movie", False)


        self.plot_settings = {"tot": True,
                         "map": True,
                         "rad": True,
                         "corr": False,
                         "ITd": True}


        # ! Grid infos
        self.dens0 = get_dens(self.on,0,par,self.directory)
        self.vphi0 = get_vphi(self.on,0,par,self.directory)
        self.nrad, self.nsec = self.dens0.nrad, self.dens0.nsec
        self.surface, self.dr, self.dphi, self.rdphi = get_grid(self.dens0)
        self.vecpos_cells = get_vecpos_cells(self.dens0) 
        self.rmin, self.rmax = self.dens0.redge[0], self.dens0.redge[-1]
        self.pmed_deg = self.dens0.pmed * 180.0 / np.pi

        self.PHI_meshgrid, self.R_meshgrid = np.meshgrid(self.dens0.pmed, self.dens0.rmed)

        # ! Planet infos
        self.Xpla, self.Ypla, self.Mpla, self.Date, self.Omegapla = get_pla(self.dens0, self.directory, self)

        self.apla = get_apla(self.directory)
        if np.isnan(self.apla):
            self.apla = 1
        self.t_orb = 2.0 * np.pi * (self.apla**1.5)
        self.Mytime = self.Date / self.t_orb
        self.dt = np.diff(self.Mytime)

        self.current_orbit = int(self.Mytime[self.curr_on])

        self.q = self.Mpla[-1]
        self.rh = compute_rh(self.q,self.apla)


        # ! Simulation infos
        self.sim_infos_dic = {"Sigma0": {"symbol" : r"$\Sigma_0$"},
        "SigmaSlope": {"symbol" : r"$\sigma$"},
        "AspectRatio": {"symbol" : r"$h$"},
        "FlaringIndex": {"symbol" : r"$f$"},
        "AlphaViscosity": {"symbol" : r"$\alpha$"},

        "EnergyEquation": {"symbol" : "EnergyEquation"},
        "AdiabaticIndex": {"symbol" : r"$\gamma$"},
        "TempPresc": {"symbol" : "TempPresc"},
        "PrescTime0": {"symbol" : "PrescTime0"},
        "ViscousHeating": {"symbol" : "ViscousHeating"},
        "ThermalCooling": {"symbol" : "ThermalCooling"},
        "StellarIrradiation": {"symbol" : "StellarIrradiation"},
        "BetaCooling": {"symbol" : "BetaCooling"},
        "ThermalDiffusion": {"symbol" : "ThermalDiffusion"},
        "EntropyDiffusion": {"symbol" : "EntropyDiffusion"},

        "ThicknessSmoothing": {"symbol" : r"$\eta$"},
        "MassTaper": {"symbol" : "MassTaper"},
        "Rmin": {"symbol" : r"$r_{min}$"},
        "Rmax": {"symbol" : r"$r_{max}$"},
        "Nrad": {"symbol" : r"$N_{rad}$"},
        "Nsec": {"symbol" : r"$N_{sec}$"},
        "WKZRmin": {"symbol" : "WKRmin"},
        "WKZRmax": {"symbol" : "WKRmax"},
        "Ntot": {"symbol" : "Ntot"},
        "Ninterm": {"symbol" : "Ninterm"}}
        for key in self.sim_infos_dic.keys():
            self.sim_infos_dic[key]["value"] = extract_awk(key, par, self.directory)

        self.sim_infos_dic["nb_outputs"] = {"symbol" : "nb_outputs", "value": self.nb_outputs}
        self.sim_infos_dic["q"] = {"symbol" : r"$q$", "value": f"{self.Mpla[-1]:.0e}"}

        self.eta = float(self.sim_infos_dic["ThicknessSmoothing"]["value"])
        self.ar = float(self.sim_infos_dic["AspectRatio"]["value"])
        self.fli = float(self.sim_infos_dic["FlaringIndex"]["value"])

        self.masstaper = extract_masstaper(par, self.directory)
        self.xs = 1.16*np.sqrt(self.Mpla[-1]/self.ar) # 
        rpospla0 = np.linalg.norm([self.Xpla[0], self.Ypla[0],0])
        self.rsup_gap = rpospla0 + self.xs
        self.rinf_gap = rpospla0 - self.xs
        self.isup_gap = np.argmin(np.abs(self.dens0.rmed-self.rsup_gap))
        self.iinf_gap = np.argmin(np.abs(self.dens0.rmed-self.rinf_gap))

        # Text simulation parameters
        self.text_sim_infos = '\n'.join([f'{self.sim_infos_dic[key]["symbol"]} = {self.sim_infos_dic[key]["value"]}' for key in self.sim_infos_dic.keys()])

        # ! Init arrays
        self.IndirectForce = np.zeros((self.nb_outputs,3))
        self.DirectForce = np.zeros((self.nb_outputs,3))

        self.Dtorque_tot = np.zeros(self.nb_outputs)
        self.Itorque_tot = np.zeros(self.nb_outputs)

        self.Dtorque_radial_density,self.Dtorque_radial_cumu = np.zeros((self.nb_outputs, self.dens0.nrad)), np.zeros((self.nb_outputs, self.dens0.nrad))
        self.Itorque_radial_density,self.Itorque_radial_cumu = np.zeros((self.nb_outputs, self.dens0.nrad)), np.zeros((self.nb_outputs, self.dens0.nrad))

        self.Dtorque_az_density,self.Dtorque_az_cumu = np.zeros((self.nb_outputs, self.dens0.nsec)), np.zeros((self.nb_outputs, self.dens0.nsec))
        self.Itorque_az_density,self.Itorque_az_cumu = np.zeros((self.nb_outputs, self.dens0.nsec)), np.zeros((self.nb_outputs, self.dens0.nsec))

        self.Dtorque_cell = np.zeros((self.nb_outputs, self.dens0.nrad, self.dens0.nsec))
        self.Itorque_cell = np.zeros((self.nb_outputs, self.dens0.nrad, self.dens0.nsec))
        self.Dtorque_cell_raveled = np.zeros((self.nb_outputs, self.dens0.nrad * self.dens0.nsec))
        self.Itorque_cell_raveled = np.zeros((self.nb_outputs, self.dens0.nrad * self.dens0.nsec))

        self.Perturbed_density = np.zeros((self.nb_outputs, self.dens0.nrad*self.dens0.nsec))
        self.Perturbed_vphi = np.zeros((self.nb_outputs, self.vphi0.nrad*self.vphi0.nsec))

        self.Dtorque_az_density_gap = np.zeros((self.nb_outputs, self.dens0.nsec))
        self.Itorque_az_density_gap = np.zeros((self.nb_outputs, self.dens0.nsec))

        # ! Plot parameters
        self.tq_tot_lim_plot = []
        self.PHI_GRID, self.R_GRID = np.meshgrid(self.dens0.pedge, self.dens0.redge)
        self.PHI, self.R = np.meshgrid(self.dens0.pedge, [0,np.max(self.dens0.redge)])
        self.PHI_gap, self.R_gap = np.meshgrid(self.dens0.pedge, [self.rinf_gap, self.rsup_gap])

        self.vmin_pert_dens,self.vmax_pert_dens = -1,1
        self.vmin_pert_vphi,self.vmax_pert_vphi = -1,1

        # # ! Init movies
        # self.output_imgs_map = []

        self.loop()
        self.read_dat()
        self.plot()
        if self.do_movie:
            self.movie_map()

    def loop(self):
        import par
        for k in range(self.on_start, self.on_end):
            print('output number =', str(k), 'out of', str(len(self.on)), end='\r')
            # print('output number =', str(k), 'out of', str(len(self.on)))
            self.dens = get_dens(self.on,k, par, self.directory)
            self.vphi = get_vphi(self.on,k, par, self.directory)

            dens = self.dens
            DensityPrime  = dens.data - (1/(2*np.pi))*np.sum(dens.data*self.dphi, axis=1)[:,np.newaxis]

            # ! Planet coordinates
            xpla, ypla, mpla, omegapla = self.Xpla[k], self.Ypla[k], self.Mpla[k], self.Omegapla[k]
            vecpospla = np.array([xpla, ypla, 0])
            vecpospla_grid = np.tile(vecpospla[:, np.newaxis, np.newaxis], (1, dens.nrad, dens.nsec)) # On repete vecpospla le long des nouveaux axes.
            # vecpospla_replicas = np.tile(vecpospla, (dens.nrad,1)).T

            eps = compute_epsilon(self.eta, self.ar, xpla, ypla, self.fli) # ? epsilon dépend de la position de la planète ?

            distances = np.sqrt(np.linalg.norm(self.vecpos_cells-vecpospla_grid, axis=0)**2+eps**2)
            dir_acc_cell = (self.vecpos_cells-vecpospla_grid) * DensityPrime/np.pow(distances,3) # intégrande
            indir_acc_cell = - DensityPrime/(dens.rmed**3)[:,np.newaxis]*self.vecpos_cells

            self.IndirectForce[k,0:3] = np.sum(indir_acc_cell*self.surface, axis=(1,2))
            self.DirectForce[k,0:3] = np.sum(dir_acc_cell*self.surface, axis=(1,2))

            self.Dtorque_cell[k,:,:]=torque_of_each_cell(dir_acc_cell,self.surface, vecpospla)
            self.Itorque_cell[k,:,:]=torque_of_each_cell(indir_acc_cell,self.surface, vecpospla)

            if self.plot_settings["map"]:
                self.Dtorque_cell_raveled[k] = self.Dtorque_cell[k].ravel()
                self.Itorque_cell_raveled[k] = self.Itorque_cell[k].ravel()

            self.Perturbed_density[k,:] = (self.dens.data[:,:]/np.average(self.dens.data[:,:]) - 1).ravel()
            self.Perturbed_vphi[k,:] = (self.vphi.data[:,:]/np.average(self.vphi.data) - 1).ravel()

            self.Dtorque_tot[k],self.Dtorque_radial_density[k,:],self.Dtorque_az_density[k,:],self.Dtorque_radial_cumu[k,:],self.Dtorque_az_cumu[k,:] = integrate_torque(dir_acc_cell, self.dens, self.surface, self.dr, self.dphi, self.rdphi, vecpospla)
            self.Itorque_tot[k],self.Itorque_radial_density[k,:],self.Itorque_az_density[k,:],self.Itorque_radial_cumu[k,:],self.Itorque_az_cumu[k,:] = integrate_torque(indir_acc_cell, self.dens, self.surface, self.dr, self.dphi, self.rdphi, vecpospla)

            self.Dtorque_az_density_gap[k,:] = integrate_torque_gap(dir_acc_cell,  self.dens, self.dr, self.iinf_gap, self.isup_gap, vecpospla)
            self.Itorque_az_density_gap[k,:] = integrate_torque_gap(indir_acc_cell, self.dens, self.dr, self.iinf_gap, self.isup_gap, vecpospla)

            self.rta_Dtorque = np.cumsum(self.Dtorque_tot[1:-1]*self.dt[1:])/self.Mytime[1:-1] # eviter division par 0
            self.rta_Itorque = np.cumsum(self.Itorque_tot[1:-1]*self.dt[1:])/self.Mytime[1:-1]

    def read_dat(self):
        # ! .dat files
        import par
        _, self.indtqdat, time_indtqdat = np.loadtxt(self.directory+"/indtq0.dat", unpack=True)
        f1, it, ot, it_ex, ot_ex, ip, op, f8, f9, time_dirtqdat = np.loadtxt(self.directory+"/tqwk0.dat",unpack=True)
        self.dirtqdat = it+ot
        self.time_indtqdat=time_indtqdat/self.t_orb
        self.time_dirtqdat=time_indtqdat/self.t_orb

        self.dirtqdat_ex = it_ex+ot_ex

        self.indexdat_masstaper = np.argmin(np.abs(self.time_indtqdat-self.masstaper))

        if par.normalize_torque == 'Yes':
            Gamma0 = compute_Gamma0(self.Mpla, self.Xpla, self.Ypla, self.directory)
            self.indtqdat /= Gamma0
            self.dirtqdat /= Gamma0
            self.dirtqdat_ex /= Gamma0
            self.Dtorque_tot /= Gamma0
            self.Itorque_tot /= Gamma0
            self.Dtorque_radial_density /= Gamma0
            self.Itorque_radial_density /= Gamma0
            self.Dtorque_az_density /= Gamma0
            self.Itorque_az_density /= Gamma0
            self.Dtorque_radial_cumu /= Gamma0
            self.Itorque_radial_cumu /= Gamma0
            self.Dtorque_az_cumu /= Gamma0
            self.Itorque_az_cumu /= Gamma0
            self.rta_Dtorque /= Gamma0
            self.rta_Itorque /= Gamma0

    def plot_tot_torque(self):
        self.fig_tot, self.ax_tot = plt.subplots(num="Total Torque",figsize=(20,10))
        # * * ##############################################################
        # * * ######################## TOTAL TORQUE ########################
        # * * ##############################################################
        ax = self.ax_tot
        # ! Plot .dat torque
        ax.plot(self.time_dirtqdat, self.dirtqdat, "-", alpha=0.5, label=r"$\Gamma_{dir}/\Gamma_0$", color="blue")
        ax.plot(self.time_indtqdat, self.indtqdat, "-", alpha=0.5, label=r"$\Gamma_{ind}/\Gamma_0$", color="orange")

        # self.ax_tot.plot(self.time_dirtqdat, self.dirtqdat_ex, "-x", label=r"$\Gamma_{dir}/\Gamma_0$ (Hill radius excluded)", color="green")
        # ! Plot total torque
        ax.scatter(self.Mytime, self.Dtorque_tot, s=20, alpha=1.0, label=r"$\Gamma_{dir}/\Gamma_0$", color="blue")
        ax.scatter(self.Mytime, self.Itorque_tot, s=20, alpha=1.0, label=r"$\Gamma_{ind}/\Gamma_0$", color="orange")
        # # ! Plot r.t.a torque
        ax.scatter(self.Mytime[1:-1], self.rta_Dtorque,label=r"$\frac{1}{t}\int_0^t \Gamma_{dir}(t^\prime)dt^\prime /\Gamma_0$", alpha=0.3, marker="s",color="blue")
        ax.scatter(self.Mytime[1:-1], self.rta_Itorque,label=r"$\frac{1}{t}\int_0^t \Gamma_{ind}(t^\prime)dt^\prime /\Gamma_0$", alpha=0.3, marker="s",color="orange")
        # ! Plot Lindblad line
        ax.axhline(y=self.lindblad, color='red', linestyle='--', label='Lindblad torque')
        # ! Simulation parameters
        ax.text(1.05, 0.5, self.text_sim_infos, transform=self.ax_tot.transAxes,va='center', ha='left')
        ax.set_xlabel(r"$t$ [Orbits]")
        ax.set_ylabel(r"$\Gamma/\Gamma_0$")

        ax.legend(loc='upper left')
        self.fig_tot.subplots_adjust(right=0.7)

        self.fig_tot.suptitle(f'{self.directory} : Torques of the disk on the planet with time')

    def plot_rad_torque(self):
        self.fig_rad, self.axs_rad = plt.subplot_mosaic([['top left','top left', 'right'],['bottom LEFT', 'bottom left','right']], num="Radial Torque",figsize=(20,10),per_subplot_kw={('bottom left','bottom LEFT'): {'projection':'polar'}}, layout='constrained', width_ratios=[0.5,0.5, 1])
        # * * ###############################################################
        # * * #################### RADIAL DENSITY TORQUE ####################
        # * * ###############################################################
        #! Total
        self.axs_rad['top left'].plot(self.time_dirtqdat, self.dirtqdat, "--x",alpha=0.5, label=r"$\Gamma_{dir}/\Gamma_0$", color="blue")
        self.axs_rad['top left'].plot(self.time_indtqdat, self.indtqdat, "--x",alpha=0.5, label=r"$\Gamma_{ind}/\Gamma_0$", color="orange")
        
        #! Dots
        self.direct_dot_rad, = self.axs_rad["top left"].plot(self.Mytime[self.curr_on], self.Dtorque_tot[self.curr_on], alpha=1.0, ms=20, marker=".",color="black")
        self.indirect_dot_rad, = self.axs_rad["top left"].plot(self.Mytime[self.curr_on], self.Itorque_tot[self.curr_on], alpha=1.0, ms=20, marker=".",color="black")

        #! Density map

        self.map_dens = self.axs_rad['bottom left'].pcolormesh(self.PHI_GRID, self.R_GRID, np.reshape(self.Perturbed_density[self.curr_on,:], (self.nrad,self.nsec)), cmap="viridis",animated=True, vmin=self.vmin_pert_dens, vmax=self.vmax_pert_dens)
        self.fig_rad.colorbar(self.map_dens, ax=self.axs_rad['bottom left'])

        #! Vphi map
        self.map_vphi = self.axs_rad['bottom LEFT'].pcolormesh(self.PHI_GRID, self.R_GRID, np.reshape(self.Perturbed_vphi[self.curr_on,:], (self.nrad,self.nsec)), cmap="inferno",animated=True, vmin=self.vmin_pert_vphi, vmax=self.vmax_pert_vphi)
        self.fig_rad.colorbar(self.map_vphi, ax=self.axs_rad['bottom LEFT'])

        #! Radial lines
        self.line_dtorque_rad, = self.axs_rad['right'].plot(self.dens.rmed, self.Dtorque_radial_density[self.curr_on,:], "--x",alpha=0.5, label=r"$\Gamma_{dir}^\prime(r)/\Gamma_0$", color="blue",animated=True)
        self.line_itorque_rad, = self.axs_rad['right'].plot(self.dens.rmed, self.Itorque_radial_density[self.curr_on,:], "--x",alpha=0.5, label=r"$\Gamma_{ind}^\prime(r)/\Gamma_0$", color="orange",animated=True)
        # self.axs_rad[1].plot(self.dens.rmed, self.Dtorque_radial_cumu[self.curr_on,:], "-x",alpha=0.5, label=r"$\int_0^r \Gamma_{dir}^\prime(x)dx/\Gamma_0$", color="blue")
        # self.axs_rad[1].plot(self.dens.rmed, self.Itorque_radial_cumu[self.curr_on,:], "-x",alpha=0.5, label=r'$\int_0^r \Gamma_{ind}^\prime(x)dx / \Gamma_0$', color="orange")

        self.axs_rad['right'].set_xlabel(r'$r/r_p$')
        self.axs_rad['right'].set_ylabel(r'$\Gamma^\prime(r)/\Gamma_0$')
        self.axs_rad['right'].set_xscale('log')
        self.axs_rad['right'].set_title(f'Densité radiale des couples à un instant donné')

        self.axs_rad['right'].set_ylim(-5,5)
        frn1 = init_frn(self.current_orbit, self.axs_rad['right'])
        frn2 = init_frn(self.current_orbit, self.axs_rad['bottom left'], xytext=(0,0), ha='center')

        for ax in [self.axs_rad['top left'], self.axs_rad['right']]:
            ax.legend(loc="upper right")

        self.axs_rad['bottom left'].set_title(r"Perturbed density $\frac{\Sigma(r,\phi)}{\overline \Sigma} - 1$")
        self.axs_rad['bottom LEFT'].set_title(r"Perturbed azimuthal velocity $\frac{v_{\phi}(r,\phi)}{\overline v_{\phi}} - 1$")
        set_angle_xticks(self.axs_rad['bottom left'])
        set_angle_xticks(self.axs_rad['bottom LEFT'])

        self.fr_numbers_rad = [frn1, frn2]

        self.fig_tot.suptitle(f'{self.directory}')

    def fit_indtq_with_corot(self):
        self.corot = self.dirtqdat - self.lindblad

        self.fig_fit_ind_corot, self.ax_fit_ind_corot = plt.subplots(3,2, figsize=(20, 10))
        corot, indtqdat = self.corot[self.indexdat_masstaper:], self.indtqdat[self.indexdat_masstaper:]

        # ! Plot corot and indtqdat
        self.ax_fit_ind_corot[0,0].axvline(x=self.time_dirtqdat[self.indexdat_masstaper], color='red', linestyle='--', label='Mass taper end')
        self.ax_fit_ind_corot[0,0].plot(self.time_dirtqdat[self.indexdat_masstaper:], corot,"-x", alpha=0.1, label=r"$\Gamma_{corot}/\Gamma_0$", color="navy")
        self.ax_fit_ind_corot[0,0].plot(self.time_indtqdat[self.indexdat_masstaper:], indtqdat,"-x", alpha=0.1, label=r"$\Gamma_{ind}/\Gamma_0$", color="orange")
        tq_tot_lim_plot = [1.*np.min(corot), 1.*np.max(corot)]
        self.ax_fit_ind_corot[0,0].set_ylim(tq_tot_lim_plot[0], tq_tot_lim_plot[1])
        self.ax_fit_ind_corot[0,0].set_xlabel(r"$t$ [Orbits]")
        self.ax_fit_ind_corot[0,0].set_ylabel(r"$\Gamma/\Gamma_0$")
        self.ax_fit_ind_corot[0,0].set_title(f"{self.directory} : "+r'$\Gamma_{corot}$ et $\Gamma_{ind}$ en fonction du temps')

        # ! Plot indtq function of corot
        self.ax_fit_ind_corot[1,0].plot(corot, indtqdat, "-")
        (a,b), pcov = np.polyfit(corot, indtqdat, 1, cov=True)
        u_a, u_b = np.sqrt(np.diag(pcov))
        self.ax_fit_ind_corot[1,0].plot(corot, a*corot+b, "-", label=f"$\\Gamma_{{ind}}=({a:.3f}\\pm{u_a:.3f})\\Gamma_{{corot}}+{b:.3f}\\pm {u_b:.3f}$", color="black")


        self.ax_fit_ind_corot[1,0].set_xlabel(r"$\Gamma_{corot}/\Gamma_0$")
        self.ax_fit_ind_corot[1,0].set_ylabel(r"$\Gamma_{ind}/\Gamma_0$")
        self.ax_fit_ind_corot[1,0].set_title(r'$\Gamma_{ind}$ en fonction de $\Gamma_{corot}$')

        # ! Plot FFT
        fft_indtqdat = np.fft.fft(indtqdat-np.average(indtqdat))
        fft_corot = np.fft.fft(corot)
        print(self.dt[0])
        freq_indtqdat = np.fft.fftfreq(len(indtqdat), self.time_indtqdat[1]-self.time_indtqdat[0])
        freq_corot = np.fft.fftfreq(len(corot), self.time_indtqdat[1]-self.time_indtqdat[0])
        fft_indtqdat_normalized = fft_indtqdat/np.max(np.abs(fft_indtqdat))
        fft_corot_normalized = fft_corot/np.max(np.abs(fft_corot))

        threshold = 0.01
        lim_indtqdat_freq_fft = freq_indtqdat[np.argmin(np.abs(np.abs(fft_indtqdat_normalized)-threshold))]
        lim_corot_freq_fft = freq_corot[np.argmin(np.abs(np.abs(fft_corot_normalized)-threshold))]
        lim_freq_fft = np.max([np.abs(lim_indtqdat_freq_fft), np.abs(lim_corot_freq_fft)])


        self.ax_fit_ind_corot[0,1].plot(freq_indtqdat, np.abs(fft_indtqdat_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{ind})$", color="orange")
        self.ax_fit_ind_corot[0,1].plot(freq_corot, np.abs(fft_corot_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{corot})$", color="navy")
        self.ax_fit_ind_corot[0,1].set_ylabel(r"$|FFT(\Gamma)|$ [normalized]")
        self.ax_fit_ind_corot[0,1].set_title(r'FFT de $\Gamma_{ind}$ et $\Gamma_{corot}$')


        self.ax_fit_ind_corot[1,1].plot(freq_indtqdat, np.real(fft_indtqdat_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{ind})$", color="orange")
        self.ax_fit_ind_corot[1,1].plot(freq_corot, np.real(fft_corot_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{corot})$", color="navy")
        self.ax_fit_ind_corot[1,1].set_ylabel(r"$\Re(FFT(\Gamma))$ [normalized]")

        self.ax_fit_ind_corot[2,1].plot(freq_indtqdat, np.imag(fft_indtqdat_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{ind})$", color="orange")
        self.ax_fit_ind_corot[2,1].plot(freq_corot, np.imag(fft_corot_normalized), "-x", alpha=0.6, label=r"$FFT(\Gamma_{corot})$", color="navy")
        self.ax_fit_ind_corot[2,1].set_ylabel(r"$\Im(FFT(\Gamma))$ [normalized]")

        # ! FIT BESSEL ???
        timefit = self.time_dirtqdat[self.indexdat_masstaper:-int(len(corot)/2)]
        corotfit = corot[:-int(len(corot)/2)]


        bessel = lambda x, alpha, gamma0, a  : gamma0 * scipy.special.jv(alpha, a*x) 
        residual = lambda params, X, Y : bessel(X,*params) - Y
        params0 = [1,1,1/100]
        res = scipy.optimize.least_squares(residual, params0, args=(timefit, corotfit))
        bessel_fit_params = res.x

        bessel_exp = lambda x, alpha, gamma0, a, tau : gamma0 * np.exp(-x/tau) * scipy.special.jv(alpha, a*x) 
        residual_exp = lambda params, X, Y : bessel_exp(X,*params) - Y
        params0 = [1,1,1/100,200]
        res_exp = scipy.optimize.least_squares(residual_exp, params0, args=(timefit, corotfit))
        bessel_exp_fit_params = res_exp.x

        sin_exp = lambda x, alpha, gamma0, omega, tau, phase : gamma0 * np.exp(-x/tau) * np.sin(omega*x + phase)
        residual_sin_exp = lambda params, X, Y : sin_exp(X,*params) - Y
        params0 = [1,1,1/100,200, 0]
        res_sin_exp = scipy.optimize.least_squares(residual_sin_exp, params0, args=(timefit, corotfit))
        sin_exp_fit_params = res_sin_exp.x

        self.ax_fit_ind_corot[2,0].plot(timefit, corotfit, "-x",color="cyan")
        self.ax_fit_ind_corot[2,0].plot(timefit, bessel(timefit, *bessel_fit_params), "-", color="cyan", alpha=0.3, label=f"{[f"{a:.2f}" for a in bessel_fit_params]}")
        self.ax_fit_ind_corot[2,0].plot(timefit, bessel_exp(timefit, *bessel_exp_fit_params), "-", color="magenta", alpha=0.3, label=f"{[f"{a:.2f}" for a in bessel_exp_fit_params]}")
        self.ax_fit_ind_corot[2,0].plot(timefit, sin_exp(timefit, *sin_exp_fit_params), "-", color="blue", alpha=0.3, label=f"{[f"{a:.2f}" for a in sin_exp_fit_params]}")

        for i in range(3):
            for j in range(2):
                if j==1:
                    self.ax_fit_ind_corot[i,j].set_xlim(0, lim_freq_fft)
                    self.ax_fit_ind_corot[i,j].set_ylim(top=1.1)
                    self.ax_fit_ind_corot[i,j].set_xlabel(r"$f$ [Orbit$^{-1}$]")
                self.ax_fit_ind_corot[i,j].legend(loc="upper right")

        self.ax_fit_ind_corot[1,0].axis('equal')
        self.fig_fit_ind_corot.tight_layout()
        # self.fig_fit_ind_corot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

    def plot_torque_map(self):
        k = self.curr_on
        self.fig_map, self.ax_map = plt.subplot_mosaic([['left','right']],per_subplot_kw={('left', 'right'): {'projection':'polar'}}, num='Map', figsize=(20,10), layout="constrained")

        fig, axs = self.fig_map, self.ax_map

        self.map_dtorque_cell       = axs['left'].pcolormesh(self.PHI_GRID,self.R_GRID, self.Dtorque_cell[k,:,:], cmap='viridis',animated=True)
        self.colorbar_phimap_dir    = fig.colorbar(self.map_dtorque_cell, ax=axs['left'])

        # self.map_itorque_cell       = axs['right'].pcolormesh(self.PHI_GRID,self.R_GRID, self.Itorque_cell[k,:,:], cmap='inferno',animated=True)

        self.map_itorque_cell       = axs['right'].pcolormesh(self.PHI_GRID,self.R_GRID, np.reshape(self.Perturbed_density[k,:], (self.nrad,self.nsec)), cmap='inferno',animated=True)
        self.colorbar_phimap_indir  = fig.colorbar(self.map_itorque_cell, ax=axs['right'])

        # ! Show direction of ITd
        norm_arrow_itd = np.max(np.linalg.norm(self.IndirectForce, axis=1))
        norm_arrow_dir = np.max(np.linalg.norm(self.DirectForce, axis=1))
        self.arrow_norm = np.max((norm_arrow_itd,norm_arrow_dir))
        self.arrow_norm = 5* 10 ** (np.floor(np.log10(self.arrow_norm))) # arrondi à la puissance inférieure
        # self.arrow_norm = norm_arrow_itd
        print(f"{self.arrow_norm:.1e}")
        # self.arrow_norm = 1e-7

        self.ITd_arrow_right, self.ITd_arrow_right_text = plot_arrow_on_polar_ax(axs['right'], (0,0), - self.IndirectForce[k, 0:2], norm=self.arrow_norm, text=r"$\mathbf a_{\star}$", arrowprops=dict(color="green"))
        # self.Direct_arrow_right = plot_arrow_on_polar_ax(axs['right'], (self.Xpla[k],self.Ypla[k]), - self.DirectForce[k, 0:2], norm=self.arrow_norm, color="blue", linewidth=2)

        print(self.DirectForce[k,0:2])
        self.Direct_arrow_right, self.Direct_arrow_right_text= plot_arrow_on_polar_ax(axs['right'], (self.Xpla[k],self.Ypla[k]), self.DirectForce[k,0:2], norm=self.arrow_norm, text=r"$\mathbf{a}_{dir}$", arrowprops=dict(color="blue"))

        rpla, phipla = from_cart_to_rad(self.Xpla[k], self.Ypla[k])
        self.planet_on_map, = plt.plot([phipla],[rpla],".", color="black")

        for pos, ax in axs.items():
            set_angle_xticks(ax)
            # ax.set_ylim(0,1.1)

        fig.suptitle(f'Couples de chaque cellule à un instant donné')

        axs['left'].set_title(r"$\Gamma_{dir}(r, \phi)$")
        # axs['right'].set_title(r"$\Gamma_{indir}(r, \phi)$")
        axs['right'].set_title(r"Perturbed density")

        frn1 = init_frn(self.current_orbit, axs['left'])
        frn2 = init_frn(self.current_orbit, axs['right'])
        self.fr_numbers_map = [frn1, frn2]

    def plot_ITd(self):
        self.fig_it, self.axs_it = plt.subplots(3,num="Indirect Term",figsize=(20,10), layout="constrained")
        self.fig_it.suptitle("Indirect Term of the disk components")
        labels = ["ITdx", "ITdy", "|ITd|"]

        for i in range(3):
            ax = self.axs_it[i]
            if i<=1:
                ax.plot(self.Mytime, self.IndirectForce[:,i], "-x")
            elif i==2:
                ax.plot(self.Mytime, np.linalg.norm(self.IndirectForce, axis=1), "-x")
            ax.set_xlabel(r"$t$ [orbits]")
            ax.set_ylabel(rf"${labels[i]}$")
            ax.set_title(rf"${labels[i]}$")


    def on_press(self,event):
        if event.key == 'right':
            self.curr_on = self.on_start + (self.curr_on + 1) % self.loop_length
        elif event.key == 'left':
            self.curr_on = self.on_start + (self.curr_on - 1) % self.loop_length
        elif event.key == 'up':
            self.curr_on = self.on_end - 1
        elif event.key == 'down':
            self.curr_on = 0

        if event.key in ['right', 'left', 'up', 'down']:
            self.update_plot()

    def plot(self):
        # ! Compute Lindblad line
        self.lindblad = np.average(self.dirtqdat[self.indexdat_masstaper:])
        
        figs_tolisten = []
        self.all_artists = []

        if self.plot_settings["tot"]:
            self.plot_tot_torque()

        if self.plot_settings["rad"]:
            self.plot_rad_torque()
            artists_rad =  [self.line_dtorque_rad, self.line_itorque_rad,self.map_dens, self.map_vphi,*self.fr_numbers_rad, self.direct_dot_rad, self.indirect_dot_rad]
            self.bm_rad = BlitManager(self.fig_rad.canvas, artists_rad)

            figs_tolisten += [self.fig_rad]
            self.all_artists += artists_rad

        if self.plot_settings["map"]:
            self.plot_torque_map()
            artists_map = [self.map_dtorque_cell, self.map_itorque_cell, *self.fr_numbers_map, self.ITd_arrow_right, self.Direct_arrow_right, self.planet_on_map, self.ITd_arrow_right_text, self.Direct_arrow_right_text]
            self.bm_map = BlitManager(self.fig_map.canvas, artists_map)

            figs_tolisten += [self.fig_map]
            self.all_artists += artists_map

        if self.plot_settings["corr"]:
            self.fit_indtq_with_corot()

        if self.plot_settings["ITd"]:
            self.plot_ITd()

        for fig in figs_tolisten :
            fig.canvas.mpl_connect('key_press_event', self.on_press)


    def update_plot(self, frame=False, *fargs):
        self.current_orbit = int(self.Mytime[self.curr_on])
        print("Current output : ", self.curr_on, end='\r')
        k = self.curr_on
        orb = self.current_orbit

        if self.plot_settings["rad"] and self.plot_settings["rad"]!= "no update":

            self.direct_dot_rad.set_data([self.Mytime[k]], [self.Dtorque_tot[k]])
            self.indirect_dot_rad.set_data([self.Mytime[k]], [self.Itorque_tot[k]])

            self.line_dtorque_rad.set_ydata(self.Dtorque_radial_density[k,:])
            self.line_itorque_rad.set_ydata(self.Itorque_radial_density[k,:])
            self.fr_numbers_rad[0].set_text(r"$t={}$ orbits".format(self.current_orbit))

            if self.plot_settings["rad"] != "no update map":
                self.map_dens.set_array(self.Perturbed_density[k])
                self.map_vphi.set_array(self.Perturbed_vphi[k])

                # for frn in self.fr_numbers_rad :
                self.fr_numbers_rad[1].set_text(r"$t={}$ orbits".format(self.current_orbit))

            self.bm_rad.update()

        if self.plot_settings["map"] and self.plot_settings["map"] != "no update":
            self.map_dtorque_cell.set_array(self.Dtorque_cell_raveled[k])
            # self.map_itorque_cell.set_array(self.Itorque_cell_raveled[k])
            self.map_itorque_cell.set_array(self.Perturbed_density[k])

            xy, xytext = compute_arrow_coords((0,0), -self.IndirectForce[k,0:2], norm=self.arrow_norm)
            self.ITd_arrow_right.xy = xy

            self.ITd_arrow_right_text.xy = xy


            xy, xytext = compute_arrow_coords((self.Xpla[k],self.Ypla[k]), self.DirectForce[k,0:2], norm=self.arrow_norm)
            self.Direct_arrow_right.xy = xy
            self.Direct_arrow_right.xytext = xytext
            self.Direct_arrow_right_text.xy = xy


            rpla, phipla = from_cart_to_rad(self.Xpla[k], self.Ypla[k])
            self.planet_on_map.set_data([phipla],[rpla])

            for frn in self.fr_numbers_map:
                frn.set_text(r"$t={}$ orbits".format(self.current_orbit))

            if self.do_movie:
                return self.all_artists
            self.bm_map.update()



    def movie_map(self):
        for k in range(self.on_start, self.on_end):
            self.curr_on = k
            self.update_plot()
            img_path = f"./output_imgs/map_{self.curr_on:04d}.png"
            self.fig_map.savefig(img_path)
            # self.output_imgs_map.append(img_path)
        self.curr_on = self.init_on
        img_path_for_ffmpeg = f"./output_imgs/map_%04d.png"

        # ani = animation.FuncAnimation(fig=self.fig_map, func=self.update_plot, frames=self.loop_length, interval=30, blit=True)
        # ani.save("./map.mp4")

        # import ffmpeg
        # (
        #     ffmpeg            
        #     .input(img_path_for_ffmpeg, start_number=self.on_start, framerate=1)
        #     # framerate=10 means the video will play at 10 of the original images per second
        #     .output(f"./{self.directory}_map.mp4", r=25, pix_fmt='yuv420p', **{'qscale:v': 3})
        #     # r=30 means the video will play at 30 frames per second
        #     .overwrite_output()
        #     .run()
        # )
# class MyForce()
        

def plot_mytq():
    matplotlib.pyplot.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': 18})
    mytorque = MyTorque()
    mytorque.loop()
    mytorque.read_dat()
    mytorque.plot()

if __name__ == "__main__":
    # pass
    matplotlib.pyplot.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': 18})
    mytorque = MyTorque(do_movie=False, on_start=98)

    plt.show()
