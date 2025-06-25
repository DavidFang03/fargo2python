import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.special
import scipy.optimize
import sys

from BLIT import BlitManager
from david_tools import *

import matplotlib.animation as animation

dark_mode = True
cmap = 'RdBu_r'
rangevminvmax = 1
plot_settings =     {"tot": False,
                        "map": True,
                        "rad": False,
                        "MULTI": False,
                        "FT": False}

transparent = False
oneshot = False

class MyUpdatedPlot():
    def __init__(   self, 
                    mytorque,
                    dim,
                    fig,
                    ax,
                    xlabel:str,
                    ylabel:str,
                    title:str,
                    xlims=None,
                    ylims=None,
                    ):


        self.dim = dim
        self.fig = fig
        self.ax = ax
        self.mytorque = mytorque
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=20)
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)

        self.curr_on = mytorque.curr_on
        self.current_orbit = mytorque.current_orbit


        self.lines          = []
        self.lines_ydata    = []
        self.dots           = []
        self.dots_xdata     = []
        self.dots_ydata     = []
        self.maps           = []
        self.maps_zdata     = []
        self.arrows         = []
        self.arrows_text    = []
        self.arrows_pos     = []
        self.arrows_delta   = []

        self.frn = []

        mytorque.FIGS[fig.myfigid]["uplots"].append(self)

    def add_line(   self,
                    xdata,
                    ydata,
                    label,
                    style:dict):
        '''
        Only ydata is supposed to change
        '''
        k = self.curr_on
        line, = self.ax.plot(xdata,ydata[k],label=label, animated=True, zorder=0, **style)
        self.lines.append(line)
        self.lines_ydata.append(ydata)


        return line

    def add_dot(self,
                xdata,
                ydata):
        '''
        Both xdata and ydata change
        '''
        k = self.curr_on
        dot, = self.ax.plot([xdata[k]],[ydata[k]], marker=".", ms=10, color="black",animated=True)
        self.dots.append(dot)
        self.dots_xdata.append(xdata)
        self.dots_ydata.append(ydata)

    def add_map(self,
                xdata,
                ydata,
                zdata,
                cmap,
                cbar=True,
                **kwargs):
        '''
        Only zdata is supposed to change. zdata must be raveled.
        '''
        k = self.curr_on
        # self.fig.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
        dimx = xdata.shape[0]
        dimy = xdata.shape[1]
        Z = np.reshape(zdata[k], (dimx - 1,dimy - 1))
        map = self.ax.pcolormesh(xdata, ydata, Z, cmap=cmap, animated=True,rasterized=True, **kwargs)
        self.ax.axis('equal')
        if cbar:
            self.fig.colorbar(map, ax=self.ax, aspect=50, shrink=0.6, orientation="horizontal")
        

        self.maps.append(map)
        self.maps_zdata.append(zdata)

    def add_arrow(self,
                    pos,
                    delta,
                    text="",
                    txt_dx=0.1,
                    txt_dy=0.1,
                    **kwargs):

        '''
        Both pos and delta must change with time.
        '''
        k = self.curr_on
        x, y = pos[k,0:2]
        dx, dy = delta[k,0:2]
        xytext = (x,y)
        xy = (x+dx, y+dy)
        r_tip = np.sqrt((x+dx)**2+(y+dy)**2) # if the tip of the arrow is outside the range of the plot, it will not be shown.

        kwargs["arrowprops"]["arrowstyle"]="simple"
        kwargs["arrowprops"]["edgecolor"]="white"

        if "lw" not in kwargs["arrowprops"] or "linewidth" not in kwargs["arrowprops"]:
            kwargs["arrowprops"]["lw"]=0.5

        arrow = self.ax.annotate(
                    "",
                    xy=xy,
                    xytext=xytext,
                    **kwargs,
                    )
        self.txt_dx, self.txt_dy = txt_dx, txt_dy
        text = self.ax.text(x+dx+txt_dx, y+dy+txt_dy, s=text, va='center', ha='center', color=arrow.arrow_patch.get_facecolor())

        range_map = np.max((self.ax.get_xlim()[1],self.ax.get_ylim()[1]))
        dr=0.5
        if r_tip>range_map:
            self.ax.set_xlim(-1.1*r_tip,1.1*r_tip)
            self.ax.set_ylim(-1.1*r_tip,1.1*r_tip)

        self.arrows.append(arrow)
        self.arrows_text.append(text)
        self.arrows_pos.append(pos)
        self.arrows_delta.append(delta)


    def add_frn(self, xytext=(10,-20), ha='left', pos=(0, 1)):
        frn = self.ax.annotate(
            r"$t={}$ orbits".format(self.current_orbit),
            pos,
            xycoords="axes fraction",
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va="center",
            animated=True,
            color="black")
        self.frn = [frn]
        return frn



    def lock(self, loc='best', **kwargs):
        '''
        Add to Blitmanager to end the plot.
        '''
        self.legend=[]
        if self.dim != "3D":
            legend = self.ax.legend(loc=loc, facecolor='white', framealpha=0.7, labelcolor="black", **kwargs)
            legend.set_animated(True)
            self.legend=[legend]
        artists = [*self.lines, *self.dots, *self.maps, *self.arrows, *self.arrows_text, *self.frn, *self.legend]
        self.mytorque.FIGS[self.fig.myfigid]["uartists"] +=  artists


    def update(self):
        self.curr_on = self.mytorque.curr_on
        self.current_orbit = self.mytorque.current_orbit
        k = self.curr_on

        for i in range(len(self.lines)):
            ydata = self.lines_ydata[i]
            line = self.lines[i]
            line.set_ydata(ydata[k])

        for i in range(len(self.dots)):
            dot = self.dots[i]
            xdata = self.dots_xdata[i]
            ydata = self.dots_ydata[i]
            x=xdata[k]
            y=ydata[k]
            dot.set_xdata([x])
            dot.set_ydata([y])

        for i in range(len(self.maps)):
            map = self.maps[i]
            zdata = self.maps_zdata[i]
            map.set_array(zdata[k])

        for i in range(len(self.arrows)):
            arrow = self.arrows[i]
            text = self.arrows_text[i]
            pos = self.arrows_pos[i]
            delta = self.arrows_delta[i]
            x, y = pos[k,0:2]
            dx, dy = delta[k,0:2]
            xytext = (x,y)
            xy = (x+dx, y+dy)

            arrow.set_position(xytext) # arrow.xytext = machin ne fait rien
            arrow.xy = xy
            text.xy = (x+dx+self.txt_dx, y+dy+self.txt_dy)

        if len(self.frn) > 0:
            frn = self.frn[0]
            frn.set_text(r"$t={}$ orbits".format(self.current_orbit))







class MyTorque():
    def __init__(self, plot_settings, **kwargs):
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
        self.on_end     = kwargs.get("on_end",      self.nb_outputs - 1)
        if self.on_start < 0:
            self.on_start = self.on_end + self.on_start
        self.init_on    = kwargs.get("init_on",     self.on_end)
        self.curr_on    = self.init_on
        self.loop_length = self.on_end - self.on_start + 1
        self.on = range(self.nb_outputs)

        self.plot_settings = plot_settings

        ignore_start_for_fft = kwargs.get("ignore_start_for_fft", 0)

        self.funcanim_map = kwargs.get("funcanim_map", False)


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

        self.current_orbit = round(self.Mytime[self.curr_on])

        self.q = self.Mpla[-1]
        self.rh = compute_rh(self.q,self.apla)
        self.typeI = self.q < 1e-4
        self.typeII = self.q > 1e-4


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

        self.Density_norm = np.zeros((self.nb_outputs, self.dens0.nrad*self.dens0.nsec))
        self.Perturbed_density = np.zeros((self.nb_outputs, self.dens0.nrad*self.dens0.nsec))
        self.Perturbed_vphi = np.zeros((self.nb_outputs, self.vphi0.nrad*self.vphi0.nsec))

        self.Dtorque_az_density_gap = np.zeros((self.nb_outputs, self.dens0.nsec))
        self.Itorque_az_density_gap = np.zeros((self.nb_outputs, self.dens0.nsec))

        # ! Plot parameters
        self.tq_tot_lim_plot = []
        self.PHI_GRID, self.R_GRID = np.meshgrid(self.dens0.pedge, self.dens0.redge)

        self.X_GRID = self.R_GRID * np.cos(self.PHI_GRID)
        self.Y_GRID = self.R_GRID * np.sin(self.PHI_GRID)

        self.vmin_pert_dens,self.vmax_pert_dens = -1,1
        self.vmin_pert_vphi,self.vmax_pert_vphi = -1,1

        # # ! Init movies
        # self.output_imgs_map = []

        # ! To be updated artistis :
        self.FIGS = {}

        # ! RUN
        self.loop()
        self.read_dat(ignore_start_for_fft)
        self.plot()


    def loop(self):
        import par
        for k in range(self.on_start, self.on_end +1):
            # print('output number =', str(k), 'out of', str(len(self.on)), end='\r')
            print('output number =', str(k), 'out of', str(len(self.on)))
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
            dir_acc_cell = (self.vecpos_cells-vecpospla_grid) * DensityPrime/distances**3 # intégrande
            indir_acc_cell = - DensityPrime/(dens.rmed**3)[:,np.newaxis]*self.vecpos_cells

            self.IndirectForce[k,0:3] = np.sum(indir_acc_cell*self.surface, axis=(1,2))
            self.DirectForce[k,0:3] = np.sum(dir_acc_cell*self.surface, axis=(1,2))

            self.Dtorque_cell[k,:,:]=torque_of_each_cell(dir_acc_cell,self.surface, vecpospla)
            self.Itorque_cell[k,:,:]=torque_of_each_cell(indir_acc_cell,self.surface, vecpospla)

            if self.plot_settings["map"]:
                self.Dtorque_cell_raveled[k] = self.Dtorque_cell[k].ravel()
                self.Itorque_cell_raveled[k] = self.Itorque_cell[k].ravel()

            self.Density_norm[k,:]  = (dens.data/self.dens0.data).ravel()
            self.Perturbed_density[k,:] = (self.dens.data[:,:]/np.average(self.dens.data[:,:]) - 1).ravel()
            self.Perturbed_vphi[k,:] = (self.vphi.data[:,:]/np.average(self.vphi.data) - 1).ravel()

            self.Dtorque_tot[k],self.Dtorque_radial_density[k,:],self.Dtorque_az_density[k,:],self.Dtorque_radial_cumu[k,:],self.Dtorque_az_cumu[k,:] = integrate_torque(dir_acc_cell, self.dens, self.surface, self.dr, self.dphi, self.rdphi, vecpospla)
            self.Itorque_tot[k],self.Itorque_radial_density[k,:],self.Itorque_az_density[k,:],self.Itorque_radial_cumu[k,:],self.Itorque_az_cumu[k,:] = integrate_torque(indir_acc_cell, self.dens, self.surface, self.dr, self.dphi, self.rdphi, vecpospla)

            self.Dtorque_az_density_gap[k,:] = integrate_torque_gap(dir_acc_cell,  self.dens, self.dr, self.iinf_gap, self.isup_gap, vecpospla)
            self.Itorque_az_density_gap[k,:] = integrate_torque_gap(indir_acc_cell, self.dens, self.dr, self.iinf_gap, self.isup_gap, vecpospla)


    def read_dat(self, ignore_start_for_fft):
        # ! .dat files
        import par
        _, self.indtqdat, time_indtqdat = np.loadtxt(self.directory+"/indtq0.dat", unpack=True)
        f1, it, ot, it_ex, ot_ex, ip, op, f8, f9, time_dirtqdat = np.loadtxt(self.directory+"/tqwk0.dat",unpack=True)
        self.dirtqdat = it+ot
        self.time_indtqdat=time_indtqdat/self.t_orb
        self.time_dirtqdat=time_indtqdat/self.t_orb

        self.dirtqdat_ex = it_ex+ot_ex

        dt_dat = np.diff(self.time_indtqdat)
        self.rta_Dtorque = np.cumsum(self.dirtqdat[1:-1]*dt_dat[1:])/self.time_dirtqdat[1:-1] # eviter division par 0
        self.rta_Itorque = np.cumsum(self.indtqdat[1:-1]*dt_dat[1:])/self.time_indtqdat[1:-1]

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

        print(f"A la dernière orbite t={round(self.Mytime[-1])} orbites")
        print(f"Gamma_dir = {self.Dtorque_tot[-1]:.2f} Gamma0")
        print(f"Gamma_ind = {self.Itorque_tot[-1]:.2f} Gamma0")
        print(f"Gamma_ind = {(self.Itorque_tot[-1]/self.Dtorque_tot[-1]) :.2f} Gamma_dir")
        print(f"Pour les rta t={round(self.Mytime[-1])} orbites")
        print(f"Gamma^rta_dir = {self.rta_Dtorque[-1]:.2f} Gamma0")
        print(f"Gamma^rta_ind = {self.rta_Itorque[-1]:.2f} Gamma0")
        print(f"Gamma^rta_ind = {(self.rta_Itorque[-1]/self.rta_Dtorque[-1]) :.2f} Gamma^rta_dir")

        # ! On veut comparer le couple indirect avec le couple de corotation
        if self.plot_settings["FT"] or self.plot_settings["rad"] or self.plot_settings["tot"]:
            if ignore_start_for_fft<self.masstaper:
                start_orb = self.masstaper
            else:
                start_orb = ignore_start_for_fft
            self.startindex_dat = np.argmin(np.abs(self.time_indtqdat-start_orb))

            lindblad = np.average(self.dirtqdat[self.startindex_dat:])
            corot = self.dirtqdat - self.dirtqdat[-1]
            self.corot = corot

            corot_cut, indtqdat_cut = corot[self.startindex_dat:], self.indtqdat[self.startindex_dat:]

            # fft_indtqdat = np.fft.fft(indtqdat_cut-np.average(indtqdat_cut))
            fft_indtqdat = np.fft.fft(indtqdat_cut)
            fft_corot = np.fft.fft(corot_cut)
            if len(indtqdat_cut)!=len(corot_cut):
                raise Exception("wtf")

            freq = np.fft.fftfreq(len(corot_cut), self.time_indtqdat[1]-self.time_indtqdat[0])
            fft_indtqdat_normalized = fft_indtqdat/np.max(np.abs(fft_indtqdat))
            fft_corot_normalized = fft_corot/np.max(np.abs(fft_corot))


            self.freq = freq
            self.fft_indtqdat_normalized = fft_indtqdat_normalized
            self.fft_corot_normalized = fft_corot_normalized


            # ! Le xlim de la freq ?
            threshold = 0.01
            maxfreq_indtqdat = freq[np.argmin(np.abs(np.abs(fft_indtqdat_normalized)-threshold))]
            maxfreq_corot = freq[np.argmin(np.abs(np.abs(fft_corot_normalized)-threshold))]
            self.maxfreq = np.max([np.abs(maxfreq_indtqdat), np.abs(maxfreq_corot)])
            if self.maxfreq<1.2 and self.typeII:
                self.maxfreq = 1.2

    def init_fig(self, id, mosaic=[['single']], num='', **kwargs):
        if num == '':
            num=f'{self.directory} {id}'
        base = dict(num=num, layout='constrained')
        fig, axs = plt.subplot_mosaic(
                mosaic, 
                **base,
                **kwargs,)
        
        fig.myfigid = id
        self.FIGS[id] = {   "fig": fig,
                            "num": num,
                            "uplots": [],
                            "uartists": []
                        }

        if mosaic == [['single']]:
            return fig, axs['single']
        else:
            return fig,axs

    def plot_tot_only(self, fig, ax):
        # ! Total
        Tot_uplot = MyUpdatedPlot(
            mytorque    = self,
            fig         = fig,
            ax          = ax,
            dim         = "2D",
            xlabel      = r'$t$ [orbits]',
            ylabel      = r'$\Gamma(t)/\Gamma_0$',
            title       = "",
            ylims       = (-1.5, 1)
        )
        Tot_uplot.ax.set_xlabel(r'$t$ [orbits]', fontsize=35)
        Tot_uplot.ax.set_ylabel(r'$\Gamma(t)/\Gamma_0$', fontsize=35)
        Tot_uplot.ax.tick_params(labelsize=35)
        rescaling_factor = 10
        # Tot_uplot.ax.plot(self.time_dirtqdat, self.dirtqdat - self.dirtqdat[-1], label=r"$(\Gamma_{\rm{dir}}-\lim \Gamma_{\rm{dir}})/\Gamma_0$", color="navy")

        # Tot_uplot.ax.plot(self.time_dirtqdat, self.dirtqdat - self.dirtqdat[-1], label=r"$\Gamma_{\rm{corot}}/\Gamma_0$", color="navy")
        # Tot_uplot.ax.plot(self.time_indtqdat, self.indtqdat*rescaling_factor, label=fr"${rescaling_factor}$" + r"$\times\Gamma_{\rm{ind}}/\Gamma_0$", color="darkorange")

        Tot_uplot.ax.plot(self.time_dirtqdat, self.dirtqdat, label=r"$\Gamma_{\rm{dir}}/\Gamma_0$", color="blue")
        Tot_uplot.ax.plot(self.time_indtqdat, self.indtqdat, label=r"$\Gamma_{\rm{ind}}/\Gamma_0$", color="orange")


        Tot_uplot.ax.axhline(y=0,xmin=-50,xmax=520, color="black", alpha=0.6)
        if self.directory=="out_crida1_novisco_f05_RES":
            Tot_uplot.ax.set_xticks([10,100,200,300,400,500],["$10$","$100$","$200$","$300$","$400$","$500$"])
            Tot_uplot.ax.set_xlim(10, 500)
        # MULTI_tot_uplot.add_dot(self.Mytime, self.Dtorque_tot)
        # MULTI_tot_uplot.add_dot(self.Mytime, self.Itorque_tot)
                # # ! Insert Plot
        # axins = Tot_uplot.ax.inset_axes([0.4, 0.15, 0.5, 0.25], zorder=50)
        # axins.plot(self.freq, np.abs(self.fft_indtqdat_normalized),"+", label=r"$FT(\Gamma_{\rm{ind}})$", color="darkorange")
        # axins.plot(self.freq, np.abs(self.fft_corot_normalized), "x", label=r"$FT(\Gamma_{\rm{corot}})$", color="navy")
        # axins.set_xlabel(r"$\nu$ [orbits$^{-1}$]", fontsize = 25)
        # axins.set_ylabel(r"$|\rm{FFT}(\Gamma)|$", fontsize = 25)
        # axins.set_xticks([0,0.01,0.02,0.03,0.04,0.05],[r"$0$",r"$0.01$","$0.02$",r"$0.03$",r"$0.04$",r"$0.05$"])
        # axins.tick_params(labelsize=27)
        # # axins.yticks_params(fontsize=12)
        # axins.set_xlim(0, 0.05)
        # axins.set_ylim(0, top=1.1)

        # Tot_uplot.ax.scatter(self.time_dirtqdat[1:-1], self.rta_Dtorque,label=r"$\frac{1}{t}\int_0^t \Gamma_{dir}(t^\prime)dt^\prime /\Gamma_0$", alpha=0.4, marker="s",color="blue")
        # Tot_uplot.ax.scatter(self.time_indtqdat[1:-1], self.rta_Itorque,label=r"$\frac{1}{t}\int_0^t \Gamma_{ind}(t^\prime)dt^\prime /\Gamma_0$", alpha=0.4, marker="s",color="orange")

        Tot_uplot.lock(fontsize=30)

        return Tot_uplot


    def plot_rad_only(self, fig, ax):
        if self.typeI:
            xlims = (0.1,4)
            ylims = (-3,3)
            loc = "lower left"
        else:
            xlims = (0.2,4)
            ylims = (-2,1.5)
            loc = "upper right"
        Rad_rad_uplot = MyUpdatedPlot(
            mytorque    = self,   
            fig         = fig,
            ax          = ax,
            dim         = "2D",
            xlabel      = r'$r/r_{\rm p}$',
            ylabel      = r'$\Gamma^\prime(r)/\Gamma_0$',
            title       = "",
            ylims       = ylims,
            xlims       = xlims,
            )
        Rad_rad_uplot.ax.set_xscale("log")


        Rad_rad_uplot.ax.axvline(x=1, ymin=-10, ymax=10, color="black", alpha=0.6)
        Rad_rad_uplot.ax.axhline(y=0, xmin=-10, xmax=10, color="black", alpha=0.6)
        Rad_rad_uplot.add_line(
            xdata = self.dens.rmed,
            ydata = self.Dtorque_radial_density,
            label = r"$\Gamma_{\rm{dir}}^\prime(r)/\Gamma_0$",
            style = dict(color="blue", linestyle="dotted")
        )
        Rad_rad_uplot.add_line(
            xdata = self.dens.rmed,
            ydata = self.Itorque_radial_density,
            label = r"$\Gamma_{\rm{ind}}^\prime(r)/\Gamma_0$",
            style = dict(color="orange", linestyle="dotted")
        )
        Rad_rad_uplot.add_line(
            xdata = self.dens.rmed,
            ydata = self.Dtorque_radial_cumu,
            label = r"$\int_0^r \Gamma_{\rm{dir}}^\prime(x){\rm d}x/\Gamma_0$",
            style = dict(color="blue", linestyle="-")
        )
        Rad_rad_uplot.add_line(
            xdata = self.dens.rmed,
            ydata = self.Itorque_radial_cumu,
            label = r'$\int_0^r \Gamma_{\rm{ind}}^\prime(x){\rm d}x / \Gamma_0$',
            style = dict(color="orange", linestyle="-")
        )
        Rad_rad_uplot.add_frn()
        Rad_rad_uplot.lock(loc=loc)
        
        return Rad_rad_uplot

    def plot_iod_only(self, fig, ax): 
        MULTI_iod_uplot = MyUpdatedPlot(
            mytorque    = self,
            fig         = fig,
            ax          = ax,
            dim         = "2D",
            xlabel      = r'$t$ [orbits]',
            ylabel      = r'$\Gamma_{\rm{ind}}/\Gamma_{\rm{dir}}$',
            title       = f'',
            # ylims       = (-1,1)
        )
        # IOD = self.indtqdat/self.dirtqdat
        IOD = self.rta_Itorque/self.rta_Dtorque
        MULTI_iod_uplot.ax.plot(self.time_dirtqdat[1:-1], IOD, color="black", alpha=0.3, label=r'$\Gamma^{\rm rta}_{\rm{ind}}/\Gamma^{\rm rta}_{\rm{dir}}$')
        # MULTI_iod_uplot.ax.plot(self.time_dirtqdat, IOD, color="black", alpha=0.3, label=r'$\Gamma_{\rm{ind}}/\Gamma_{\rm{dir}}$')
        # MULTI_iod_uplot.add_dot(self.Mytime, self.Itorque_tot/self.Dtorque_tot)
        MULTI_iod_uplot.lock()

        return MULTI_iod_uplot

    def multiplot(self):
        self.fig_MULTI, self.axs_MULTI = self.init_fig(
                id="MULTI",
                num=f"{self.directory} MULTI",
                mosaic =[['top left', 'right'],['bottom left','right']], 
                figsize=(20,10))
        # ! Total
        MULTI_tot_uplot = self.plot_tot_only(fig=self.fig_MULTI, ax=self.axs_MULTI['top left'])

        # ! Radial
        MULTI_rad_uplot = self.plot_rad_only(fig=self.fig_MULTI, ax=self.axs_MULTI['right'])

        # ! ind over dir
        MULTI_iod_uplot = self.plot_iod_only(fig=self.fig_MULTI, ax=self.axs_MULTI['bottom left'])

    def plot_FT(self):
        self.fig_FT, self.axs_FT = plt.subplot_mosaic([['top left', 'top right'],['bottom left','right'],['bottom left','bottom right']],num=f"{self.directory} FT", figsize=(20, 10), layout='constrained',height_ratios=[2,1, 1])
        axs = self.axs_FT
        freq = self.freq
        self.fig_FT.myfigid="FT"
        self.FIGS["FT"] = {  "fig": self.fig_FT,
                                    "uplots":[],
                                    "uartists":[]
                                    }

        corot_or_dir = "corot"
        if corot_or_dir == "dir":
            corot = self.dirtqdat 
            title = r'$\Gamma_{\rm{dir}}$ et $\Gamma_{\rm{ind}}$ with time'
            xlabel = r"$\Gamma_{\rm{dir}}/\Gamma_0$"
        else:
            corot = self.corot
            title = r'$\Gamma_{\rm{corot}}$ et $\Gamma_{\rm{ind}}$ with time'
            xlabel = r"$\Gamma_{\rm{corot}}/\Gamma_0$"

        indtqdat = self.indtqdat
        corot_cut = self.corot[self.startindex_dat:]
        indtqdat_cut = self.indtqdat[self.startindex_dat:]
        time = self.time_dirtqdat
        # time_cut = self.time_dirtqdat[self.startindex_dat:]
        # ! Plot corot and indtqdat
        FT_tot_uplot = MyUpdatedPlot(
            mytorque    = self,
            fig         = self.fig_FT,
            ax          = self.axs_FT["top left"],
            dim         = "2D",
            xlabel      = r'$t$ [orbits]',
            ylabel      = r'$\Gamma(t)/\Gamma_0$',
            title       = title
        )
        FT_tot_uplot.ax.plot(time, corot,"-x", alpha=0.4, label=r"$\Gamma_{\rm{corot}}/\Gamma_0$", color="navy")
        FT_tot_uplot.ax.plot(time, indtqdat ,"-x", alpha=0.4, label=r"$\Gamma_{\rm{ind}}/\Gamma_0$", color="orange")
        FT_tot_uplot.ax.axvline(x=time[self.startindex_dat], color='red', linestyle='--', label='The left side is ignored.')
        FT_tot_uplot.lock()


        # ! Plot indtq function of corot
        FT_fit_uplot = MyUpdatedPlot(
            mytorque    = self,
            fig         = self.fig_FT,
            ax          = self.axs_FT["bottom left"],
            dim         = "2D",
            xlabel      = xlabel,
            ylabel      = r"$\Gamma_{\rm{ind}}/\Gamma_0$",
            title       = r'$\Gamma_{\rm{ind}}$ en fonction de $\Gamma_{\rm{corot}}$'
        )

        FT_fit_uplot.ax.plot(corot_cut, indtqdat_cut, "-")
        (a,b), pcov = np.polyfit(corot_cut, indtqdat_cut, 1, cov=True)
        u_a, u_b = np.sqrt(np.diag(pcov))
        FT_fit_uplot.ax.plot(corot_cut, a*corot_cut+b, "-", label=f"$\\Gamma_{{\\rm ind}}=({a:.3f}\\pm{u_a:.3f})\\Gamma_{{\\rm {corot_or_dir}}}+{b:.3f}\\pm {u_b:.3f}$", color="black")
        FT_fit_uplot.ax.axis('equal')

        # ! Plot FFT
        fonctions = [np.abs, np.real, np.imag]
        ylabels = [r"$|FFT(\Gamma)|$", r"$\Re(FFT(\Gamma))$", r"$\Im(FFT(\Gamma))$"]
        positions = ["top right", "right", "bottom right"]

        for i in range(3):
            f = fonctions[i]
            ylabel = ylabels[i]
            pos = positions[i]
            axs[pos].plot(freq, f(self.fft_indtqdat_normalized), "-x", alpha=0.6, label=r"$FT(\Gamma_{\rm{ind}})$", color="orange")
            axs[pos].plot(freq, f(self.fft_corot_normalized), "-x", alpha=0.6, label=r"$FT(\Gamma_{\rm{corot}})$", color="navy")
            axs[pos].set_xlabel(r"$f$ [Orbit$^{-1}$]")
            axs[pos].set_ylabel(f"{ylabel} [normalized]")
            axs[pos].set_xlim(0, self.maxfreq)
            axs[pos].set_ylim(top=1.1)

        FT_fit_uplot.lock()

    def plot_map(self):

        self.fig_map, self.ax_map = self.init_fig(
            id="map",
            num=f"{self.directory} map {cmap}",
            figsize=(10,10)
        )
        # plt.subplots(num=num_map, figsize=(10, 10), layout='constrained')
        # self.fig_map.subplots_adjust(top=0.9, bottom=0, left=0, right=1, wspace=0, hspace=0)
        self.ax_map.set_box_aspect(1)

        MAP_uplot = MyUpdatedPlot(
            mytorque    = self,
            fig         = self.fig_map,
            ax          = self.ax_map,
            dim         = "3D",
            xlabel      = r'$x/r_{\rm p}$',
            ylabel      = r'$y/r_{\rm p}$',
            title       = r"Normalized density $\frac{\Sigma(\mathbf{r},t)}{\Sigma(\mathbf{r}, 0)}$"
        )
        if self.typeII:
            vmin, vmax = 0,2
        else:
            vmin, vmax = 1-rangevminvmax/10,1+rangevminvmax/10
            # vmin, vmax = 0.995, 1.005
        MAP_uplot.add_map(self.X_GRID,
                self.Y_GRID,
                self.Density_norm,
                cmap=cmap,
                cbar=True,
                vmin=vmin,
                vmax=vmax)


        MAP_uplot.add_frn(xytext=(-10,-20), ha='right', pos=(1, 1))

        norm_arrow_itd = np.max(np.linalg.norm(self.IndirectForce, axis=1))
        norm_arrow_dir = np.max(np.linalg.norm(self.DirectForce, axis=1))
        arrow_norm = np.max((norm_arrow_itd,norm_arrow_dir))
        arrow_norm = 2 * 10 ** (np.floor(np.log10(arrow_norm))) # arrondi à la puissance inférieure
        # self.arrow_norm = norm_arrow_itd
        print(f"Arrow norm : {arrow_norm:.1e}")
        # self.arrow_norm = 1e-7

        if self.typeI:
            itxt_dx=0.1
            itxt_dy=0.1
            dtxt_dx=-0.1
            dtxt_dy=-0.15
        else:
            itxt_dx=0.1
            itxt_dy=0.2
            dtxt_dx=-0.1
            dtxt_dy=-0.1

        MAP_uplot.add_arrow(0* self.IndirectForce, - self.IndirectForce/arrow_norm,text=r"$\mathbf a_{\star,\rm d}$", arrowprops=dict(facecolor="orange"), txt_dx=itxt_dx, txt_dy=itxt_dy)

        MAP_uplot.add_arrow(np.column_stack([self.Xpla,self.Ypla]), self.DirectForce/arrow_norm,text=r"$\mathbf{a}_{\rm{dir}}$", arrowprops=dict(facecolor="blue"), txt_dx=dtxt_dx, txt_dy=dtxt_dy)
        if self.typeI:
            MAP_uplot.ax.set_xlim(-1.5,3.5)
            MAP_uplot.ax.set_ylim(-3.5,1.5)
        else:
            MAP_uplot.ax.set_xlim(-3,3)
            MAP_uplot.ax.set_ylim(-3,3)

        MAP_uplot.lock()


    def plot(self):
        do_tot = self.plot_settings["tot"]
        do_rad = self.plot_settings["rad"]
        do_MULTI = self.plot_settings["MULTI"]
        do_FT = self.plot_settings["FT"]

        if do_MULTI:
            self.multiplot()

        if do_tot:
            self.fig_tot, self.ax_tot = self.init_fig(
                id="tot",
                figsize=(15,10))

            self.plot_tot_only(self.fig_tot, self.ax_tot)
        if do_rad:
            self.fig_rad, self.ax_rad = self.init_fig(
                id="rad",
                figsize=(10,10)
            )
            self.plot_rad_only(self.fig_rad, self.ax_rad)
                
        if self.plot_settings["FT"]:
            self.plot_FT()
        if self.plot_settings["map"]:
            self.plot_map()

        for myfigid in self.FIGS:
            fig = self.FIGS[myfigid]["fig"]
            fig.canvas.mpl_connect('key_press_event', self.on_press)
            self.FIGS[myfigid]["bm"] = BlitManager(fig.canvas, self.FIGS[myfigid]["uartists"])



    def on_press(self,event):
        if event.key == 'right':
            self.curr_on = self.on_start + (self.curr_on - self.on_start + 1) % self.loop_length
        elif event.key == 'left':
            self.curr_on = self.on_start + (self.curr_on - self.on_start - 1) % self.loop_length
        elif event.key == 'up':
            self.curr_on = self.on_end - 1
        elif event.key == 'down':
            self.curr_on = 0

        if event.key in ['right', 'left', 'up', 'down']:
            self.update_plot(self.curr_on)


    def update_plot(self, i):
        self.current_orbit = round(self.Mytime[self.curr_on])
        for fig_id, setting in self.plot_settings.items():
            if setting and setting!='no update':
                uplots = self.FIGS[fig_id]["uplots"]
                for uplot in uplots:
                    uplot.update()
                if self.funcanim_map==False:
                    bm = self.FIGS[fig_id]["bm"]
                    bm.update()

        if self.funcanim_map:
            return self.FIGS["map"]["uartists"]




if dark_mode:
    # transparent = True

    # matplotlib.pyplot.rcParams['axes.labelcolor'] = "white"
    # matplotlib.pyplot.rcParams['xtick.color'] = "white"
    # matplotlib.pyplot.rcParams['ytick.color'] = "white"
    plt.style.use('dark_background')
    matplotlib.pyplot.rcParams['axes.facecolor'] = "white"
    matplotlib.pyplot.rcParams['text.color'] = "white"
matplotlib.pyplot.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({"font.weight": "bold"})
matplotlib.rcParams.update({"axes.labelweight": "bold"})

if __name__ == "__main__":

    import gc




    # for index_cmap in range():

    gc.collect()
    plt.close()
    # mytorque = MyTorque(plot_settings, on_start=-1, ignore_start_for_fft=400)
    mytorque = MyTorque(plot_settings, on_start=-1, ignore_start_for_fft=10)
    if not oneshot:
        if not os.path.exists(f"./output_imgs"):
            os.mkdir(f"./output_imgs")
        if plot_settings["map"]:
            name = f"./output_imgs/{mytorque.FIGS['map']['num']}"
            if dark_mode:
                name+="-dark"
            name+=".pdf"
            mytorque.fig_map.savefig(name, dpi=600, bbox_inches='tight', transparent=transparent)
        if plot_settings["rad"]:
            name = f"./output_imgs/{mytorque.FIGS['rad']['num']}"
            if dark_mode:
                name+="-dark"
            name+=".pdf"
            mytorque.fig_rad.savefig(name, dpi=600, bbox_inches='tight', transparent=transparent)
        if plot_settings["tot"]:
            name = f"./output_imgs/{mytorque.FIGS['tot']['num']}"
            if dark_mode:
                name+="-dark"
            name+=".pdf"
            mytorque.fig_tot.savefig(name, dpi=600, bbox_inches='tight', transparent=transparent)
    else:
        print(mytorque.FIGS.keys())
        plt.show()
