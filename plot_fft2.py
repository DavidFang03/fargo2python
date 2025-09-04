import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata

import modules.tools_plot as tools_plot 

from BLIT import BlitManager
from david_tools import *

dark_mode = True

rangevminvmax = 1
plot_settings =     {"tot": False,
                        "map": True,
                        "rad": False,
                        "MULTI": False,
                        "FT": False}

transparent = False
oneshot = False
ext = ".png"

class MyUpdatedPlot():
    def __init__(   self, 
                    myclass,
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
        self.myclass = myclass
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=20)
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)

        self.curr_on = myclass.curr_on
        self.current_orbit = myclass.current_orbit


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

        myclass.FIGS[fig.myfigid]["uplots"].append(self)

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
                    Pos,
                    Delta,
                    text="",
                    txt_dx=0.1,
                    txt_dy=0.1,
                    **kwargs):

        '''
        Both pos and delta must change with time.
        '''
        k = self.curr_on
        self.txt_dx, self.txt_dy = txt_dx, txt_dy
        xy, xytext, postext = tools_plot.get_coords_for_arrow(Pos[k,0:2], Delta[k,0:2], self.txt_dx, self.txt_dy)


        kwargs["arrowprops"]["arrowstyle"]="simple"
        kwargs["arrowprops"]["edgecolor"]="white"

        if "lw" not in kwargs["arrowprops"] or "linewidth" not in kwargs["arrowprops"]:
            kwargs["arrowprops"]["lw"]=0.5

        # arrow = self.ax.annotate(
        #             "",
        #             xy=xy,
        #             xytext=xytext,
        #             annotation_clip=False,
        #             **kwargs,
        #             )

        arrow = FancyArrowPatch(posA=xytext,posB=xy,clip_on=True, mutation_scale=20,**kwargs["arrowprops"]) 
        self.ax.add_patch(arrow)

        text = self.ax.text(*postext, s=text, va='center', ha='center', color=arrow.get_facecolor(), clip_on=True)


        self.arrows.append(arrow)
        self.arrows_text.append(text)
        self.arrows_pos.append(Pos)
        self.arrows_delta.append(Delta)


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
            color=frn_color)
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
        self.myclass.FIGS[self.fig.myfigid]["uartists"] +=  artists


    def update(self):
        self.curr_on = self.myclass.curr_on
        self.current_orbit = self.myclass.current_orbit
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
            Pos = self.arrows_pos[i]
            Delta = self.arrows_delta[i]
            xy, xytext, postext = tools_plot.get_coords_for_arrow(Pos[k,0:2], Delta[k,0:2], self.txt_dx, self.txt_dy)


            arrow.set_positions(xytext, xy) # arrow.xytext = machin ne fait rien
            # arrow.xy = xy
            text.set_position(postext)

        if len(self.frn) > 0:
            frn = self.frn[0]
            frn.set_text(r"$t={}$ orbits".format(self.current_orbit))







class MyFFT2():
    def __init__(self, plot_settings, **kwargs):
        '''
        Possible kwargs:
        on_start (int) and on_end (int) : range of output numbers (change it for debug so it loops faster). Default : first and last output of the directory.
        init_on (int) : initial time (if you want to initially show the data of output $n$ e.g.). Default : 'on_end'.
        '''
        import par
        if isinstance(par.directory, list):
            self.directory = par.directory[0]
        else:
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

        self.funcanim_map = kwargs.get("funcanim_map", False)


        # ! Grid infos
        self.dens0 = get_dens(self.on,0,par,self.directory)
        self.vphi0 = get_vphi(self.on,0,par,self.directory)
        self.nrad, self.nsec = self.dens0.nrad, self.dens0.nsec
        self.surface, self.dr, self.dphi, self.rdphi = get_grid(self.dens0)
        self.vecpos_cells = get_vecpos_cells(self.dens0) 
        self.rmin, self.rmax = self.dens0.redge[0], self.dens0.redge[-1]
        self.pmed_deg = self.dens0.pmed * 180.0 / np.pi

        self.PHI, self.R = np.array(self.dens0.pmed), np.array(self.dens0.rmed)
        self.PHI_meshgrid, self.R_meshgrid = np.meshgrid(self.dens0.pmed, self.dens0.rmed)

        nx = int(self.nsec/2)
        ny = int(self.nrad/2)
        self.xi = np.linspace(min(self.PHI), max(self.PHI), nx)
        self.yi = np.linspace(min(self.R), max(self.R), ny)
        self.X_reg, self.Y_reg = np.meshgrid(self.xi, self.yi)

        fx = np.fft.fftfreq(nx, d=(self.xi[1] - self.xi[0]))
        fy = np.fft.fftfreq(ny, d=(self.yi[1] - self.yi[0]))
        self.FX, self.FY = np.meshgrid(fx, fy)

        # ! Planet infos
        self.Xpla, self.Ypla, self.Mpla, self.Date, self.Omegapla = get_pla(self.dens0, self.directory, self)

        self.apla = get_apla(self.directory)
        if np.isnan(self.apla):
            self.apla = 1
        self.t_orb = 2.0 * np.pi * (self.apla**1.5)
        self.Mytime = self.Date / self.t_orb
        self.dt = np.diff(self.Mytime)

        self.current_orbit = round(self.Mytime[self.curr_on])



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

        # Text simulation parameters
        self.text_sim_infos = '\n'.join([f'{self.sim_infos_dic[key]["symbol"]} = {self.sim_infos_dic[key]["value"]}' for key in self.sim_infos_dic.keys()])

        # ! Plot parameters
        self.PHI_GRID, self.R_GRID = np.meshgrid(self.dens0.pedge, self.dens0.redge)

        self.X_GRID = self.R_GRID * np.cos(self.PHI_GRID)
        self.Y_GRID = self.R_GRID * np.sin(self.PHI_GRID)


        self.vmin_pert_dens,self.vmax_pert_dens = -1,1
        self.vmin_pert_vphi,self.vmax_pert_vphi = -1,1

        # # ! Init data
        self.Density_norm = np.zeros((self.nb_outputs, self.dens0.nrad*self.dens0.nsec))
        self.Density_norm_FFT = [None for _ in range(self.nb_outputs)]

        self.Perturbed_density = np.zeros((self.nb_outputs, self.dens0.nrad*self.dens0.nsec))
        self.Perturbed_vphi = np.zeros((self.nb_outputs, self.vphi0.nrad*self.vphi0.nsec))


        # ! To be updated artistis :
        self.FIGS = {}

        # ! RUN
        self.loop()
        self.plot()


    def loop(self):
        import par
        for k in range(self.on_start, self.on_end +1):
            print('output number =', str(k), 'out of', str(len(self.on)), end='\r')
            # print('output number =', str(k), 'out of', str(len(self.on)))
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

            dens_norm = dens.data/self.dens0.data

            dens_norm_interp = griddata((self.PHI, self.R), dens_norm, (self.X_reg, self.Y_reg), method='cubic')
            dens_norm_fft = np.fft.fft2(dens_norm_interp)

            self.Density_norm[k,:]  = dens_norm.ravel()
            self.Density_norm_FFT[k,:]  = dens_norm_fft.ravel()
            self.Perturbed_density[k,:] = (self.dens.data[:,:]/np.average(self.dens.data[:,:]) - 1).ravel()
            self.Perturbed_vphi[k,:] = (self.vphi.data[:,:]/np.average(self.vphi.data) - 1).ravel()



    

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


    def plot_map(self):

        self.fig_map, self.axs_map = self.init_fig(
            mosaic=[['left','right']],
            id="map",
            num=f"{self.directory} map {cmap}",
            figsize=(10,10)
        )
        # plt.subplots(num=num_map, figsize=(10, 10), layout='constrained')
        # self.fig_map.subplots_adjust(top=0.9, bottom=0, left=0, right=1, wspace=0, hspace=0)
        axl = self.axs_map['left']
        axr = self.axs_map['right']
        self.ax_map.set_box_aspect(1)

        MAP_uplot = MyUpdatedPlot(
            myclass    = self,
            fig         = self.fig_map,
            ax          = axl,
            dim         = "3D",
            xlabel      = r'$x/r_{\rm p}$',
            ylabel      = r'$y/r_{\rm p}$',
            title       = r"Normalized density $\frac{\Sigma(\mathbf{r},t)}{\Sigma(\mathbf{r}, 0)}$"
        )
        MAP_FFT_uplot = MyUpdatedPlot(
            myclass    = self,
            fig         = self.fig_map,
            ax          = axr,
            dim         = "3D",
            xlabel      = r'??',
            ylabel      = r'??',
            title       = r"FFT of $\frac{\Sigma(\mathbf{r},t)}{\Sigma(\mathbf{r}, 0)}$"
        )

        vminfft, vmaxfft = 0,2

        MAP_FFT_uplot.add_map(self.FX,
                self.FY,
                self.Density_norm_FFT,
                cmap=cmap,
                cbar=True,
                vmin=vminfft,
                vmax=vmaxfft)


        MAP_uplot.add_frn(xytext=(-10,-20), ha='right', pos=(1, 1))

        MAP_uplot.lock()


    def plot(self):
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




cmap = 'RdBu_r'
frn_color = 'black'
if dark_mode:
    # transparent = True

    # matplotlib.pyplot.rcParams['axes.labelcolor'] = "white"
    # matplotlib.pyplot.rcParams['xtick.color'] = "white"
    # matplotlib.pyplot.rcParams['ytick.color'] = "white"
    plt.style.use('dark_background')
    matplotlib.pyplot.rcParams['axes.facecolor'] = "white"
    matplotlib.pyplot.rcParams['text.color'] = "white"
    matplotlib.pyplot.rcParams['axes.facecolor'] = 'black'
    cmap = 'berlin'
    frn_color = 'white'

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
    myFFT2 = MyFFT2(plot_settings, on_start=150, on_end=155 )
    print(myFFT2.FIGS.keys())
    plt.show()
