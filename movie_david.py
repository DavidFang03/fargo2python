from plot_davidv2 import MyTorque
import matplotlib.animation as animation


dark_mode = True
funcanim_map=True
transparent = False
oneshot = False

cmap = 'RdBu_r'

plot_settings =     {"tot": False,
                        "map": True,
                        "rad": False,
                        "MULTI": False,
                        "FT": False}
mytorque = MyTorque(plot_settings, on_start=0)

def iterate(i):
    print(i)
    mytorque.curr_on = i
    return mytorque.update_plot(i)

ani = animation.FuncAnimation(mytorque.FIGS["map"]["fig"], iterate, frames=mytorque.nb_outputs, 
                        interval=1, blit=True, repeat=False)

videopath = f"./output_vids/{mytorque.FIGS["map"]["num"]}-dark"
ani.save(f"{videopath}.mp4")
