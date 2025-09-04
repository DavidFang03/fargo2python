from plot_davidv2 import MyTorque
import matplotlib.animation as animation
import gc



print("coucou")

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
mytorque = MyTorque(plot_settings, funcanim_map=True)
# mytorque = MyTorque(plot_settings, on_start=0, on_end=1)

def iterate(i):
    print(i)
    mytorque.curr_on = i + mytorque.on_start
    return mytorque.update_plot(i)

ani = animation.FuncAnimation(mytorque.FIGS['map']['fig'], iterate, frames=mytorque.loop_length, interval=1, blit=True, repeat=False)
# ani = animation.FuncAnimation(mytorque.FIGS['map']['fig'], iterate, frames=10, 
#                         interval=1, blit=True, repeat=False)

videopath = f"./output_vids/{mytorque.FIGS['map']['num'].replace(' ','-')}-dark"
ani.save(f"{videopath}.mp4", writer="ffmpeg", fps=6)

# ## SI TROP LOURD
# nb_outputs = mytorque.nb_outputs
# for i in range(nb_outputs):
#     mytorque = MyTorque(plot_settings, on_start=i, on_end=i)
#     mytorque.FIGS['map']['fig'].savefig(f"./output_movie/map_{i:05}.png", dpi=250)
#     gc.collect()

