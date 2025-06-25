import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

plt.figure()
plt.plot([0,1],[0,1], label=r'$\mathbf{r}$')
plt.legend(prop={'size': 35})
plt.savefig("trash.png")
