import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# import local_conditions as local
import sys

folder = str(sys.argv[1])
# folder = 'bessel'
# folder = 'stroemgren'
# folder = 'sloan'
if folder == "bessel":
    bands = ["U", "B", "V", "R", "I"]
elif folder == "sloan":
    bands = ["u", "g", "r", "i", "z"]
elif folder == "stroemgren":
    bands = ["u", "v", "b", "y"]
elif folder == "wircam":
    bands = ["Y", "J", "H"]  # , 'Ks']
elif folder == "panstarrs":
    bands = ["w", "g", "r", "i", "z", "y"]
elif folder == "des":
    bands = ["u", "g", "r", "i", "z", "y"]
elif folder == "gft":
    bands = ["g", "r", "i", "z", "y", "J", "H"]

for band in bands:
    print(band)
    xx = []
    yy = []
    line2 = 0
    filename = band + ".txt"
    with open("%s/%s" % (folder, filename), "r") as file:
        for line in file:
            if line[0] != "#" and len(line) > 3:
                a, b = line.split()
                xx.append(float(a))
                yy.append(float(b))

    ang = np.array(xx)  # Angstrom
    Trans = np.array(yy)
    if folder == "gft":
        Trans = Trans * 1e2  #
        ang *= 10

    plt.plot(ang, Trans, label=band, lw=1.5)

plt.ylim(0.0, 100)
plt.xlabel(r"$\lambda$ (Angstroms)", fontsize=15)
plt.xlim(3500, 20000)
plt.ylabel("Transmission (%)", fontsize=15)
plt.legend(loc="upper right")
# plt.title('%s' % folder)
plt.grid(True)
plt.savefig("filter_%s.png" % folder)
plt.show()
