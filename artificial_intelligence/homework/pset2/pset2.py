import learnThu8
import numpy as np


def gd(fxn, dfxn, x0, step_size=.01, max_iter=1000, eps=0.00001):
    x = x0
    ylist = []
    xlist = []

    for i in range(max_iter):
        ylist.append(fxn(x))
        xlist.append(x)
        diff = step_size * dfxn(x)
        x -= diff

        if (np.absolute(diff) < eps).all():
            break

    return x, ylist, xlist


learnThu8.gd = gd

if __name__ == "__main__":
    learnThu8.t6()
