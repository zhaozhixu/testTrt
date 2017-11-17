import numpy as np

E = 2.718281828
def func(d,a):
    cx = a[0] + d[0] * a[2]
    cy = a[1] + d[1] * a[3]
    if d[2] < 1:
        w = a[2] * np.exp(d[2])
    else:
        w = a[2] * d[2] * E
    if d[3] < 1:
        h = a[3] * np.exp(d[3])
    else:
        h = a[3] * d[3] * E

    # print [cx, cy, w, h]
    res = []
    res.append(cx - w / 2)
    res.append(cy - h / 2)
    res.append(cx + w / 2)
    res.append(cy + h / 2)
    return res

print func([0.0,0.1,0.2,0.3], [0.0,0.1,0.2,0.3])
print func([0.4,0.5,0.6,0.7], [0.4,0.5,0.6,0.7])
print func([0.8,0.9,0.10,0.11], [0.8,0.9,0.10,0.11])
print func([0.12,0.13,0.14,0.15], [0.12,0.13,0.14,0.15])
print func([0.16,0.17,0.18,0.19], [0.16,0.17,0.18,0.19])
print func([0.20,0.21,0.22,0.23], [0.20,0.21,0.22,0.23])
