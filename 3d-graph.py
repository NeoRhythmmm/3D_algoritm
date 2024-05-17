import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 7))

vertices = [
    [(-180, 20, 0), (180, 20, 0)], 
    [(-180, 20, 140), (180, 20, 140)], 
    [(-180, 20, 0), (-180, 20, 140)], 
    [(180, 20, 140), (180, 20, 0)],
    [(0, 20000, 0), (0, 20000, 140)],
    [(-180, 20, 140), (0, 20000, 140)],
    [(180, 20, 140), (0, 20000, 140)],
    [(-180, 20, 0), (0, 20000, 0)],
    [(180, 20, 0), (0, 20000, 0)]
]

ax.add_collection3d(Poly3DCollection(vertices, facecolors='white', linewidths=1, edgecolors='r', alpha=.25))

for v in vertices:
    for point in v:
        ax.text(*point, f'({point[0]}, {point[1]}, {point[2]})')

ax.set(xlabel='Ширина (градусы)', ylabel='Глубина (Hz)', zlabel='Высота (dB)',
       xlim=(-180, 180), ylim=(20, 20000), zlim=(0, 140))

plt.show()
