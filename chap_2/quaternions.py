# Quaternion : q = w + xi + yj + zk (i^2 = j^2 = k^2 = jk = -1 et ij = k, jk = i, ki = j)
# Algèbre associative mais non commutative
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, obj):
        if not isinstance(obj, Quaternion):
            obj = Quaternion(obj)
        return Quaternion(self.w + obj.w, self.x + obj.x, self.y + obj.y, self.z + obj.z)

    def __sub__(self, obj):
        return self + (-obj)

    def __mul__(self, obj):
        if isinstance(obj, Quaternion):
            w = self.w * obj.w - self.x * obj.x - self.y * obj.y - self.z * obj.z
            x = self.w * obj.x + self.x * obj.w + self.y * obj.z - self.z * obj.y
            y = self.w * obj.y - self.x * obj.z + self.y * obj.w + self.z * obj.x
            z = self.w * obj.z + self.x * obj.y - self.y * obj.x + self.z * obj.w
            return Quaternion(w, x, y, z)
        else:
            Quaternion(self.w * obj, self.x * obj, self.y * obj, self.z * obj)

    def __radd__(self, obj):
        return self + obj

    def __rsub__(self, obj):
        return -self + obj

    def __rmul__(self, obj):
        return self * obj

    def conjg(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def abs2(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def inv(self) -> Quaternion:
        return self.conjg() * (1 / self.abs2())

    def __truediv__(self, obj):
        if isinstance(obj, Quaternion):
            return self * obj.inv()
        else:
            return self * (1 / obj)

    def __rtruediv__(self, obj):
        return self.inv() * obj

    def normalize(self, tolerance=0.00001) -> Quaternion:
        abs2 = self.abs2()
        qn = self
        if abs(abs2 - 1.0) > tolerance:
            n = math.sqrt(abs2)
            qn = Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)
        return qn

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def rotate_vector(self, angle: float, axis: Quaternion) -> Quaternion:
        v = self.normalize()
        axis = axis.normalize()
        q = Quaternion(math.cos(angle / 2), angle * axis.x, angle * axis.y, angle * axis.z).normalize()
        qc = q.conjg()
        vp = (q * v * qc).normalize()
        return vp


if __name__ == "__main__":
    v = Quaternion(0, 0, 0, 1)
    axis = Quaternion(0, 0, 1, 0)
    angle = 2 * math.pi / 3
    vp1 = v.rotate_vector(angle, axis)
    vp2 = v.rotate_vector(-angle, axis)
    plt.show()
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial vector
    ax.quiver(0, 0, 0, v.x, v.y, v.z, color='r', label='Initial Vector')

    # Plot the rotated vector
    ax.quiver(0, 0, 0, vp1.x, vp1.y, vp1.z, color='b', label='Rotated Vector')
    ax.quiver(0, 0, 0, vp2.x, vp2.y, vp2.z, color='g', label='Rotated Vector')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation of Vector using Quaternion')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=float)

    # Define the faces of the cube
    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [0, 3, 7, 4]]
    ]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cube = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)

    ax.add_collection3d(cube)

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(frame):
        angle = frame * np.pi / 180
        axis = Quaternion(0, 1, 1, 1).normalize()
        rotated_faces = []
        for face in faces:
            rotated_face = []
            for vertex in face:
                v = Quaternion(0, vertex[0], vertex[1], vertex[2])
                rotated_vertex = v.rotate_vector(angle, axis).to_numpy()
                rotated_face.append(rotated_vertex)
            rotated_faces.append(rotated_face)

        cube.set_verts(rotated_faces)
        return cube

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    #ani.save('rotating_cube.gif', writer=PillowWriter(fps=20))
    plt.show()
