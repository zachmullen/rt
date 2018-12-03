import itertools
import math
import multiprocessing
import numpy as np
from PIL import Image

CHUNK_SIZE = 50  # scanlines
MAX_BOUNCES = 3
EPSILON = 1e-8


def render_block(block):  # multiprocess pool target, cannot be an instance method
    start, stop, scene = block
    length = stop - start
    img = np.empty((length, scene.camera.hres, 3), dtype=np.uint8)
    for i, j in itertools.product(range(scene.camera.hres), range(length)):
        img[j, i] = np.minimum(scene._color(scene.camera.cast(i, j+start)).astype(np.uint8), 255)
    return img


def solve_quadratic(a, b, c):
    discr = b * b - 4 * a * c
    if discr < 0:
        return None, None
    elif discr == 0:
        x0 = x1 = - 0.5 * b / a
    else:
        if b > 0:
            q = -0.5 * (b + math.sqrt(discr))
        else:
            q = -0.5 * (b - math.sqrt(discr))
        x0 = q / a
        x1 = c / q
    return sorted((x0, x1))


def refract(i, n, index):
    cosi = max(-1., min(1., i.dot(n)))  # clamp due to precision loss
    etai = 1
    etat = index

    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -n

    eta = etai / etat
    k = 1. - eta * eta * (1. - cosi * cosi);
    if k < 0:
        return None
    else:
        return eta * i + (eta * cosi - math.sqrt(k)) * n


class Shape(object):
    def refract(self, ray, p):
        n = self.surface_normal(p)
        refract_dir = refract(ray.dir, n, self.mat.r_index)
        if refract_dir is None:
            return None
        return Ray(p + EPSILON * refract_dir, refract_dir)


class PointLight(object):
    __slots__ = ['pos', 'intensity']

    def __init__(self, pos, intensity):
        self.pos = pos
        self.intensity = intensity

    def getdirs(self, p):
        v = self.pos - p
        l = np.linalg.norm(v)
        i = min(self.intensity / (math.pi * 4 * l * l), 1.)
        return [(v, l, i)]


class DirectionalLight(object):
    __slots__ = ['_neg_dir', 'intensity']

    def __init__(self, dir, intensity):
        self._neg_dir = -dir
        self.intensity = intensity

    def getdirs(self, p):
        return [(self._neg_dir, float('inf'), self.intensity)]


class Ray(object):
    __slots__ = ['orig', 'dir']

    def __init__(self, orig, dir, norm=True):
        self.orig = orig
        if norm:
            self.dir = dir / np.linalg.norm(dir)
        else:
            self.dir = dir


class Material(object):
    __slots__ = ['color', 'amb', 'diff', 'reflect', 'refract', 'r_index']

    def __init__(self, color, amb, diff, reflect, refract=0., r_index=1):
        self.color = color
        self.amb = amb
        # self.spec = spec
        self.diff = diff
        self.reflect = reflect
        self.refract = refract
        self.r_index = r_index


class Sphere(Shape):
    __slots__ = ['center', 'r_sq', 'mat']

    def __init__(self, center, r, mat):
        self.center = center
        self.r_sq = r * r
        self.mat = mat

    def intersect(self, ray):
        L = ray.orig - self.center
        a = ray.dir.dot(ray.dir)
        b = 2 * ray.dir.dot(L)
        c = L.dot(L) - self.r_sq
        t0, t1 = solve_quadratic(a, b, c)
        if t0 is None:
            return None

        if abs(t0) < EPSILON:
            return t1

        if t0 < 0:
            t0 = t1
            if t0 < 0:
                return None  # both intersection values are negative, sphere is behind camera

        return t0

    def surface_normal(self, p):
        v = p - self.center
        return v / np.linalg.norm(v)


class Plane(Shape):
    def __init__(self, p, normal, mat):
        self.p = p
        self.normal = normal / np.linalg.norm(normal)
        self.mat = mat

    def intersect(self, ray):
        num = (self.p - ray.orig).dot(self.normal)
        denom = ray.dir.dot(self.normal)
        if abs(denom) < EPSILON:
            return None
        t = num / denom
        return None if t < 0 else t

    def surface_normal(self, p):
        return self.normal


class Camera(object):
    def __init__(self, hres, vres, fov):  # fixed camera, fov is in degrees
        self.ar = hres / float(vres)
        self.hres = hres
        self.vres = vres

        self._tanterm = math.tan(fov / 2 * math.pi / 180)

    def cast(self, i, j):
        px = (2 * ((i + 0.5) / self.hres) - 1) * self._tanterm * self.ar
        py = (1 - 2 * ((j + 0.5) / self.vres)) * self._tanterm
        return Ray(np.array((0, 0, 0)), np.array((px, py, -1)))


class Scene(object):
    def __init__(self, camera, actors, lights, bg_color):
        self.camera = camera
        self.actors = actors
        self.lights = lights
        self.bg_color = bg_color

    def render(self):
        # returns a numpy array, the raytraced 2D image
        pool = multiprocessing.Pool()
        chunks = range(0, self.camera.vres, CHUNK_SIZE)
        blocks = [(i, min(i+CHUNK_SIZE, self.camera.vres), self) for i in chunks]
        rows = pool.map(render_block, blocks)
        return np.concatenate(rows)

    def _color(self, ray, ttl=MAX_BOUNCES):
        obj = None
        min_t = float('inf')

        for actor in self.actors:
            t = actor.intersect(ray)
            if t is None:
                continue
            if t < min_t:
                min_t = t
                obj = actor

        if obj is None:
            return self.bg_color
        else:
            p = ray.orig + ray.dir * min_t  # intersection with object
            return sum((
                (obj.mat.color * obj.mat.amb),
                self._diffuse(obj, p),
                self._reflect(obj, ray, p, ttl),
                self._refract(obj, ray, p, ttl)
            ))

    def _diffuse(self, obj, p):
        if obj.mat.diff <= 0:
            return np.array((0, 0, 0))

        # Cast a ray from current point to each light source
        for l in self.lights:
            # TODO blend multiple lights instead of just assuming we have only 1
            light_rays = [(Ray(p, dir), t, i) for dir, t, i in l.getdirs(p)]
            total = np.zeros([3])

            for ray, t, i in light_rays:
                total += i * self._light_or_shadow(ray, obj, p, t)
            return total / float(len(light_rays))

    def _light_or_shadow(self, ray, obj, p, light_t):  # helper for diffuse shading
        for actor in self.actors:
            t = actor.intersect(ray)
            if t is not None and EPSILON < t < light_t:
                return np.array((0, 0, 0))
        return obj.mat.diff * obj.mat.color * obj.surface_normal(p).dot(ray.dir)

    def _reflect(self, obj, ray, p, ttl):
        if ttl == 0 or obj.mat.reflect <= 0:
            return np.array((0, 0, 0))
        n = obj.surface_normal(p)
        reflect_dir = ray.dir - 2 * n * (ray.dir.dot(n))
        reflect_ray = Ray(p + reflect_dir * EPSILON, reflect_dir)
        return obj.mat.reflect * self._color(reflect_ray, ttl - 1)

    def _refract(self, obj, ray, p, ttl):
        if ttl == 0 or obj.mat.refract <= 0:
            return np.array((0, 0, 0))
        refract_ray = obj.refract(ray, p)
        if refract_ray is None:
            return np.array((0, 0, 0))
        return obj.mat.refract * self._color(refract_ray, ttl - 1)


if __name__ == '__main__':
    s1_mat = Material(np.array((255, 50, 50)), amb=0.2, diff=0.5, reflect=0.3)
    s2_mat = Material(np.array((30, 250, 120)), amb=0.2, diff=0.5, reflect=0.3)
    s3_mat = Material(np.array((255, 255, 255)), amb=0.1, diff=0.3, reflect=0.7)
    s4_mat = Material(
        np.array((255, 255, 255)), amb=0, diff=0, reflect=.15, refract=.9, r_index=1.3)
    s5_mat = Material(np.array((50, 50, 255)), amb=.2, diff=.8, reflect=0)

    floor_mat = Material(np.array((150, 150, 240)), amb=0.2, diff=0.5, reflect=0.3)

    s1 = Sphere(np.array((0, -1, -6)), 1, s1_mat)
    s2 = Sphere(np.array((-4, 0.4, -8)), 2, s2_mat)
    s3 = Sphere(np.array((4, 2, -10)), 2.5, s3_mat)
    s4 = Sphere(np.array((-1, -1, -4)), 0.8, s4_mat)
    s5 = Sphere(np.array((1.8, -1.6, -4)), 0.4, s5_mat)
    floor = Plane(np.array((0, -2, 0)), np.array((0, 1, 0)), floor_mat)

    light = PointLight(np.array((0, 0, -2.5)), 700)
    # light = DirectionalLight(np.array((0, -1, 0)))

    camera = Camera(hres=1024, vres=768, fov=65)
    scene = Scene(camera, [s1, s2, s3, s4, s5, floor], [light], np.array((0, 0, 0)))
    Image.fromarray(scene.render()).save('out.png')

    # Next up:
    # . Multiple lights
    # . Specular light
    # . Area light (disk? plane? buncha points?)
    # . Light color
    # . Textures
    # . Fresnel
