use image::{Rgb, RgbImage};
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

const EPSILON: f64 = 1e-8;
const MAX_HOPS: u8 = 5;

fn solve_quadratic(a: f64, b: f64, c: f64) -> (f64, f64) {
    let (x0, x1);
    let discr = b * b - 4. * a * c;

    if discr < 0. {
        return (f64::NAN, f64::NAN);
    } else if discr == 0. {
        x0 = -0.5 * b / a;
        x1 = x0;
    } else {
        let q = match b {
            b if b > 0. => -0.5 * (b + discr.sqrt()),
            _ => -0.5 * (b - discr.sqrt()),
        };
        x0 = q / a;
        x1 = c / q;
    }
    match (x0, x1) {
        (x0, x1) if x0 < x1 => (x0, x1),
        _ => (x1, x0),
    }
}

fn clamp_byte(v: f64) -> u8 {
    match v {
        v if v > 255. => 255,
        v if v < 0. => 0,
        _ => v as u8,
    }
}

fn scale_color(color: Rgb<u8>, scalar: f64) -> Rgb<u8> {
    Rgb([
        clamp_byte(color[0] as f64 * scalar),
        clamp_byte(color[1] as f64 * scalar),
        clamp_byte(color[2] as f64 * scalar),
    ])
}

#[derive(Copy, Clone)]
struct Material {
    color: Rgb<u8>,
    ambient: f64,
    diffuse: f64,
    reflect: f64,
}

trait Shape {
    fn surface_normal(&self, point: XYZ) -> XYZ;
    fn intersect(&self, ray: &Ray) -> Option<f64>;
    fn ambient_color(&self, p: XYZ) -> Rgb<u8>;
    fn material(&self, p: XYZ) -> Material;
}

struct Sphere {
    center: XYZ,
    r_sq: f64,
    mat: Material,
}

impl Shape for Sphere {
    fn surface_normal(&self, point: XYZ) -> XYZ {
        (point - self.center).norm()
    }

    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let l = ray.orig - self.center;
        let a = ray.dir.dot(ray.dir);
        let b = ray.dir.dot(l) * 2.;
        let c = l.dot(l) - self.r_sq;
        let (mut t0, t1) = solve_quadratic(a, b, c);
        if t0.is_nan() {
            return None;
        }
        if t0.abs() < EPSILON {
            return Some(t1);
        }
        if t0 < 0. {
            t0 = t1;
            if t0 < 0. {
                return None;
            }
        }

        if t0 < 0. {
            t0 = t1;
            if t0 < 0. {
                return None;
            }
        }
        Some(t0)
    }

    fn ambient_color(&self, _point: XYZ) -> Rgb<u8> {
        scale_color(self.mat.color, self.mat.ambient)
    }

    fn material(&self, _point: XYZ) -> Material {
        self.mat
    }
}

#[derive(Copy, Clone)]
struct XYZ {
    x: f64,
    y: f64,
    z: f64,
}

impl XYZ {
    fn len(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn norm(&self) -> XYZ {
        *self * (1. / self.len())
    }

    fn dot(&self, rhs: XYZ) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Add for XYZ {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for XYZ {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for XYZ {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

struct Ray {
    orig: XYZ,
    dir: XYZ,
}

struct Camera {
    hres: u32,
    vres: u32,
    ar: f64,
    tanterm: f64,
}

struct Scene {
    bg: Rgb<u8>,
    actors: Vec<Box<dyn Shape>>,
}

impl Scene {
    fn compute_color(&self, ray: &Ray, ttl: u8) -> Rgb<u8> {
        let mut min_t: f64 = f64::INFINITY;
        let mut nearest_idx: Option<usize> = Option::None;

        for (i, actor) in self.actors.iter().enumerate() {
            let intersection = actor.intersect(ray);
            match intersection {
                Some(t) => {
                    if t < min_t {
                        min_t = t;
                        nearest_idx = Some(i);
                    }
                }
                None => {}
            }
        }

        match nearest_idx {
            Some(i) => {
                let intersection_point = (ray.dir * min_t) + ray.orig;
                let actor = &self.actors[i];
                actor.ambient_color(intersection_point)
            }
            None => self.bg,
        }
    }
}

impl Camera {
    fn new(width: u32, height: u32, fov_degrees: f64) -> Camera {
        Camera {
            hres: width,
            vres: height,
            ar: width as f64 / height as f64,
            tanterm: (fov_degrees / 2. * PI / 180.).tan(),
        }
    }

    fn render(&self, scene: &Scene) -> Box<RgbImage> {
        let mut img = Box::new(RgbImage::new(self.hres, self.vres));
        for x in 0..self.hres {
            for y in 0..self.vres {
                let ray: Ray = self.cast(x, y);
                img.put_pixel(x, y, scene.compute_color(&ray, MAX_HOPS));
            }
        }
        img
    }

    // Compute the ray from the camera at the given pixel coords
    fn cast(&self, x: u32, y: u32) -> Ray {
        let px = (2. * (((x as f64) + 0.5) / (self.hres as f64)) - 1.) * self.tanterm * self.ar;
        let py = (1. - 2. * (((y as f64) + 0.5) / (self.vres as f64))) * self.tanterm;
        Ray {
            orig: XYZ {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            dir: XYZ {
                x: px,
                y: py,
                z: -1.,
            }
            .norm(),
        }
    }
}

fn main() -> Result<(), image::ImageError> {
    let camera = Camera::new(100, 100, 60.);
    let mut actors: Vec<Box<dyn Shape>> = Vec::new();
    actors.push(Box::new(Sphere {
        center: XYZ {
            x: 0.,
            y: 0.,
            z: -5.,
        },
        r_sq: 1.,
        mat: Material {
            color: Rgb([255, 0, 0]),
            diffuse: 1.,
            ambient: 1.,
            reflect: 1.,
        },
    }));

    let img = camera.render(&Scene {
        bg: Rgb([0, 50, 50]),
        actors: actors,
    });
    img.save("rust.png")?;

    Ok(())
}
