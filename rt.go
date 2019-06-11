package main

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

const MAX_BOUNCES = 3
const EPSILON = 1e-8

func SolveQuadratic(a, b, c float64) (float64, float64) {
	var q, x0, x1 float64
	discr := b*b - 4*a*c
	if discr < 0 {
		return math.NaN(), math.NaN()
	} else if discr == 0 {
		x0 = -0.5 * b / a
		x1 = x0
	} else {
		if b > 0 {
			q = -0.5 * (b + math.Sqrt(discr))
		} else {
			q = -0.5 * (b - math.Sqrt(discr))
		}
		x0 = q / a
		x1 = c / q
	}
	if x0 < x1 {
		return x0, x1
	} else {
		return x1, x0
	}
}

func ClampByte(v float64) uint8 {
	if v >= 255. {
		return 225
	}
	if v < 0 {
		return 0
	}
	return uint8(v)
}

func ScaleColor(c color.NRGBA, scalar float64) color.NRGBA {
	return color.NRGBA{
		R: ClampByte(float64(c.R) * scalar),
		G: ClampByte(float64(c.G) * scalar),
		B: ClampByte(float64(c.B) * scalar),
		A: ClampByte(float64(c.A) * scalar)}
}

func SumVec3(vecs ...Vec3) Vec3 {
	total := Vec3{}
	for _, c := range vecs {
		total = total.Plus(c)
	}
	return total
}

func Refract(i Vec3, n Vec3, index float64) (Vec3, bool) {
	cosi := math.Max(-1., math.Min(1., i.Dot(n))) // clamp due to precision loss
	etai := 1.
	etat := index

	if cosi < 0 {
		cosi = -cosi
	} else {
		etai, etat = etat, etai
		n = n.Scale(-1)
	}

	eta := etai / etat
	k := 1. - eta*eta*(1.-cosi*cosi)
	if k < 0 {
		return Vec3{}, false
	} else {
		return i.Scale(eta).Plus(n.Scale(eta*cosi - math.Sqrt(k))), true
	}
}

type Shape interface {
	Refract(ray Ray, p Vec3) (Ray, bool)
	SurfaceNormal(point Vec3) Vec3
	Intersect(ray Ray) float64
	Material() Material
}

type Vec3 struct {
	X float64
	Y float64
	Z float64
}

func (vec Vec3) Len() float64 {
	return math.Sqrt(vec.X*vec.X + vec.Y*vec.Y + vec.Z*vec.Z)
}

func (vec Vec3) Normalize() Vec3 {
	l := vec.Len()
	return Vec3{vec.X / l, vec.Y / l, vec.Z / l}
}

func (lhs Vec3) Plus(rhs Vec3) Vec3 {
	return Vec3{lhs.X + rhs.X, lhs.Y + rhs.Y, lhs.Z + rhs.Z}
}

func (lhs Vec3) Minus(rhs Vec3) Vec3 {
	return Vec3{lhs.X - rhs.X, lhs.Y - rhs.Y, lhs.Z - rhs.Z}
}

func (lhs Vec3) Mul(rhs Vec3) Vec3 {
	return Vec3{lhs.X * rhs.X, lhs.Y * rhs.Y, lhs.Z * rhs.Z}
}

func (vec Vec3) Scale(scale float64) Vec3 {
	return Vec3{vec.X * scale, vec.Y * scale, vec.Z * scale}
}

func (lhs Vec3) Dot(rhs Vec3) float64 {
	return lhs.X*rhs.X + lhs.Y*rhs.Y + lhs.Z*rhs.Z
}

type Ray struct {
	Orig Vec3
	Dir  Vec3
}

type LightDir struct {
	Dir       Vec3
	Len       float64
	Intensity float64
}

type PointLight struct {
	// TODO color
	Pos       Vec3
	Intensity float64
}

func (sphere Sphere) Refract(ray Ray, p Vec3) (Ray, bool) {
	n := sphere.SurfaceNormal(p)
	refractDir, hasRefraction := Refract(ray.Dir, n, sphere.Material().RefractIndex)
	if !hasRefraction {
		return Ray{}, false
	}
	return Ray{p.Plus(refractDir.Scale(EPSILON)), refractDir}, true
}

func (plane Plane) Refract(ray Ray, p Vec3) (Ray, bool) {
	return Ray{}, false
}

func (light *PointLight) GetDirs(point Vec3) []LightDir {
	v := light.Pos.Minus(point)
	l := v.Len()
	i := math.Min(light.Intensity/(l*l*math.Pi*4), 1.)
	return []LightDir{{v, l, i}}
}

type Material struct {
	Color        Vec3
	Ambient      float64
	Diffuse      float64
	Reflect      float64
	Refract      float64
	RefractIndex float64
}

type Sphere struct {
	Center   Vec3
	RSquared float64
	Mat      Material
}

func (sphere Sphere) Intersect(ray Ray) float64 {
	L := ray.Orig.Minus(sphere.Center)
	a := ray.Dir.Dot(ray.Dir)
	b := 2. * ray.Dir.Dot(L)
	c := L.Dot(L) - sphere.RSquared
	t0, t1 := SolveQuadratic(a, b, c)

	if math.IsNaN(t0) {
		return math.NaN()
	}

	if math.Abs(t0) < EPSILON {
		return t1
	}

	if t0 < 0 {
		t0 = t1
		if t0 < 0 {
			return math.NaN()
		}
	}
	return t0
}

func (sphere Sphere) SurfaceNormal(point Vec3) Vec3 {
	v := point.Minus(sphere.Center)
	return v.Normalize()
}

func (sphere Sphere) Material() Material {
	return sphere.Mat
}

type Plane struct {
	Point  Vec3
	Normal Vec3
	Mat    Material
}

func (plane Plane) Intersect(ray Ray) float64 {
	num := plane.Point.Minus(ray.Orig).Dot(plane.Normal)
	denom := ray.Dir.Dot(plane.Normal)
	if math.Abs(denom) < EPSILON {
		return math.NaN()
	}
	t := num / denom
	if t < 0 {
		return math.NaN()
	}
	return t
}

func (plane Plane) Material() Material {
	return plane.Mat
}

func (plane Plane) SurfaceNormal(Vec3) Vec3 {
	return plane.Normal
}

type Camera struct {
	HRes    int
	VRes    int
	FOV     float64
	ar      float64
	tanterm float64
}

func NewCamera(hres, vres int, fov float64) Camera {
	ar := float64(hres) / float64(vres)
	tanterm := math.Tan(fov / 2. * math.Pi / 180.)
	return Camera{hres, vres, fov, ar, tanterm}
}

func (camera Camera) Cast(i, j int) Ray {
	px := (2.*((float64(i)+0.5)/float64(camera.HRes)) - 1.) * camera.tanterm * camera.ar
	py := (1. - 2.*((float64(j)+0.5)/float64(camera.VRes))) * camera.tanterm
	return Ray{Vec3{}, Vec3{px, py, -1.}.Normalize()}
}

type Scene struct {
	Camera          Camera
	Actors          []Shape
	Lights          []PointLight // TODO a Light interface
	BackgroundColor Vec3
}

func (scene Scene) color(ray Ray, ttl uint8) Vec3 {
	idx := -1
	minT := float64(math.Inf(1))

	for i, actor := range scene.Actors {
		t := actor.Intersect(ray)
		if math.IsNaN(t) {
			continue
		}
		if t < minT {
			minT = t
			idx = i
		}
	}

	if idx == -1 {
		return scene.BackgroundColor
	} else {
		actor := scene.Actors[idx]
		p := ray.Orig.Plus(ray.Dir.Scale(minT))
		return SumVec3(
			actor.Material().Color.Scale(actor.Material().Ambient),
			scene.Diffuse(actor, p),
			scene.Reflect(actor, ray, p, ttl),
			scene.Refract(actor, ray, p, ttl))
	}
}

func (scene Scene) Refract(actor Shape, ray Ray, p Vec3, ttl uint8) Vec3 {
	if ttl == 0 || actor.Material().Refract <= 0 {
		return Vec3{}
	}
	refractRay, hasRefraction := actor.Refract(ray, p)
	if !hasRefraction {
		return Vec3{}
	}
	return scene.color(refractRay, ttl-1).Scale(actor.Material().Refract)
}

func (scene Scene) Reflect(actor Shape, ray Ray, p Vec3, ttl uint8) Vec3 {
	if ttl == 0 || actor.Material().Reflect <= 0 {
		return Vec3{}
	}
	n := actor.SurfaceNormal(p)
	reflectDir := ray.Dir.Minus(n.Scale(2 * ray.Dir.Dot(n))).Normalize()
	reflectRay := Ray{p.Plus(reflectDir.Scale(EPSILON)), reflectDir}
	return scene.color(reflectRay, ttl-1).Scale(actor.Material().Reflect)
}

func (scene Scene) Diffuse(actor Shape, p Vec3) Vec3 {
	if actor.Material().Diffuse <= 0 {
		return Vec3{}
	}

	for _, l := range scene.Lights {
		// TODO blend multiple lights instead of assuming just 1
		dirs := l.GetDirs(p)
		total := Vec3{}
		for _, ld := range dirs {
			ray := Ray{p, ld.Dir.Normalize()}
			lightOrShadow := scene.LightOrShadow(ray, actor, p, ld.Len)
			total = total.Plus(lightOrShadow.Scale(ld.Intensity))
		}
		return total
	}
	return Vec3{}
}

func (scene Scene) LightOrShadow(ray Ray, obj Shape, p Vec3, lightT float64) Vec3 {
	for _, actor := range scene.Actors {
		t := actor.Intersect(ray)
		if !math.IsNaN(t) && t > EPSILON && t < lightT {
			return Vec3{}
		}
	}
	return obj.Material().Color.Scale(obj.SurfaceNormal(p).Dot(ray.Dir)).Scale(obj.Material().Diffuse)
}

type ScanLine struct {
	I      int
	Pixels []color.NRGBA
}

func (scene Scene) Render() image.Image {
	ch := make(chan ScanLine, 20)
	for i := 0; i < scene.Camera.HRes; i++ {
		go RenderScanLine(ch, i, scene)
	}

	img := image.NewRGBA(image.Rect(0, 0, scene.Camera.HRes, scene.Camera.VRes))
	for i := 0; i < scene.Camera.HRes; i++ {
		line := <-ch
		for y, px := range line.Pixels {
			img.Set(line.I, y, px)
		}
	}
	return img
}

func RenderScanLine(ch chan ScanLine, i int, scene Scene) {
	line := make([]color.NRGBA, scene.Camera.VRes)
	for j := 0; j < scene.Camera.VRes; j++ {
		colorVec := scene.color(scene.Camera.Cast(i, j), MAX_BOUNCES)
		line[j] = color.NRGBA{
			ClampByte(colorVec.X),
			ClampByte(colorVec.Y),
			ClampByte(colorVec.Z),
			255}
	}
	ch <- ScanLine{i, line}
}

func main() {
	s1Mat := Material{Vec3{255, 50, 50}, 0.3, 0.3, 0.3, 0, 0}
	s2Mat := Material{Vec3{30, 250, 120}, 0.2, 0.5, 0.3, 0, 0}
	s3Mat := Material{Vec3{20, 20, 255}, 0.1, 0.2, 0.7, 0, 0}
	s4Mat := Material{Vec3{255, 255, 255}, 0, .1, .25, .9, 1.18}
	floorMat := Material{Vec3{150, 150, 240}, 0.3, 0.4, 0.3, 0, 0}

	s1 := Sphere{Vec3{0., 0., -6}, 1.4, s1Mat}
	s2 := Sphere{Vec3{-4, 0.4, -8}, 2, s2Mat}
	s3 := Sphere{Vec3{4, 2, -10}, 2.5, s3Mat}
	s4 := Sphere{Vec3{-.2, -1, -4}, 0.8, s4Mat}
	floor := Plane{Vec3{0, -2, 0}, Vec3{0, 1, 0}, floorMat}
	light := PointLight{Vec3{0, 0, -2.5}, 700.}

	camera := NewCamera(1024, 768, 65.)
	scene := Scene{camera, []Shape{floor, s1, s2, s3, s4}, []PointLight{light}, Vec3{30, 30, 30}}
	writer, err := os.Create("out.png")
	if err == nil {
		png.Encode(writer, scene.Render())
	} else {
		os.Stderr.Write([]byte("Could not open out.png, aborting\n"))
	}
}
