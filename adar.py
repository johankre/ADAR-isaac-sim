import carb
import numpy as np 
from shapely import Point

from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager, IsaacEvents
from isaacsim.util.debug_draw import _debug_draw

import omni
import omni.timeline
from omni.physx import get_physx_scene_query_interface

from pxr import Gf, Sdf, UsdPhysics, UsdGeom


_OPPOSING_PAIRS = [
    (1, 6), # top <-> bottom
    (3, 4), # left <-> right
    (0, 7), # top-left <-> bottom-right
    (2, 5), # top-right <-> bottom-left

]

class Hit:
    def __init__(self, point: Point, normal: Point) -> None:
        self.point = point
        self.normal = normal

class Hits:
    def __init__(self) -> None:
        self.hits: list[Hit] = []
    
    def append(self, hit: Hit) -> None:
        self.hits.append(hit)

    def points(self) -> list[Point]:
        return [hit.point for hit in self.hits]

    def normals(self) -> list[Point]:
        return [hit.normal for hit in self.hits]

    def __iter__(self):
        return iter(self.hits)

class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.line = (start, end)

    def start(self) -> Point:
        return self.line[0]
    
    def end(self) -> Point:
        return self.line[1]

    def direction(self) -> np.ndarray:
        d = np.array([
            self.end().x - self.start().x,
            self.end().y - self.start().y,
            self.end().z - self.start().z,], dtype=np.float32
        )

        return d / np.linalg.norm(d)

class Lines:
    def __init__(self, lines: list[Line]) -> None:
        self.lines: list[Line] = lines

    @classmethod
    def from_origin_to_endpoints(cls, origin: Point, endpoints: list[Point]): 
        """
        Build Lines from a shared origin and a list of endpoints.
        """
        return cls([Line(origin, ep) for ep in endpoints])

    def __iter__(self):
        return iter(self.lines)

class Adar:
    def __init__(self, origin=Point(0.0, 0.0, 1.0), max_range=5.0, num_points=5000, wave_length=0.1):
        self.origin: Point = origin
        self.max_range = max_range
        self.num_points = num_points
        self.wave_length = wave_length
        self.ortho_tol = 0.02 # cosine of angle threshold for orthogonality (eg. 0.02 ~ 88.85 degrees)

        self._stage = omni.usd.get_context().get_stage()
        self._sensor_path = "/World/AdarSensor"
        self._camera_path = self._sensor_path + "/Camera"

        self._sensor_sphere(radius=0.1)
        self._create_camera()

        # filterd hits that represent the ADAR point cloud
        self.points = []

        self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._scene_query = get_physx_scene_query_interface()
        self._dirs = self._generate_directions()

    def _sensor_sphere(self, radius=0.1):
        """
        Creates a visual sphere at the sensor origin representing the sensor.
        """

        prim = self._stage.GetPrimAtPath(self._sensor_path)
        if not prim.IsValid():
            xform = UsdGeom.Xform.Define(self._stage, Sdf.Path(self._sensor_path))
            sphere = UsdGeom.Sphere.Define(self._stage, xform.GetPath().AppendChild("Sphere"))
            sphere.CreateRadiusAttr().Set(radius)

            sphere.CreateDisplayColorAttr().Set([(0.0, 0.4, 1.0)])  # Blue color

            # Always keep it located at the current origin
            xform_prim = self._stage.GetPrimAtPath(self._sensor_path)
            xformable = UsdGeom.Xformable(xform_prim)

            # Clear ops to avoid stacking transforms if called again
            xformable.ClearXformOpOrder()
            t = xformable.AddTranslateOp()
            t.Set(Gf.Vec3d(float(self.origin.x), float(self.origin.y), float(self.origin.z)))

    def _generate_directions(self):
        i = np.arange(self.num_points)
        phi = (1 + np.sqrt(5)) / 2

        z = 1 - 2 * (i + 0.5) / self.num_points
        lat = np.arcsin(z)
        lon = (2 * np.pi * i / (phi**2)) % (2 * np.pi)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        endpoints = [Point(float(xi), float(yi), float(zi)) for (xi, yi, zi) in zip(x, y, z)]
        return Lines.from_origin_to_endpoints(self.origin, endpoints)
    
    def _create_camera(self):
        """
        Creates a camera at the sensor origin for visualization purposes.
        """
        camera_prim = self._stage.GetPrimAtPath(self._camera_path)

        if camera_prim.IsValid():
            return
        
        camera_prim = UsdGeom.Camera.Define(self._stage, Sdf.Path(self._camera_path))

        # Place the camera relative to the sensor frame
        cam_xform = UsdGeom.Xformable(camera_prim.GetPrim())
        cam_xform.ClearXformOpOrder()

        cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 90.0))
        
    def _is_orthogonal_to_normal(self, ray_dir, normal, threshold=0.1):
        d = np.asarray(ray_dir, dtype=np.float32)
        n = np.asarray(normal, dtype=np.float32)

        d_norm = np.linalg.norm(d)
        n_norm = np.linalg.norm(n)

        if d_norm == 0 or n_norm == 0:
            return False
        
        d /= d_norm
        n /= n_norm 
        dot = float(np.dot(d, n))

        return abs(dot) < threshold

    def _scan(self):
        hits = Hits()
        for line in self._dirs:
            ray = self._scene_query.raycast_closest(line.start, line.end, self.max_range)

            if not ray['hit']:
                continue

            p = ray["position"]
            n = ray["normal"]

            point = Point(float(p[0]), float(p[1]), float(p[2]))
            normal = Point(float(n[0]), float(n[1]), float(n[2]))

            hit = Hit(point, normal)

            if self._is_parallel_to_normal(line.direction, n, self.ortho_tol):
                hits.append(hit)
        
        return hits

    def is_clean_edge(self, point_of_interest, probe_points, probe_normals, threshold=0.1):
        """
        Note: this check is in practice not needed.

        Returns True if a clean edge is detected at the point of intrest.

        A clean edge is defined as, if for any two oppsoing probe pair:
            i. The vector from POI to each probe point is orthogonal to the probe normal, and
            ii. The probe normals are not parallel to each other
        """
        poi = np.asarray(point_of_interest, dtype=np.float32)
        for idx1, idx2 in _OPPOSING_PAIRS:
            point_1 = probe_points[idx1]
            point_2 = probe_points[idx2]

            pos_1 = np.asarray(point_1, dtype=np.float32)
            pos_2 = np.asarray(point_2, dtype=np.float32)

            # Check if the vector from POI to each probe point is orthogonal to the probe normal
            vec_to_1 = pos_1 - poi
            vec_to_2 = pos_2 - poi

            normal_1 = probe_normals[idx1]
            normal_2 = probe_normals[idx2]

            if not self._is_orthogonal_to_normal(vec_to_1, normal_1, threshold):
                continue
            if not self._is_orthogonal_to_normal(vec_to_2, normal_2, threshold):
                continue

            # Check if the probe normals are not parallel to each other
            if self._is_parallel_to_normal(normal_1, normal_2, threshold):
                continue

            return True

        return False

    def _is_parallel_to_normal(self, n1, n2, threshold=0.1):
        n1 = np.asarray(n1, dtype=np.float32)
        n2 = np.asarray(n2, dtype=np.float32)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        dot = float(np.dot(n1, n2))
        return abs(dot) > (1 - threshold)

    def build_3x3_probe(self, point_of_interest: Point) -> Lines:
        """
        Builds a 3x3 grid of probe points around the given point of interest.
        The offset plane is perpendicular to the ray direction at the point of interest, so all probe rays are parallel 
        to the point of intrest ray.
        """
        origin = np.asarray(self.origin, dtype=np.float32)
        poi = np.asarray(point_of_interest, dtype=np.float32)

        ray_dir = poi - origin
        ray_dir /= np.linalg.norm(ray_dir)

        axis1, axis2 = self._axes_from_direction(ray_dir)

        probe_rays: list[Line] = []
        probe_spacing = self.wave_length / 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue

                offset = axis1 * i * probe_spacing + axis2 * j * probe_spacing
                probe_origin = origin + offset
                probe_point = poi + offset

                probe_ray = Line(Point(probe_origin), Point(probe_point))
                probe_rays.append(probe_ray)

        return Lines(probe_rays)
        
    def check_hit(self, hit):
        if hit['hit']:
            return True
        return False
    
    def floor_filter(self, point: Point, floor_height=0.01):
        """
        Returns True if the point is above the floor height.
        """
        return point.z > floor_height

    def _axes_from_direction(self, dir):
        """
        Given a direction vector, compute two orthogonal axes that are perpendicular to it.
        """
        dir = np.asarray(dir, dtype=np.float32)
        dir /= np.linalg.norm(dir)

        # Find an arbitrary vector that is not parallel to dir
        if abs(dir[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Compute the first axis as the cross product of dir and the arbitrary vector
        axis1 = np.cross(dir, arbitrary)

        # Compute the second axis as the cross product of dir and axis1
        axis2 = np.cross(dir, axis1)

        return axis1, axis2

    def is_surface_planar(self, point_of_interest, point_of_interest_normal, probe_points, probe_normals, curvature_threshold=0.01):
        """
        Returns True if the surface is estimated to be planar based on the probe points and normals.
        """

        for p, n in zip(probe_points, probe_normals):
            if not self._is_parallel_to_normal(point_of_interest_normal, n, curvature_threshold):
                return False

            # Check if the vector from POI to each probe point is orthogonal to the probe normal (i.e. the probe point lies on the tangent plane defined by the POI normal)
            vec_to_probe = np.asarray(p, dtype=np.float32) - np.asarray(point_of_interest, dtype=np.float32)
            if not self._is_orthogonal_to_normal(vec_to_probe, n, curvature_threshold):
                return False
            
        return True

    def surface_interpolation(self, points, normals):
        """
        Given a set of points and their corresponding normals (Hermite data), perform quadratic surface interpolation to
        estimate the surface in a given region centerd at the point of interest.

        We seek to fit a quadratic surface implicit function of the form:
            f(x) = x^T A x + b^T x + c = 0
        s.t. f(points[i]) = 0 and grad_f(points[i]) = normals[i] for all i, where grad_f is the gradient of f.

        returns f, grad_f, hess_f
        """

        def build_system_quadratic(points, normals):
            P = np.asarray(points, dtype=np.float32)
            G = np.asarray(normals, dtype=np.float32)

            rows = []
            rhs = []
            for (x, y, z), (nx, ny, nz) in zip(P, G):
                # f(x_i) = 0 constraints gives us: 
                # per row: [x_i^2, y_i^2, z_i^2, x_i*y_i, x_i*z_i, y_i*z_i, x_i, y_i, z_i, 1] * [A11, A22, A33, A12, A13, A23, b1, b2, b3, c]^T = 0
                row_f = [x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, 1.0]
                rows.append(row_f)
                rhs.append(0.0)

                # grad_f(x_i) = 2 A x_i + b = n_i constraints gives us:
                # x constraint: [2*x_i, 0, 0, 2*y_i, 2*z_i, 0, 1, 0, 0, 0] * [A11, A22, A33, A12, A13, A23, b1, b2, b3, c]^T = n_i_x
                # y constraint: [0, 2*y_i, 0, 2*x_i, 0, 2*z_i, 0, 1, 0, 0] * [A11, A22, A33, A12, A13, A23, b1, b2, b3, c]^T = n_i_y
                # z constraint: [0, 0, 2*z_i, 0, 2*x_i, 2*y_i, 0, 0, 1, 0] * [A11, A22, A33, A12, A13, A23, b1, b2, b3, c]^T = n_i_z

                row_nx = [2*x, 0.0, 0.0, 2*y, 2*z, 0.0, 1.0, 0.0, 0.0, 0.0]
                rows.append(row_nx)
                rhs.append(nx)

                row_ny = [0.0, 2*y, 0.0, 2*x, 0.0, 2*z, 0.0, 1.0, 0.0, 0.0]
                rows.append(row_ny)
                rhs.append(ny)

                row_nz = [0.0, 0.0, 2*z, 0.0, 2*x, 2*y, 0.0, 0.0, 1.0, 0.0]
                rows.append(row_nz)
                rhs.append(nz)
            
            M = np.vstack(rows)
            y = np.asarray(rhs, dtype=np.float32)

            return M, y
        
        def solve_ls_quadratic(M, y):
            """
            Solves ||M theta - y||_2 for theta
            """
            theta, residuals, rank, singular_values = np.linalg.lstsq(M, y, rcond=None)
            return theta, residuals, rank, singular_values
        
        def quadratic_fn_from_theta(theta):
            """
            Creates a quadratic function from the estimated parameters.
            """
            theta = np.asarray(theta, dtype=np.float32).ravel()
            a11, a22, a33, a12, a13, a23, b1, b2, b3, c = theta
            
            def f(x, y, z):
                return (a11 * x**2 + a22 * y**2 + a33 * z**2 + 
                        a12 * x * y + a13 * x * z + a23 * y * z + 
                        b1 * x + b2 * y + b3 * z + c)

            return f

        def quadratic_gradient_fn_from_theta(theta):
            """
            Creates a gradient function from the estimated parameters.
            """
            theta = np.asarray(theta, dtype=np.float32).ravel()
            a11, a22, a33, a12, a13, a23, b1, b2, b3, _ = theta
            
            def grad_f(x, y, z):
                df_dx = 2 * a11 * x + a12 * y + a13 * z + b1
                df_dy = 2 * a22 * y + a12 * x + a23 * z + b2
                df_dz = 2 * a33 * z + a13 * x + a23 * y + b3
                return np.stack([df_dx, df_dy, df_dz], axis=-1)

            return grad_f
        
        def quadratic_hessian_fn_from_theta(theta):
            """
            Creates a hessian function from the estimated parameters.
            """
            theta = np.asarray(theta, dtype=np.float32).ravel()
            a11, a22, a33, a12, a13, a23, _, _, _, _ = theta
            
            def hess_f(x, y, z):
                d2f_dx2 = 2 * a11
                d2f_dy2 = 2 * a22
                d2f_dz2 = 2 * a33
                d2f_dxdy = a12
                d2f_dxdz = a13
                d2f_dydz = a23

                return np.array([[d2f_dx2, d2f_dxdy, d2f_dxdz],
                                 [d2f_dxdy, d2f_dy2, d2f_dydz],
                                 [d2f_dxdz, d2f_dydz, d2f_dz2]])

            return hess_f

        M, y = build_system_quadratic(points, normals)
        theta, _, _, _ = solve_ls_quadratic(M, y)

        f = quadratic_fn_from_theta(theta)
        grad_f = quadratic_gradient_fn_from_theta(theta)
        hess_f = quadratic_hessian_fn_from_theta(theta)

        return f, grad_f, hess_f

    def reflection_intensity(self, points, wave_length):
        k = 4 * np.pi / wave_length
        F = np.sum(np.exp(1j * k * 2 * (np.asarray(points) - np.asarray(self.origin))))
        intensity = abs(F)**2
        return intensity
    
    def evaluate_surface_curvature(self, grad_f, hess_f, center_point):
        """
        This implementation evaluates just the curvature at the point of interest on the estimated surface.

        We base the curvature estimation on Goldman 2005 "Curvature formulas for implicit curves and surfaces":
            For surfaces there are more than one way of measuring the curvature. 
            
            Based on principal curvatures k_1 and k_2, we can define:
                i. Gaussian curvature K = k_1 * k_2 = (grad_f^T * adj(hess_f) * grad_f) / ||grad_f||^4, where adj(hess_f) is the adjugate of the hessian matrix
                ii. Mean curvature H = (k_1 + k_2) / 2 = (grad_f^T * hess_f * grad_f^T - grad_f^T * grad_f * np.trace(hess_f)) / (2 * |grad_f|^3)
            Note that if either k_1 or k_2 is zero, then K = 0 indicating a parabolic point.
            If both k_1 and k_2 are zero, then K = H = 0 indicating a planar point.

            The principal curvatures k_1 and k_2 can be computed from the mean and Gaussian curvatures by solving the quadratic equation:
                k_1, k_2 = H ± sqrt(H^2 - K)
            
        Acoustic backscattering
        Note: unsure how to best do this, K is not enough. 
        We are unable to diferentiate between a parabolic point (eg. cylinder) and a planar point with just K.
        For H we can have a planar point with H = 0, but also a saddle point with H = 0.
        """

        def gaussian_curvature(grad_f, hess_f, x, y, z):
            g = np.asarray(grad_f(x, y, z), dtype=np.float64)
            H = np.asarray(hess_f(x, y, z), dtype=np.float64)
 
            grad_norm = np.linalg.norm(g)
 
            # Build adjugate (transpose of cofactor matrix) of H
            adj_H = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    minor = np.delete(np.delete(H, i, axis=0), j, axis=1)
                    adj_H[j, i] = ((-1) ** (i + j)) * np.linalg.det(minor)
 
            K = float(g @ adj_H @ g) / (grad_norm ** 4)
            return K

        def mean_curvature(grad_f, hess_f, x, y, z):
            g = np.asarray(grad_f(x, y, z), dtype=np.float64)
            H = np.asarray(hess_f(x, y, z), dtype=np.float64)
 
            grad_norm = np.linalg.norm(g)
 
            numerator = float(g @ H @ g) - (grad_norm ** 2) * np.trace(H)
            mean_H = numerator / (2.0 * grad_norm ** 3)
            return mean_H
 
        x, y, z = center_point[0], center_point[1], center_point[2]
 
        K = gaussian_curvature(grad_f, hess_f, x, y, z)
        H = mean_curvature(grad_f, hess_f, x, y, z)
 
        # Recover principal curvatures from K and H:
        #   k_1, k_2 = H ± sqrt(H^2 - K)
        discriminant = max(H ** 2 - K, 0.0)
        sqrt_disc = np.sqrt(discriminant)
        k1 = H + sqrt_disc
        k2 = H - sqrt_disc
 
        return K, H, k1, k2
        

    def _draw_probe_points(self, probe_points):
        points = [carb.Float3(x, y, z) for (x, y, z) in probe_points]
        colors = [carb.ColorRgba(0.0, 1.0, 0.0, 1.0) for _ in probe_points] # Green color
        sizes  = [5.0 for _ in probe_points]

        self._debug_draw.draw_points(points, colors, sizes)

    def _draw_points(self, points):
        if len(points) == 0:
            return
        
        # draw all hits as red points
        pts = [carb.Float3(x, y, z) for (x, y, z), _ in points]
        colors = [carb.ColorRgba(1.0, 0.0, 0.0, 1.0) for _, _ in points]
        sizes  = [intensity * 0.05 for _, intensity in points]

        points.extend(pts)
        colors.extend(colors)
        sizes.extend(sizes)   

        self._debug_draw.draw_points(pts, colors, sizes)

    def _print_points(self):
        print("Points:")
        for point in self.points:
            print(f"  {point}")

        
# --- World / physics setup (must match the prim path) ---
PHYSICS_SCENE_PATH = "/World/PhysicsScene"

stage = omni.usd.get_context().get_stage()

# Create PhysicsScene at the canonical path
prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
if not prim.IsValid():
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(PHYSICS_SCENE_PATH))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)

# Create World bound to that PhysicsScene
world = World(physics_prim_path=PHYSICS_SCENE_PATH)

world.initialize_physics()
world.reset()

adar = Adar()

def update(dt: float):
    adar._debug_draw.clear_points()

    points = []

    hits = adar._scan()
    for hit in hits:
        if not adar.floor_filter(hit.point, floor_height=0.01):
            continue

        probe_rays = adar.build_3x3_probe(hit.point)
        probe_hits = Hits()

        for ray in probe_rays:
            probe_hit = adar._scene_query.raycast_closest(ray.start, ray.direction, adar.max_range)
            if not adar.check_hit(probe_hit):
                break

            p = Point(probe_hit["position"])
            n = Point(probe_hit["normal"])

            probe_hit = Hit(p, n)
            probe_hits.append(probe_hit)

        else:
            probe_hits.append(hit)

            intensity = adar.reflection_intensity(probe_hits.points(), adar.wave_length)
            points.append((hit, intensity))

            _, grad_f, hess_f = adar.surface_interpolation(probe_hits.points, probe_hits.normals)
            K, H, k_1, k_2 = adar.evaluate_surface_curvature(grad_f, hess_f, hit)

    adar._draw_points(points)

physics_cb_id = None
ready_cb_id = None

timeline = omni.timeline.get_timeline_interface()

_warmup_cb_id = None
_step_cb_id = None

def _register_step_cb():
    global _step_cb_id
    if _step_cb_id is None:
        _step_cb_id = SimulationManager.register_callback(
            update,
            IsaacEvents.POST_PHYSICS_STEP,   # or PRE_PHYSICS_STEP
        )
        print("Physics step callback registered:", _step_cb_id)

def _on_warmup(_dt: float = 0.0):
    global _warmup_cb_id
    # one-shot
    if _warmup_cb_id is not None:
        SimulationManager.deregister_callback(_warmup_cb_id)
        _warmup_cb_id = None
    _register_step_cb()

# If already playing, warmup may already have happened; register immediately.
if timeline.is_playing():
    _register_step_cb()
else:
    _warmup_cb_id = SimulationManager.register_callback(
        _on_warmup,
        IsaacEvents.PHYSICS_WARMUP,
    )
    timeline.play()

print("Adar initialized and running.")
