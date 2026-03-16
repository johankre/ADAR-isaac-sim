import carb
import numpy as np 

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

class Adar:
    def __init__(self, origin=(0.0, 0.0, 1.0), max_range=5.0, num_points=5000):
        self.origin = carb.Float3(*map(float, origin))
        self.max_range = max_range
        self.num_points = num_points
        self.ortho_tol = 0.02 # cosine of angle threshold for orthogonality (eg. 0.02 ~ 88.85 degrees)

        self._stage = omni.usd.get_context().get_stage()
        self._sensor_path = "/World/AdarSensor"
        self._sensor_sphere(radius=0.1)

        # filterd hits that represent the ADAR point cloud
        self.points = []

        self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._scene_query = get_physx_scene_query_interface()
        self._dirs = self._generate_directions()

    def _sensor_sphere(self, radius=0.1):
        """
        Creates a visual sphere at the sensor origin.
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

        return np.stack((x, y, z), axis=-1)
    
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

        return abs(dot) > (1 - threshold)

    def _scan(self):
        hits = []
        hits_normals = []
        for dir in self._dirs:
            hit = self._scene_query.raycast_closest(self.origin, dir, self.max_range)

            if not hit['hit']:
                continue


            n = hit["normal"]
            if self._is_orthogonal_to_normal(dir, n, self.ortho_tol):
                p = hit["position"]
                hits.append((float(p[0]), float(p[1]), float(p[2])))
                hits_normals.append((float(n[0]), float(n[1]), float(n[2])))
        
        return hits, hits_normals

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

    def build_3x3_probe(self, point_of_interest, probe_spacing: float = 0.01):
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

        probe_rays = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue

                offset = axis1 * i * probe_spacing + axis2 * j * probe_spacing
                probe_origin = origin + offset
                probe_point = poi + offset
                probe_rays.append((probe_origin, probe_point))

        return probe_rays
        
    def check_hit(self, hit):
        if hit['hit']:
            return True
        return False

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
    
    def evaluate_surface_curvature(self, grad_f, hess_f, center_point):
        """
        Note: there are many ways to try to estimate the curvature of an implicit surface (need input)
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
        pass
        

    def _draw_probe_points(self, probe_points):
        points = [carb.Float3(x, y, z) for (x, y, z) in probe_points]
        colors = [carb.ColorRgba(0.0, 1.0, 0.0, 1.0) for _ in probe_points] # Green color
        sizes  = [5.0 for _ in probe_points]

        self._debug_draw.draw_points(points, colors, sizes)

    def _draw_points(self, points):
        if len(points) == 0:
            return
        
        # draw all hits as red points
        pts = [carb.Float3(x, y, z) for (x, y, z) in points]
        colors = [carb.ColorRgba(1.0, 0.0, 0.0, 1.0) for _ in points]
        sizes  = [10.0 for _ in points]

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

    hits, hit_normals = adar._scan()
    for (hit, hit_normal) in zip(hits, hit_normals):
        probe_rays = adar.build_3x3_probe(hit, 0.03)
        probe_hits = []

        for probe_origin, probe_target in probe_rays:
            dir = probe_target - probe_origin
            probe_hit = adar._scene_query.raycast_closest(probe_origin, dir, adar.max_range)
            if not adar.check_hit(probe_hit):
                break

            probe_hits.append(probe_hit)

        else:
            probe_points = [hit["position"] for hit in probe_hits]
            probe_normals = [hit["normal"] for hit in probe_hits]
            adar._draw_probe_points(probe_points)
            
            # include the center point and normal of interest as well
            probe_points.append(hit)
            probe_normals.append(hit_normal)


            f, grad_f, hess_f = adar.surface_interpolation(probe_points, probe_normals)
            print(f"Hessian at hit {hit}: {hess_f(hit[0], hit[1], hit[2])}")

            points.append(hit)

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
