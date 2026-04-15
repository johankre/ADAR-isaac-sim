from abc import ABC, abstractmethod
from dataclasses import dataclass

from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager, IsaacEvents
from isaacsim.util.debug_draw import _debug_draw

import omni
import omni.timeline

from pxr import Gf, Sdf, UsdPhysics

import carb
import numpy as np 
from sklearn.cluster import DBSCAN


import omni
from omni.physx import get_physx_scene_query_interface

from pxr import Gf, Sdf, UsdGeom

@dataclass
class Point:
    position: np.ndarray
    normal: float

class AdarStepStrategy(ABC):
    @abstractmethod
    def execute(self, adar):
        """
        Executes a a given process for point cloud generation.
        """
        raise NotImplementedError("AdarStepStrategy is an abstract base class and cannot be instantiated directly.")
    
class IntensityStepStrategy(AdarStepStrategy):
    def execute(self, adar):
        def reflection_intensity(points, wave_length):
            k = 2 * np.pi / wave_length
            p = np.asarray(points, dtype=np.float32)

            o = np.asarray(adar.origin, dtype=np.float32)
            s = np.sqrt(np.sum(np.power(p - o, 2), axis=-1))
            F = np.sum(np.exp(1j * k * 2 * s))
            intensity = abs(F)**2 / s.size ** 2

            return intensity

        adar._debug_draw.clear_points()

        points = []

        hits, hit_normals = adar._scan()
        for (hit, hit_normal) in zip(hits, hit_normals):
            if not adar.floor_filter(hit, floor_height=0.01):
                continue

            probe_rays = adar.build_3x3_probe(hit)
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
                probe_points.append(hit)
                probe_normals.append(hit_normal)

                intensity = reflection_intensity(probe_points, adar.wave_length)
                points.append((hit, intensity))


        adar._draw_points(points)

class ClusteringStepStrategy(AdarStepStrategy):
    def execute(self, adar):

        def theta(position: np.ndarray, normal: np.ndarray):
            """
            Computes the angle between the ray direction and the surface normal at the hit point.
            """
            ray_dir = position - np.asarray(adar.origin, dtype=np.float32)
            ray_dir = ray_dir/ np.linalg.norm(ray_dir)

            normal = np.asarray(normal, dtype=np.float32)
            normal = normal / np.linalg.norm(normal)

            dot = np.clip(np.dot(ray_dir, normal), -1.0, 1.0)
            return np.arccos(np.abs(dot))
        
        def reflection_direction(ray_dir: np.ndarray, normal: np.ndarray) -> np.ndarray:
            """
            Computes the reflection direction based on the ray direction and surface normal.

            Uses the specular reflection formula:
                d_out = d_in - 2 * dot(d_in, n) * n
            """
            ray_norm = np.linalg.norm(ray_dir)
            normal_norm = np.linalg.norm(normal)

            if ray_norm < 1e-6:
                print(f"DEBUG: ray_dir has zero norm: {ray_norm}, ray_dir={ray_dir}")
                
            if normal_norm < 1e-6:
                print(f"DEBUG: normal has zero norm: {normal_norm}, normal={normal}")

            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            normal = normal / np.linalg.norm(normal)

            dir_out =  ray_dir - 2 * np.dot(ray_dir, normal) * normal
            return dir_out / np.linalg.norm(dir_out)
        
        def reflected_hit_distance_sensor_plane(point_origin: np.ndarray, point_hit: np.ndarray, hit_normal: np.ndarray) -> float:
            """
            Computes the distance from the sensor origin to where the reflected ray intersects the sensor plane.
            
            Given a hit point and its surface normal, computes the reflection direction and finds where
            the reflected ray intersects the sensor plane (defined by origin and adar.direction normal).
            Returns the distance from the origin to this intersection point on the plane.
            """
            sensor_origin = np.asarray(adar.origin, dtype=np.float32)
            plane_normal = adar.direction
            
            # Normalize vectors
            ray_dir = point_hit - point_origin
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            hit_normal = hit_normal / np.linalg.norm(hit_normal)
            
            reflect_dir = reflection_direction(ray_dir, hit_normal)
            
            # Find intersection of reflected ray with sensor plane
            # Ray: R(t) = position + t * reflect_dir
            # Plane: (R - origin) · plane_normal = 0
            # Solving: (position - origin + t * reflect_dir) · plane_normal = 0
            denom = np.dot(reflect_dir, plane_normal)
            
            # If reflected ray is parallel to plane, return large distance
            if abs(denom) < 1e-6:
                return float('inf')
            
            t = np.dot(sensor_origin - point_hit, plane_normal) / denom
            
            # Only consider intersections in front of the hit point (t > 0)
            if t < 0:
                return float('inf')
            
            # Compute intersection point
            intersection = point_hit + t * reflect_dir
            
            # Distance from origin to intersection point on the plane
            distance = np.linalg.norm(intersection - sensor_origin)
            return float(distance)
        
        def score_function(dist: float, sigma=0.5):
            """
            Computes a score based on the distance of the reflected hit, where closer hits get higher scores.
            """
            return np.exp(- (dist ** 2) / (sigma ** 2))
        
        def cluster_hits(hits: np.ndarray, eps=0.5, min_samples=5):
            """
            Clusters the hit points using DBSCAN and returns the cluster labels.
            """
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(hits)
            return clustering.labels_
        
        def is_valid_refelction(reflected_dir: np.ndarray) -> bool:
            """
            Checks if the reflected ray direction is valid (i.e. it points back towards the sensor plane).
            """
            plane_normal = adar.direction
            return np.dot(reflected_dir, plane_normal) > 0
        
        def sensor_plane_hit_point(reflected_dir: np.ndarray, hit_position: np.ndarray) -> np.ndarray:
            """
            Computes where the outgoing reflected ray would intersect the sensor plane.

            The sensor plane passes through `origin` and is oriented perpendicular to the
            vector from origin to the first surface hit — i.e. its normal is that incoming
            ray direction.  We use the standard ray-plane intersection:

                t = dot(origin - hit_position, plane_normal) / dot(reflected_dir, plane_normal)
                intersection = hit_position + t * reflected_dir
            """
            plane_normal = adar.direction

            denom = np.dot(reflected_dir, plane_normal)

            t = np.dot(origin - hit_position, plane_normal) / denom
            return hit_position + t * reflected_dir

        adar._debug_draw.clear_points()
        adar._debug_draw.clear_lines()

        hits, hit_normals = adar._scan_all()
        origin = np.asarray(adar.origin, dtype=np.float32)

        hits = np.asarray(hits, dtype=np.float32)
        hit_normals = np.asarray(hit_normals, dtype=np.float32)
        hit_normals = hit_normals / np.linalg.norm(hit_normals)

        point_items: list[Point] = []

        # For debug visualization of reflection paths
        #   origin → first hit          : yellow
        #   hit → next bounce           : cyan
        #   last bounce → sensor plane  : magenta
        lines_origin_to_first = ([], [])
        lines_bounces         = ([], [])
        lines_to_plane        = ([], [])
        
        for position, hit_normal in zip(hits, hit_normals):
            ray_origin = origin

            ray_dir = position - ray_origin
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            if np.dot(ray_dir, hit_normal) <= 0:
                hit_normal = -hit_normal

            score = score_function(reflected_hit_distance_sensor_plane(ray_origin, position, hit_normal))
            if score > 0.1:
                point_items.append(Point(position=position, normal=score))
                continue
            
            distance_traveled = np.linalg.norm(position - ray_origin)
            current_hit = position
            current_normal = hit_normal

            # Accumulate path segments for this ray; only commit them once we
            # know at least one bounce along the chain scored > 0.1.
            first_hit_of_ray = position          # the very first surface hit from origin
            pending_bounces = ([], [])    # bounce segments accumulated so far
            
            for i in range(3):
                if distance_traveled > adar.max_range * 2:
                    break
                
                prev_ray_dir = ray_dir
                ray_dir = reflection_direction(prev_ray_dir, current_normal)

                # Offset the ray origin slightly away from the surface to avoid self-intersection
                ray_bias = 1e-2
                biased_ray_origin = current_hit + ray_dir * ray_bias

                refelction_hit = adar._scene_query.raycast_closest(biased_ray_origin, ray_dir, (adar.max_range * 2 - distance_traveled) / 2)

                if not adar.check_hit(refelction_hit):
                    break

                hit_position = np.asarray(refelction_hit["position"], dtype=np.float32)
                hit_normal = np.asarray(refelction_hit["normal"], dtype=np.float32)
                
                hit_normal = hit_normal / np.linalg.norm(hit_normal)
                if np.dot(ray_dir, hit_normal) <= 0:
                    hit_normal = -hit_normal

                ray_origin = current_hit
                current_hit = hit_position
                current_normal = hit_normal

                score = score_function(reflected_hit_distance_sensor_plane(ray_origin, current_hit, current_normal))
                distance_traveled += np.linalg.norm(hit_position - ray_origin)
                
                pending_bounces[0].append(tuple(ray_origin))
                pending_bounces[1].append(tuple(hit_position))

                if score > 0.1 and is_valid_refelction(ray_dir):
                    last_ray_dir = ray_dir

                    total_distances_traveled = distance_traveled + np.linalg.norm(hit_position - origin)
                    psuedo_direction = (hit_position - origin) / np.linalg.norm(hit_position - origin)
                    estimated_postion = origin + psuedo_direction * (total_distances_traveled / 2)
                    point_items.append(Point(position=estimated_postion, normal=score))
                
                    # Segment 1: sensor origin → first surface hit (yellow)
                    lines_origin_to_first[0].append(tuple(origin))
                    lines_origin_to_first[1].append(tuple(first_hit_of_ray))

                    # Segment 2: all bounce-to-bounce legs (cyan)
                    lines_bounces[0].extend(pending_bounces[0])
                    lines_bounces[1].extend(pending_bounces[1])

                    # Segment 3: last bounce → where the outgoing ray hits the sensor plane (magenta)
                    # current_hit is now the position of the last bounce
                    last_ray_dir = reflection_direction(current_hit - ray_origin, current_normal)
                    plane_pt = sensor_plane_hit_point(last_ray_dir, current_hit)
                    lines_to_plane[0].append(tuple(current_hit))
                    lines_to_plane[1].append(tuple(plane_pt))
                    break


        # Draw all three path segments in distinct colours
        if lines_origin_to_first[0]:
            adar._draw_lines(*lines_origin_to_first, carb.ColorRgba(1.0, 1.0, 0.0, 1.0), size=1.5)  # yellow
        if lines_bounces[0]:
            adar._draw_lines(*lines_bounces,         carb.ColorRgba(0.0, 1.0, 1.0, 1.0), size=1.5)  # cyan
        if lines_to_plane[0]:
            adar._draw_lines(*lines_to_plane,        carb.ColorRgba(1.0, 0.0, 1.0, 1.0), size=1.5)  # magenta



        if len(point_items) == 0:
            return

        positions = np.asarray([p.position for p in point_items], dtype=np.float32)
        scores = np.asarray([p.normal for p in point_items], dtype=np.float32)
        labels = cluster_hits(positions, eps=0.5, min_samples=2)

        unique_labels = np.unique(labels)
        points = []
        for cluster_id in unique_labels:
            # ignore noise points labeled as -1 by DBSCAN
            if cluster_id == -1:
                continue

            cluster_points = positions[labels == cluster_id]
            cluster_weights = scores[labels == cluster_id]
            weighted_pos = np.average(cluster_points, axis=0, weights=cluster_weights)
            points.append((weighted_pos, float(np.mean(cluster_weights)) * 2.0))

        adar._draw_points(points)

class CurvatureStepStrategy(AdarStepStrategy):
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
    def execute(self, adar):
                #_, grad_f, hess_f = adar.surface_interpolation(probe_points, probe_normals)
                #K, H, k_1, k_2 = adar.evaluate_surface_curvature(grad_f, hess_f, hit)
        raise NotImplementedError("CurvatureStepStrategy is not implemented yet.")

class AdarRayDirectionStrategy():
    @abstractmethod
    def generate_directions(self, adar):
        """
        Generates ray directions for the ADAR scan pattern.
        """
        raise NotImplementedError("AdarRayDirectionStrategy is an abstract base class and cannot be instantiated directly.")
    
class UniformSphereStrategy(AdarRayDirectionStrategy):
    def generate_directions(self, adar):
        i = np.arange(adar.num_points)
        phi = (1 + np.sqrt(5)) / 2

        z = 1 - 2 * (i + 0.5) / adar.num_points
        lat = np.arcsin(z)
        lon = (2 * np.pi * i / (phi**2)) % (2 * np.pi)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        dirs = np.stack((x, y, z), axis=-1)

        mask = (np.dot(dirs, adar.direction) > 0)
        filtered = dirs[mask]

        return filtered

class PlaneScanStrategy(AdarRayDirectionStrategy):
    def generate_directions(self, adar):
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d = adar.direction / np.linalg.norm(adar.direction)

        # If the scan direction is parallel to the local up vector, choose a different reference vector
        if np.allclose(d, up) or np.allclose(d, -up):
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Compute orthogonal basis vectors for the scan plane
        right = np.cross(d, up)
        right /= np.linalg.norm(right)
        local_up = np.cross(right, d)

        theta = np.linspace(-np.pi / 2, np.pi / 2, adar.num_points)
        x = np.cos(theta)
        y = np.sin(theta)
        dirs = np.outer(x, d) + np.outer(y, right)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        return dirs



_OPPOSING_PAIRS = [
    (1, 6), # top <-> bottom
    (3, 4), # left <-> right
    (0, 7), # top-left <-> bottom-right
    (2, 5), # top-right <-> bottom-left
]

STEP_STRATEGIES = {
    "intensity": IntensityStepStrategy(),
    "clustering": ClusteringStepStrategy(),
    "curvature": CurvatureStepStrategy(),
}

SCAN_STRATEGIES = {
    "uniform_sphere": UniformSphereStrategy(),
    "plane_scan": PlaneScanStrategy(),
}

class Adar:
    def __init__(self, origin=(0.0, 0.0, 2.0), max_range=5.0, num_points=10000, wave_length=0.002, step_strategy="intensity", scan_strategy="uniform_sphere"):
        self.origin = carb.Float3(*map(float, origin))
        self.direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # ADAR faces along the positive X-axis in its local frame
        self.max_range = max_range
        self.num_points = num_points
        self.wave_length = wave_length
        self.scan_strategy = SCAN_STRATEGIES.get(scan_strategy, SCAN_STRATEGIES["uniform_sphere"])
        self.ortho_tol = 0.1
        self.step_strategy = STEP_STRATEGIES.get(step_strategy, STEP_STRATEGIES["intensity"])

        self._stage = omni.usd.get_context().get_stage()
        self._sensor_path = "/World/AdarSensor"
        self._camera_path = self._sensor_path + "/Camera"

        self._sensor_sphere(radius=0.1)
        self._create_camera()

        # filterd hits that represent the ADAR point cloud
        self.points = []

        self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._scene_query = get_physx_scene_query_interface()
        self._dirs = self.scan_strategy.generate_directions(self)

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
        cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -45.0))
        
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

    def _scan_all(self):
        hits = []
        normals = []
        for dir in self._dirs:
            hit = self._scene_query.raycast_closest(self.origin, dir, self.max_range)
            if hit['hit']:
                p = hit["position"]
                n = hit["normal"]
                hits.append((float(p[0]), float(p[1]), float(p[2])))
                normals.append((float(n[0]), float(n[1]), float(n[2])))

        return hits, normals

    def _scan(self):
        hits = []
        hits_normals = []
        for dir in self._dirs:
            hit = self._scene_query.raycast_closest(self.origin, dir, self.max_range)

            if not hit['hit']:
                continue

            n = hit["normal"]
            if self._is_parallel_to_normal(dir, n, self.ortho_tol):
                p = hit["position"]
                hits.append((float(p[0]), float(p[1]), float(p[2])))
                hits_normals.append((float(n[0]), float(n[1]), float(n[2])))
        
        return hits, hits_normals

    def _is_clean_edge(self, point_of_interest, probe_points, probe_normals, threshold=0.1):
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

    def build_3x3_probe(self, point_of_interest):
        """
        Builds a 3x3 grid of probe points around the given point of interest.
        The offset plane is perpendicular to the ray direction at the point of interest, so all probe rays are parallel 
        to the point of intrest ray.
        """
        origin = np.asarray(self.origin, dtype=np.float32)
        poi = np.asarray(point_of_interest, dtype=np.float32)

        ray_dir = poi - origin
        ray_len = np.linalg.norm(ray_dir)
        ray_dir /= ray_len

        axis1, axis2 = self._axes_from_direction(ray_dir)

        n = 3
        probe_rays = []
        probe_spacing = self.wave_length / 2
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
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
    
    def floor_filter(self, point, floor_height=0.01):
        """
        Returns True if the point is above the floor height.
        """
        return point[2] > floor_height

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
    
    def set_step_strategy(self, strategy):
        self.step_strategy = strategy
    
    def set_scan_strategy(self, strategy):
        self.scan_strategy = strategy
        self._dirs = self.scan_strategy.generate_directions(self)

    def run_step(self):
        self.step_strategy.execute(self)

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
        sizes  = [intensity * 10 for _, intensity in points]

        points.extend(pts)
        colors.extend(colors)
        sizes.extend(sizes)   

        self._debug_draw.draw_points(pts, colors, sizes)
    
    def _draw_lines(self, starts, ends, colors: carb.ColorRgba, size=2.0):
        start_points = [carb.Float3(x, y, z) for (x, y, z) in starts]
        end_points = [carb.Float3(x, y, z) for (x, y, z) in ends]
        color_list = [colors for _ in starts]
        size_list = [size for _ in starts]

        self._debug_draw.draw_lines(start_points, end_points, color_list, size_list)


PHYSICS_SCENE_PATH = "/World/PhysicsScene"

def create_phsics_scene(stage):
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid(): 
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path(PHYSICS_SCENE_PATH))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)

def build_test_scene(stage):
    # Ground plane
    ground_path = "/World/Ground"
    ground_prim = stage.GetPrimAtPath(ground_path)
    if not ground_prim.IsValid():
        ground_xform = UsdGeom.Xform.Define(stage, Sdf.Path(ground_path))
        ground_cube = UsdGeom.Cube.Define(stage, ground_xform.GetPath().AppendChild("Cube"))
        ground_cube.CreateSizeAttr().Set(1.0)
        ground_cube.AddScaleOp().Set(Gf.Vec3d(20.0, 20.0, 0.01))
        # Ensure the ground can be hit by raycasts
        UsdPhysics.CollisionAPI.Apply(ground_cube.GetPrim())

    # Cylinder
    cylinder_path = "/World/Cylinder"
    cylinder_prim = stage.GetPrimAtPath(cylinder_path)
    if not cylinder_prim.IsValid():
        cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(cylinder_path))
        cylinder.CreateRadiusAttr().Set(0.2)
        cylinder.CreateHeightAttr().Set(4.0)
        # Place cylinder on the ground (centered in Z)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 0.5))
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())


def create_cylinder_mover(speed=0.5, amplitude=3.0, height=0.5):
    """
    Creates a cylinder movement function with its own internal state.
    
    Args:
        speed: Radians per second
        amplitude: Movement amplitude in meters
        height: Height of cylinder center
        
    Returns:
        A move_cylinder function that only depends on stage and dt
    """
    state = {
        "time": 0.0,
        "speed": speed,
        "amplitude": amplitude,
        "height": height,
    }
    
    def move_cylinder(stage, dt):
        """
        Moves the cylinder along a quarter-circle arc in the X-Y plane.
        
        Args:
            stage: The USD stage
            dt: Delta time
        """
        state["time"] += dt

        angle = (np.pi / 2.0) * (0.5 * (np.sin(state["speed"] * state["time"]) + 1.0))
        x = state["amplitude"] * np.cos(angle)
        y = state["amplitude"] * np.sin(angle)

        cyl_prim = stage.GetPrimAtPath("/World/Cylinder")
        if cyl_prim.IsValid():
            xform = UsdGeom.Xformable(cyl_prim)
            ops = xform.GetOrderedXformOps()
            if ops:
                ops[0].Set(Gf.Vec3d(x, y, state["height"]))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(x, y, state["height"]))
    
    return move_cylinder


def register_step_callback(adar: Adar, stage):
    """
    Registers a callback that moves the scene and runs ADAR after each physics step.
    
    Args:
        adar: The Adar instance
        stage: The USD stage
        
    Returns:
        The callback ID
    """
    # Create the cylinder mover with its own internal state
    #move_cylinder = create_cylinder_mover(speed=0.5, amplitude=3.0, height=0.5)

    def update(dt: float):
        #move_cylinder(stage, dt)
        adar.run_step()

    step_cb_id = SimulationManager.register_callback(
        update,
        IsaacEvents.POST_PHYSICS_STEP,   # or PRE_PHYSICS_STEP
    )
    return step_cb_id
    
def log_to_file(hits, hit_normals, filename="adar_hits.csv"):
    with open(filename, "w") as f:
        f.write("x,y,z,nx,ny,nz\n")
        for hit_pos, normal in zip(hits, hit_normals):
            f.write(f"{hit_pos[0]},{hit_pos[1]},{hit_pos[2]},{normal[0]},{normal[1]},{normal[2]}\n")
    print(f"Logged {len(hits)} hits to {filename}")


def main():
    print("main started - initializing ADAR sensor...")
    stage = omni.usd.get_context().get_stage()

    create_phsics_scene(stage)
    #build_test_scene(stage)

    world = World(physics_prim_path=PHYSICS_SCENE_PATH)
    world.initialize_physics()
    world.reset()

    adar = Adar()
    adar.set_step_strategy(STEP_STRATEGIES["clustering"])
    adar.set_scan_strategy(SCAN_STRATEGIES["uniform_sphere"])
    register_step_callback(adar, stage)

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
    
    print("Adar initialized and running.")

main()