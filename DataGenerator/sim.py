import numpy as np
import skimage.graph
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt, gaussian_filter


class Costmap:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H - 1, W - 1]
        self.goal = [10, 10]

    @staticmethod
    def cost_from_mask_gaussian(own_mask, sigma=6.0, other_mask=None):
        if not np.any(own_mask):
            cost = np.full(own_mask.shape, -1.0, dtype=np.float32)
            return cost
        d = distance_transform_edt(~own_mask).astype(np.float32)
        c01 = np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        cost = (2.0 * c01 - 1.0).astype(np.float32)
        return cost

    def calculateCostmaps(self, occupancy_map, sigma=6.0):
        mcps = {}
        combined_mask = np.any(
            np.stack(list(occupancy_map.values())) > 0, axis=0
        )
        for key in occupancy_map:
            own_mask = occupancy_map[key] > 0
            other_mask = combined_mask & (~own_mask)
            cost = Costmap.cost_from_mask_gaussian(
                own_mask=own_mask, sigma=sigma, other_mask=other_mask
            )
            mcps[key] = cost
        return mcps

    def calculateCost(self, obstacles_by_class: dict, traj_sigma=1.5, class_offsets=None):
        """class_offsets: optional {class_name: pixels}. The offset inflates ONLY the
        cost mask for that class (-> wider berth in the planned trajectory_map target),
        while the feature maps (binary/radii/sin/cos) keep the true obstacle radius.
        class_offsets=None reproduces the original behavior byte-for-byte."""
        self.obstacles_by_class = obstacles_by_class

        rows, cols = np.ogrid[: self.H, : self.W]

        occupancy_map = {}
        binary_occupancy_map = {}
        radii_maps = {}
        sin_angle_maps = {}
        cos_angle_maps = {}

        for key, obstacle in obstacles_by_class.items():
            occupancy_map[key] = np.zeros((self.H, self.W))
            binary_occupancy_map[key] = np.zeros((self.H, self.W))
            radii_maps[key] = np.zeros((self.H, self.W), dtype=np.float32)
            sin_angle_maps[key] = np.zeros((self.H, self.W), dtype=np.float32)
            cos_angle_maps[key] = np.zeros((self.H, self.W), dtype=np.float32)

            offset = 0.0 if not class_offsets else float(class_offsets.get(key, 0.0))
            for item in obstacle:
                r, c = item["pos"]
                radius = item["rad"]
                angle = item.get("angle", 0.0)
                dist_sqrt = (rows - r) ** 2 + (cols - c) ** 2
                mask = dist_sqrt <= radius ** 2                       # true obstacle -> feature maps
                eff_radius = max(0.0, radius + offset)                # clamp so a big - offset can't grow it
                cost_mask = dist_sqrt <= eff_radius ** 2              # inflated -> cost/path target
                occupancy_map[key][cost_mask] = 1
                binary_occupancy_map[key][r, c] = 1
                radii_maps[key][mask] = radius
                sin_angle_maps[key][mask] = np.sin(angle)
                cos_angle_maps[key][mask] = np.cos(angle)

        final_mcps = self.calculateCostmaps(occupancy_map)

        # Fuse per-class costmaps and shift so all values are positive
        full_cm = np.sum(list(final_mcps.values()), axis=0)
        full_cm_shifted = full_cm - full_cm.min() + 1e-3

        # Goal attraction: pull path toward goal
        goal_mask = np.zeros((self.H, self.W), dtype=np.float32)
        goal_mask[self.goal[0], self.goal[1]] = 1.0
        goal_dist = distance_transform_edt(1.0 - goal_mask).astype(np.float32)
        goal_dist /= goal_dist.max()
        full_cm_shifted += 0.5 * goal_dist

        # Compute optimal trajectory
        path, cost = route_through_array(full_cm_shifted, self.robot, self.goal)

        # Convert path to H×W trajectory heatmap
        trajectory_map = np.zeros((self.H, self.W), dtype=np.float32)
        for r, c in path:
            trajectory_map[r, c] = 1.0

        # Soften with Gaussian blur for smoother diffusion target
        if traj_sigma > 0:
            trajectory_map = gaussian_filter(trajectory_map, sigma=traj_sigma)
            tmax = trajectory_map.max()
            if tmax > 0:
                trajectory_map /= tmax

        return (
            trajectory_map,
            binary_occupancy_map,
            radii_maps,
            sin_angle_maps,
            cos_angle_maps,
        )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = Costmap(H=64, W=64)
    cm.goal = [10, 10]
    data = {
        "chair": [
            {"pos": (20, 30), "rad": 3, "angle": 0.5},
            {"pos": (50, 7), "rad": 2, "angle": 1.2},
        ],
        "table": [
            {"pos": (35, 40), "rad": 3, "angle": 2.0},
            {"pos": (15, 50), "rad": 2, "angle": 0.0},
        ],
    }

    traj, bom, rm, sin_m, cos_m = cm.calculateCost(data)

    # Build combined occupancy for overlay
    occ = np.zeros((cm.H, cm.W))
    for key in bom:
        rows, cols = np.ogrid[: cm.H, : cm.W]
        for item in data[key]:
            r, c = item["pos"]
            radius = item["rad"]
            mask = (rows - r) ** 2 + (cols - c) ** 2 <= radius ** 2
            occ[mask] = 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Trajectory heatmap
    im0 = axes[0].imshow(traj, cmap="hot", origin="upper")
    axes[0].set_title("Trajectory Heatmap (target)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # 2) Trajectory + obstacles + goal/robot overlay
    axes[1].imshow(traj, cmap="hot", origin="upper", alpha=0.6)
    axes[1].imshow(
        np.ma.masked_where(occ == 0, occ),
        cmap="Blues",
        origin="upper",
        alpha=0.7,
    )
    axes[1].plot(cm.goal[1], cm.goal[0], "g*", markersize=14, label="Goal")
    axes[1].plot(cm.robot[1], cm.robot[0], "cs", markersize=10, label="Robot")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Trajectory + Obstacles")

    # 3) Binary path (thresholded) over occupancy
    binary_path = (traj > 0.1).astype(np.float32)
    axes[2].imshow(occ, cmap="Greys", origin="upper", alpha=0.3)
    axes[2].imshow(
        np.ma.masked_where(binary_path == 0, binary_path),
        cmap="Greens",
        origin="upper",
        alpha=0.9,
    )
    axes[2].plot(cm.goal[1], cm.goal[0], "g*", markersize=14, label="Goal")
    axes[2].plot(cm.robot[1], cm.robot[0], "cs", markersize=10, label="Robot")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_title("Thresholded Path + Obstacles")

    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    plt.tight_layout()
    plt.savefig("trajectory_viz.png", dpi=150)
    print("Saved trajectory_viz.png")
