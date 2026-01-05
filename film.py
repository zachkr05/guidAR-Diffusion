# film.py - GPR-FiLM version

import numpy as np
from typing import List, Tuple, Dict
from scipy.interpolate import interp1d
from scipy import ndimage as ndi


class TrajectoryFeatureExtractor:
    """Extract features from trajectory pair for GPR input."""
    
    def __init__(self, n_points: int = 50, grid_size: int = 64):
        self.n_points = n_points
        self.grid_size = grid_size
    
    def extract(self, original_traj: List[Tuple[int, int]], user_traj: List[Tuple[int, int]]) -> np.ndarray:
        orig = self._resample(original_traj)
        user = self._resample(user_traj)
        
        displacement = user - orig
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        
        mean_displacement = np.mean(displacement_magnitude)
        max_displacement = np.max(displacement_magnitude)
        std_displacement = np.std(displacement_magnitude)
        
        max_idx = np.argmax(displacement_magnitude)
        max_position = max_idx / self.n_points
        max_location = orig[max_idx] / self.grid_size
        
        mean_direction = np.mean(displacement, axis=0)
        mean_direction_norm = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)
        
        orig_length = self._path_length(orig)
        user_length = self._path_length(user)
        length_ratio = user_length / (orig_length + 1e-8)
        
        area_between = np.sum(displacement_magnitude) / self.n_points
        
        features = np.array([
            mean_displacement / self.grid_size,
            max_displacement / self.grid_size,
            std_displacement / self.grid_size,
            max_position,
            max_location[0],
            max_location[1],
            mean_direction_norm[0],
            mean_direction_norm[1],
            length_ratio - 1.0,
            area_between / self.grid_size,
        ])
        return features
    
    def _resample(self, traj: List[Tuple[int, int]]) -> np.ndarray:
        traj = np.array(traj, dtype=np.float32)
        if len(traj) < 2:
            return np.tile(traj[0], (self.n_points, 1))
        diffs = np.diff(traj, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dists = np.concatenate([[0], np.cumsum(dists)])
        total = cum_dists[-1]
        if total < 1e-6:
            return np.tile(traj[0], (self.n_points, 1))
        sample_d = np.linspace(0, total, self.n_points)
        interp_r = interp1d(cum_dists, traj[:, 0], kind='linear', fill_value='extrapolate')
        interp_c = interp1d(cum_dists, traj[:, 1], kind='linear', fill_value='extrapolate')
        return np.stack([interp_r(sample_d), interp_c(sample_d)], axis=1)
    
    def _path_length(self, traj: np.ndarray) -> float:
        diffs = np.diff(traj, axis=0)
        return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))


class GPRFiLM:
    """Gaussian Process Regression for FiLM parameters."""
    
    def __init__(self, length_scale: float = 0.5, noise_var: float = 0.01, output_scale: float = 1.0):
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.output_scale = output_scale
        self.X_train = None
        self.y_gamma = None
        self.y_beta = None
        self._alpha_gamma = None
        self._alpha_beta = None
        self._L = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist
        sq_dist = cdist(X1, X2, metric='sqeuclidean')
        return self.output_scale * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
    
    def fit(self, X: np.ndarray, y_gamma: np.ndarray, y_beta: np.ndarray):
        from scipy.linalg import cholesky, solve_triangular
        self.X_train = X.copy()
        self.y_gamma = y_gamma.copy()
        self.y_beta = y_beta.copy()
        
        K = self._rbf_kernel(X, X)
        K += self.noise_var * np.eye(len(X))
        
        try:
            self._L = cholesky(K, lower=True)
        except:
            K += 1e-6 * np.eye(len(X))
            self._L = cholesky(K, lower=True)
        
        self._alpha_gamma = solve_triangular(self._L.T, solve_triangular(self._L, y_gamma, lower=True))
        self._alpha_beta = solve_triangular(self._L.T, solve_triangular(self._L, y_beta, lower=True))
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from scipy.linalg import solve_triangular
        if self.X_train is None:
            M = X.shape[0]
            return np.zeros(M), np.ones(M) * self.output_scale, np.zeros(M), np.ones(M) * self.output_scale
        
        K_star = self._rbf_kernel(X, self.X_train)
        gamma_mean = K_star @ self._alpha_gamma
        beta_mean = K_star @ self._alpha_beta
        
        v = solve_triangular(self._L, K_star.T, lower=True)
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-8)
        
        return gamma_mean, np.sqrt(var), beta_mean, np.sqrt(var)
    
    def add_observation(self, x: np.ndarray, gamma: float, beta: float):
        x = x.reshape(1, -1)
        if self.X_train is None:
            self.fit(x, np.array([gamma]), np.array([beta]))
        else:
            X_new = np.vstack([self.X_train, x])
            y_gamma_new = np.append(self.y_gamma, gamma)
            y_beta_new = np.append(self.y_beta, beta)
            self.fit(X_new, y_gamma_new, y_beta_new)


class GPRFiLMManager:
    """Manages GPR-FiLM models for all classes."""
    
    def __init__(self, num_classes: int = 4, grid_size: int = 64):
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.feature_extractor = TrajectoryFeatureExtractor(n_points=50, grid_size=grid_size)
        self.gpr_models: Dict[int, GPRFiLM] = {c: GPRFiLM() for c in range(num_classes)}
        self.observations: Dict[int, List[Dict]] = {c: [] for c in range(num_classes)}
    
    def add_trajectory_edit(self, original_traj, user_traj, affected_class, target_gamma, target_beta):
        features = self.feature_extractor.extract(original_traj, user_traj)
        self.observations[affected_class].append({
            'features': features, 'gamma': target_gamma, 'beta': target_beta
        })
        self.gpr_models[affected_class].add_observation(features, target_gamma, target_beta)
    
    def get_all_predictions(self) -> Dict[int, Dict[str, float]]:
        predictions = {}
        for class_id in range(self.num_classes):
            if len(self.observations[class_id]) > 0:
                last_obs = self.observations[class_id][-1]
                features = last_obs['features'].reshape(1, -1)
                gamma_mean, gamma_std, beta_mean, beta_std = self.gpr_models[class_id].predict(features)
                predictions[class_id] = {
                    'gamma': gamma_mean[0], 'gamma_std': gamma_std[0],
                    'beta': beta_mean[0], 'beta_std': beta_std[0],
                    'n_observations': len(self.observations[class_id])
                }
            else:
                predictions[class_id] = {
                    'gamma': 0.0, 'gamma_std': 1.0, 'beta': 0.0, 'beta_std': 1.0, 'n_observations': 0
                }
        return predictions
    
    def compute_target_params_from_irl(self, current_costmap, target_costmap, class_activation):
        mask = ndi.maximum_filter(class_activation, size=12)
        mask = ndi.gaussian_filter(mask.astype(np.float32), sigma=3)
        region = mask > 0.1
        
        if not region.any():
            return 0.0, 0.0
        
        c = current_costmap[region]
        t = target_costmap[region]
        #m = mask[region]
        delta = t - c
        


        mean_delta = np.mean(delta)
        max_delta = np.max(np.abs(delta))


        beta = float(np.clip(mean_delta * 3.0, -1.0, 1.0))  # Direct cost addition
        gamma = float(np.clip(max_delta * 2.0, -2.0, 2.0))  # Multiplicative boost
        
        return gamma, beta
        #A = np.column_stack([c * m, m])
        #try:
        #    params, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
        #    gamma, beta = params
        #except:
        #    gamma = np.mean(delta / (c + 1e-8))
        #    beta = np.mean(delta)
        
        #return float(np.clip(gamma, -2.0, 5.0)), float(np.clip(beta, -1.0, 2.0))


def apply_gpr_film_modulation(
    base_costmap: np.ndarray,
    class_activations: np.ndarray,
    gpr_film_manager: GPRFiLMManager
) -> np.ndarray:
    """
    Apply GPR-FiLM modulation to a base costmap.
    
    This is the key function - it applies learned (gamma, beta) as post-processing.
    """
    modified = base_costmap.copy()
    
    predictions = gpr_film_manager.get_all_predictions()
    
    # Find which class dominates each pixel
    class_dominance = np.argmax(class_activations, axis=0)
    any_class_present = np.max(class_activations, axis=0) > 0.1
    
    for class_id, params in predictions.items():
        if params['n_observations'] == 0:
            continue
        
        gamma = params['gamma']
        beta = params['beta']
        
        if abs(gamma) < 0.01 and abs(beta) < 0.01:
            continue
        
        # Only pixels where THIS class dominates
        this_class_dominates = (class_dominance == class_id) & any_class_present
        class_act = class_activations[class_id]
        class_region = np.where(this_class_dominates, class_act, 0.0)
        class_region = ndi.gaussian_filter(class_region.astype(np.float32), sigma=1.0)
        
        # FiLM: modified = base * (1 + gamma * region) + beta * region
        modified = modified * (1 + gamma * class_region) + beta * class_region
    
    return np.clip(modified, 0, 1)
