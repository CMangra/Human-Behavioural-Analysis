import numpy as np

# Constraint A: Maximum Anatomical Extension
MAX_LEG_EXTENSION_DEG = 60.0

# Constraint B: Sequence Balance Limit (Mean Symmetry Variance)
MAX_MEAN_SYMMETRY_VARIANCE = 10.0

# Constraint C: LiDAR Foot-Ground Consistency
# Tolerance for how far the stance foot can float/sink (e.g. 20 cm)
MAX_FOOT_GROUND_ERROR_M = 0.20


def evaluate_biomechanical_plausibility(theta_l_deg_list, theta_r_deg_list, symmetry_variance_list,
                                        lowest_foot_dist_list):
    # Constraint A
    max_extension = max(np.max(theta_l_deg_list), np.max(theta_r_deg_list))
    passes_constraint_a = bool(max_extension <= MAX_LEG_EXTENSION_DEG)

    # Constraint B
    mean_variance = np.mean(symmetry_variance_list)
    passes_constraint_b = bool(abs(mean_variance) <= MAX_MEAN_SYMMETRY_VARIANCE)

    # Constraint C
    mean_foot_error = np.mean(np.abs(lowest_foot_dist_list))
    passes_constraint_c = bool(mean_foot_error <= MAX_FOOT_GROUND_ERROR_M)

    is_valid = passes_constraint_a and passes_constraint_b and passes_constraint_c

    return {
        "is_valid": is_valid,
        "passes_constraint_a": passes_constraint_a,
        "passes_constraint_b": passes_constraint_b,
        "passes_constraint_c": passes_constraint_c,
        "max_extension_observed_deg": float(max_extension),
        "mean_symmetry_variance_deg": float(mean_variance),
        "mean_foot_ground_error_m": float(mean_foot_error)
    }