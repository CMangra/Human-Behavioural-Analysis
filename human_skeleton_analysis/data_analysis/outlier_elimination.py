import numpy as np

MAX_LEG_EXTENSION_DEG = 60.0

MAX_MEAN_SYMMETRY_VARIANCE = 10.0


def evaluate_biomechanical_plausibility(theta_l_deg_list, theta_r_deg_list, symmetry_variance_list):
    # Constraint A: Does any leg swing past 60 degrees?
    max_extension = max(np.max(theta_l_deg_list), np.max(theta_r_deg_list))
    passes_constraint_a = bool(max_extension <= MAX_LEG_EXTENSION_DEG)

    # Constraint B: Is the overall sequence balanced?
    mean_variance = np.mean(symmetry_variance_list)
    passes_constraint_b = bool(abs(mean_variance) <= MAX_MEAN_SYMMETRY_VARIANCE)

    return {
        "is_valid": passes_constraint_a and passes_constraint_b,
        "passes_constraint_a": passes_constraint_a,
        "passes_constraint_b": passes_constraint_b,
        "max_extension_observed_deg": float(max_extension),
        "mean_symmetry_variance_deg": float(mean_variance)
    }