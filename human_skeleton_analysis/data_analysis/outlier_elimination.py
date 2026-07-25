import numpy as np
import pandas as pd

# Constraint A: Maximum Anatomical Extension
MAX_LEG_EXTENSION_DEG = 60.0

# Constraint B: Sequence Balance Limit (Bio-LSTM Cosine Delta)
# A cosine delta mean near 0 indicates bilateral balance over the sequence.
MAX_MEAN_COSINE_DELTA = 0.15


def evaluate_biomechanical_plausibility(theta_l_deg_list, theta_r_deg_list, bio_lstm_delta_list):
    """
    Evaluates a sequence of 3D poses against established biomechanical literature constraints.
    Returns a dictionary detailing if the sequence passes or fails, and the specific metrics.
    """
    # Constraint A
    max_extension = max(np.max(theta_l_deg_list), np.max(theta_r_deg_list))
    passes_constraint_a = bool(max_extension <= MAX_LEG_EXTENSION_DEG)

    # Constraint B
    mean_sym_delta = np.mean(bio_lstm_delta_list)
    passes_constraint_b = bool(abs(mean_sym_delta) <= MAX_MEAN_COSINE_DELTA)

    is_valid = passes_constraint_a and passes_constraint_b

    return {
        "is_valid": is_valid,
        "passes_constraint_a": passes_constraint_a,
        "passes_constraint_b": passes_constraint_b,
        "max_extension_observed_deg": float(max_extension),
        "mean_bio_lstm_cosine_delta": float(mean_sym_delta)
    }