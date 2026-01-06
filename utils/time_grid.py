"""
Utility functions for time grid construction
Used for Asian option averaging schedules
"""

def generate_observation_indices(step: int, total_steps: int = 252):
    """
    Generate observation indices for Asian option averaging.

    Parameters
    ----------
    step : int
        Observation interval (e.g. every 15 days)
    total_steps : int
        Total time steps (default: 252)

    Returns
    -------
    list[int]
        Indices used for arithmetic averaging
    """
    return list(range(step, total_steps, step))
