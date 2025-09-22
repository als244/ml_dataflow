
def select_bins(N: int, select_frac: float) -> list[int]:
    """
    Selects a fraction of N bins, maximizing the spacing between them.
    This ensures the first and last possible bins are candidates for selection.

    Args:
        N: The total number of bins available (e.g., 100).
        select_frac: The fraction of bins to select (a float between 0.0 and 1.0).

    Returns:
        A list of integer indices for the selected bins.
    
    Raises:
        ValueError: If N is not a positive integer or if select_frac is not in [0, 1].
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= select_frac <= 1.0):
        raise ValueError("select_frac must be between 0.0 and 1.0.")

    # 1. Determine the total number of bins to select.
    num_to_select = int(N * select_frac)

    selected_indices = set()

    if num_to_select == 0:
        return selected_indices
    if num_to_select == 1:
        selected_indices.add(0)
        return selected_indices

    # 2. To maximize spacing, we treat the bins as points from 0 to N-1.
    #    The ideal step size will spread num_to_select points across this range,
    #    ensuring the first (0) and last (N-1) bins are included.
    #    This is analogous to numpy.linspace(0, N-1, num_to_select).
    ideal_step = float(N - 1) / (num_to_select - 1)

    # 3. Iterate and select bins based on the ideal step.
    #    We use round() to pick the nearest integer bin, which distributes
    #    the selections more symmetrically than floor().
    current_position = 0.0
    for _ in range(num_to_select):
        index = int(round(current_position))
        selected_indices.add(index)
        current_position += ideal_step
        
    return selected_indices



def select_periodic_bins(N: int, select_frac: float) -> list[int]:
    """
    Selects a fraction of N bins in a nearly periodic manner.

    Args:
        N: The total number of bins available (e.g., 100).
        select_frac: The fraction of bins to select (a float between 0.0 and 1.0).

    Returns:
        A list of integer indices for the selected bins.
    
    Raises:
        ValueError: If N is not a positive integer or if select_frac is not in [0, 1].
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0.0 <= select_frac <= 1.0):
        raise ValueError("select_frac must be between 0.0 and 1.0.")

    # 1. Determine the total number of bins to select.
    num_to_select = int(N * select_frac)

    selected_indices = set()

    if num_to_select == 0:
        return selected_indices

    # 2. Calculate the ideal (often fractional) step size between selections
    #    to distribute them as evenly as possible across N bins.
    #    This is the core of the periodic behavior.
    ideal_step = float(N) / num_to_select

    # 3. Iterate and select bins based on the ideal step.
    #    We use a floating-point accumulator for the position and round down
    #    to get the actual bin index.
    current_position = 0.0
    for _ in range(num_to_select):
        index = math.floor(current_position)
        
        # Ensure the index stays within bounds [0, N-1]
        # This can happen due to floating point inaccuracies on the last element.
        if index < N:
            selected_indices.add(index)
        else:
            selected_indices.add(N - 1)
            
        current_position += ideal_step

    return selected_indices