import numpy as np
from scipy.optimize import minimize

# Gradient Descent Optimizer
def gradient_descent_optimizer(initial_params, compute_expectation, backend, lr=0.1, max_iter=100, epsilon=1e-6):
    """
    Simple Gradient Descent from scratch using finite differences.
    Arguments:
        initial_params: Initial parameter vector.
        compute_expectation: Cost function to minimize.
        backend: Qiskit backend.
        lr: Learning rate.
        max_iter: Number of iterations.
        epsilon: Small shift for gradient approximation.
    Returns:
        Optimized parameters.
    """
    params = np.copy(initial_params)
    n = len(params)

    target_cost = -4.0
    patience = 3
    hit_target_count = 0

    for i in range(max_iter):
        grad = np.zeros(n)

        for j in range(n):
            shift = np.zeros(n)
            shift[j] = epsilon
            plus = compute_expectation(params + shift, backend)
            minus = compute_expectation(params - shift, backend)
            grad[j] = (plus - minus) / (2 * epsilon)

        params -= lr * grad

        if i % 10 == 0 or i == max_iter - 1:
            cost = compute_expectation(params, backend)
            print(f"[GD Iter {i+1}] Cost: {cost:.6f}")

        cost = compute_expectation(params, backend)
        if cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {cost}")
    return params



# COBYLA Optimizer
def cobyla_optimizer(initial_params, compute_expectation, backend, step_size=0.1, max_iter=100):
    """
    Simple COBYLA-like optimizer using coordinate-wise exploratory steps.
    Arguments:
        initial_params: Starting parameters.
        compute_expectation: Cost function to minimize.
        backend: Qiskit backend.
        step_size: Initial step size.
        max_iter: Iterations.
    Returns:
        Optimized parameters.
    """
    params = np.copy(initial_params)
    n = len(params)

    target_cost = -4.0
    patience = 3
    hit_target_count = 0

    for i in range(max_iter):
        improved = False
        current_cost = compute_expectation(params, backend)

        for j in range(n):
            for direction in [+1, -1]:
                candidate = np.copy(params)
                candidate[j] += direction * step_size
                new_cost = compute_expectation(candidate, backend)

                if new_cost < current_cost:
                    params = candidate
                    current_cost = new_cost
                    improved = True
                    break  # accept the step and re-evaluate

        if not improved:
            step_size *= 0.5  # reduce step size if no improvement

        if i % 10 == 0 or i == max_iter - 1:
            print(f"[COBYLA Iter {i+1}] Cost: {current_cost:.6f}")

        if current_cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {current_cost}")
    return params



# SPSA Optimizer

def spsa_optimizer(initial_params, compute_expectation, backend, max_iter=100, a=0.2, c=0.1, alpha=0.602, gamma=0.101):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) from scratch.
    Arguments:
        initial_params: numpy array of initial parameters.
        compute_expectation: callable function to evaluate cost.
        backend: Qiskit backend (used inside compute_expectation).
        max_iter: number of iterations.
        a, c: gain sequences for learning rate and perturbation.
        alpha, gamma: exponents controlling decay of a_k and c_k.
    Returns:
        Final optimized parameters.
    """
    params = np.copy(initial_params)
    n = len(initial_params)

    target_cost = -4.0
    patience = 3
    hit_target_count = 0

    for k in range(max_iter):
        ak = a / (k + 1)**alpha
        ck = c / (k + 1)**gamma
        delta = 2 * np.random.randint(0, 2, size=n) - 1  # random ±1 vector

        params_plus = params + ck * delta
        params_minus = params - ck * delta

        loss_plus = compute_expectation(params_plus, backend)
        loss_minus = compute_expectation(params_minus, backend)

        # Gradient estimate
        gk = (loss_plus - loss_minus) / (2.0 * ck * delta)

        # Parameter update
        params = params - ak * gk

        current_cost = compute_expectation(params, backend)
        if k % 10 == 0 or k == max_iter - 1:
            print(f"[SPSA Iter {k+1}] Cost: {current_cost:.6f}")

        if current_cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {current_cost}")
    return params


#Adam Optimizer
def adam_optimizer(initial_params, compute_expectation, backend, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=100):
    """
    Adam Optimizer from scratch using finite differences for gradient estimation.
    Arguments:
        initial_params: Initial parameter vector.
        compute_expectation: Cost function to minimize.
        backend: Qiskit backend.
        learning_rate: Step size for parameter updates.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        epsilon: Small constant for numerical stability.
        max_iter: Maximum number of optimization iterations.
    Returns:
        Optimized parameter vector.
    """
    params = np.array(initial_params)
    m = np.zeros_like(params)  # First moment
    v = np.zeros_like(params)  # Second moment
    t = 0

    target_cost = -4.0
    patience = 3
    hit_target_count = 0

    for k in range(max_iter):
        t += 1
        # Compute gradients (approximating it using finite differences)
        grad = np.zeros_like(params)
        epsilon_fd = 1e-5  # Small perturbation to compute the gradient
        for i in range(len(params)):
            params_eps = np.copy(params)
            params_eps[i] += epsilon_fd
            grad[i] = (compute_expectation(params_eps, backend) - compute_expectation(params, backend)) / epsilon_fd

        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Parameter update
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        current_cost = compute_expectation(params, backend)
        if k % 10 == 0 or k == max_iter - 1:
            print(f"[Adam Iter {k+1}] Cost: {current_cost:.6f}")

        if current_cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {current_cost}")
    return params


#Simulated Annealing
def simulated_annealing_optimizer(initial_params, compute_expectation, backend, T_start=1.0, T_min=1e-5, alpha=0.9, max_iter=100):
    """
    Simulated Annealing optimizer from scratch for parameter optimization.
    Arguments:
        initial_params: Initial parameter vector.
        compute_expectation: Cost function to minimize.
        backend: Qiskit backend.
        T_start: Starting temperature for annealing.
        T_min: Minimum temperature to stop the process.
        alpha: Cooling rate (0 < alpha < 1).
        max_iter: Maximum number of iterations.
    Returns:
        Optimized parameter vector.
    """
    params = np.array(initial_params)
    T = T_start
    current_cost = compute_expectation(params, backend)

    target_cost = -4.0
    patience = 3
    hit_target_count = 0

    for k in range(max_iter):
        # Generate a new candidate by random perturbation
        new_params = params + np.random.uniform(-0.1, 0.1, size=params.shape)
        new_cost = compute_expectation(new_params, backend)
        
        # Calculate the acceptance probability
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            params = new_params
            current_cost = new_cost
        
        # Cool down the temperature
        T *= alpha

        if current_cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

        if k % 10 == 0 or k == max_iter - 1:
            print(f"[Simulated Annealing Iter {k+1}] Cost: {current_cost:.6f}")
        
        if T < T_min:
            break

    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {current_cost}")
    return params


#Momentum based Gradient Descent
def momentum_gradient_descent_optimizer(initial_params, compute_expectation, backend, learning_rate=0.01, beta=0.9, max_iter=100):
    """
    Gradient Descent with Momentum for parameter optimization.
    Arguments:
        initial_params: Initial parameter vector.
        compute_expectation: Cost function to minimize.
        backend: Qiskit backend.
        learning_rate: Step size for updating parameters.
        beta: Momentum term (controls the contribution of previous gradients).
        max_iter: Maximum number of iterations.
    Returns:
        Optimized parameter vector.
    """
    params = np.array(initial_params)
    velocity = np.zeros_like(params)

    target_cost = -4.0
    patience = 3
    hit_target_count = 0
    
    for k in range(max_iter):
        # Compute gradients (approximating using finite differences)
        grad = np.zeros_like(params)
        epsilon = 1e-5  # Small perturbation to compute the gradient
        for i in range(len(params)):
            params_eps = np.copy(params)
            params_eps[i] += epsilon
            grad[i] = (compute_expectation(params_eps, backend) - compute_expectation(params, backend)) / epsilon
        
        # Update velocity and parameters
        velocity = beta * velocity + (1 - beta) * grad
        params -= learning_rate * velocity

        current_cost = compute_expectation(params, backend)

        if current_cost <= target_cost:
            hit_target_count += 1
            if hit_target_count >= patience:
                print(f"Early stopping: Cost reached {target_cost} for {patience} consecutive iterations.")
                break
        else:
            hit_target_count = 0

        if k % 10 == 0 or k == max_iter - 1:
            print(f"[Momentum based GD Iter {k+1}] Cost: {current_cost:.6f}")
    
    print(f"Final optimized parameters: {params}")
    print(f"Final Optimized Cost: {current_cost}")
    return params