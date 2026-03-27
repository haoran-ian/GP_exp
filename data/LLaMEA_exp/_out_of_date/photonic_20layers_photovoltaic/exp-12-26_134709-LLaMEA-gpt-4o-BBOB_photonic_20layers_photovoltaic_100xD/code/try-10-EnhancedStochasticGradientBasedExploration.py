import numpy as np

class EnhancedStochasticGradientBasedExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def estimate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            if self.evaluations >= self.budget:
                break
            x_step = np.copy(x)
            x_step[i] += epsilon
            grad[i] = (func(x_step) - func(x)) / epsilon
            self.evaluations += 1
        return grad

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
        best_x = x
        best_value = func(x)
        self.evaluations += 1

        step_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
        perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])
        
        momentum = np.zeros(self.dim)
        beta = 0.9  # Momentum coefficient

        while self.evaluations < self.budget:
            grad = self.estimate_gradient(func, x)
            if self.evaluations >= self.budget:
                break

            # Update momentum
            momentum = beta * momentum + (1 - beta) * grad
            # Introduce stochastic perturbation with adaptive adjustment
            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
            x_new = x - step_size * momentum + perturbation
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

            value = func(x_new)
            self.evaluations += 1

            if value < best_value:
                best_value = value
                best_x = x_new
                step_size *= 1.1  # Slightly increase step size if improving
                perturbation_strength *= 0.9  # Reduce perturbation as solution improves
            else:
                step_size *= 0.7  # Reduce step size if not improving
                perturbation_strength *= 1.2  # Increase perturbation to escape local optima

            x = x_new

        return best_x