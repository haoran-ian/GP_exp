import numpy as np

class ImprovedAdaptiveGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def estimate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_step = np.copy(x)
            x_step[i] += epsilon
            grad[i] = (func(x_step) - func(x)) / epsilon
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return grad

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
        best_x = x
        best_value = func(x)
        self.evaluations += 1

        step_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
        perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])
        
        while self.evaluations < self.budget:
            grad = self.estimate_gradient(func, x)
            if self.evaluations >= self.budget:
                break
            
            # Introduce stochastic perturbation to improve exploration
            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
            x_new = x - step_size * grad + perturbation
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

            value = func(x_new)
            self.evaluations += 1

            if value < best_value:
                best_value = value
                best_x = x_new
                step_size *= 1.2 + 0.05 * np.linalg.norm(grad)  # Increase step size dynamically
                perturbation_strength *= 0.8  # Decrease perturbation as solution improves
            else:
                step_size *= 0.5  # Reduce step size if not improving
                perturbation_strength *= 1.1  # Increase perturbation to escape local optima

            x = x_new

        return best_x