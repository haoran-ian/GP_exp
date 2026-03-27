import numpy as np

class MomentumAdaptiveGradientSearch:
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
        best_x = None
        best_value = float('inf')
        
        num_starts = 5  # Number of random starts
        for _ in range(num_starts):
            if self.evaluations >= self.budget:
                break
            
            x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
            current_value = func(x)
            self.evaluations += 1

            step_size = 0.15 * (bounds[:, 1] - bounds[:, 0])
            perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])
            velocity = np.zeros_like(x)
            beta = 0.9  # Momentum factor
            
            while self.evaluations < self.budget:
                grad = self.estimate_gradient(func, x)
                if self.evaluations >= self.budget:
                    break

                velocity = beta * velocity - step_size * grad
                perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
                x_new = x + velocity + perturbation
                x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

                value = func(x_new)
                self.evaluations += 1

                if value < current_value:
                    current_value = value
                    x = x_new
                    step_size *= 1.2
                    perturbation_strength *= 0.8
                else:
                    step_size *= 0.5
                    perturbation_strength *= 1.1

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x