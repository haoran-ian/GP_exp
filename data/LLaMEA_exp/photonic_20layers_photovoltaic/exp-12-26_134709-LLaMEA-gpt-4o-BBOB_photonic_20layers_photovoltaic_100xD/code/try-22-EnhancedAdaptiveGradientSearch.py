import numpy as np

class EnhancedAdaptiveGradientSearch:
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

            step_size = 0.1 * (bounds[:, 1] - bounds[:, 0])  # Initial step size
            perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])
            decay_rate = 0.99  # Decay rate for annealing perturbation
            
            while self.evaluations < self.budget:
                grad = self.estimate_gradient(func, x)
                if self.evaluations >= self.budget:
                    break
                
                perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
                x_new = x - step_size * grad + perturbation
                x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

                value = func(x_new)
                self.evaluations += 1

                if value < current_value:
                    current_value = value
                    x = x_new
                    step_size *= 1.1  # Less aggressive increase
                    perturbation_strength *= decay_rate  # Annealing strategy
                else:
                    step_size *= 0.6  # More aggressive reduction
                    perturbation_strength /= decay_rate

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x