import numpy as np

class StochasticGradientPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def estimate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        perturbations = np.random.normal(0, epsilon, size=x.shape)  # Using Gaussian noise for more diverse gradient estimation
        for i in range(len(x)):
            x_step = np.copy(x)
            x_step[i] += perturbations[i]
            grad[i] = (func(x_step) - func(x)) / perturbations[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return grad

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_x = None
        best_value = float('inf')
        
        num_starts = 10  # Increased number of random restarts for better coverage
        for _ in range(num_starts):
            if self.evaluations >= self.budget:
                break
            
            x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
            current_value = func(x)
            self.evaluations += 1

            step_size = 0.1 * (bounds[:, 1] - bounds[:, 0])  # Further adjusted step size for finer sensitivity
            perturbation_strength = 0.1 * (bounds[:, 1] - bounds[:, 0])
            
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
                    step_size *= 1.1  # More cautious increase in step size
                    perturbation_strength *= 0.9  # More cautious decrease in perturbation
                else:
                    step_size *= 0.6  # Reduced decrease in step size for more adaptive learning
                    perturbation_strength *= 1.05  # Reduced increase to maintain exploration potential

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x