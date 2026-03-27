import numpy as np

class EnhancedGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def estimate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_step = np.copy(x)
            x_step[i] += epsilon
            gradient_estimation = (func(x_step) - func(x)) / epsilon
            grad[i] = gradient_estimation
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return grad

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_x = None
        best_value = float('inf')
        
        num_starts = 5  # Number of random starts
        elite_fraction = 0.2  # Fraction of best solutions to consider as elites
        elite_solutions = []

        for _ in range(num_starts):
            if self.evaluations >= self.budget:
                break
            
            x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=self.dim)
            current_value = func(x)
            self.evaluations += 1

            step_size = 0.15 * (bounds[:, 1] - bounds[:, 0])
            perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])
            
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
                    step_size *= 1.2
                    perturbation_strength *= 0.8
                else:
                    step_size *= 0.5
                    perturbation_strength *= 1.1

                # Record elite solutions
                elite_solutions.append((current_value, np.copy(x)))
                elite_solutions.sort(key=lambda sol: sol[0])
                elite_solutions = elite_solutions[:int(elite_fraction * len(elite_solutions)) + 1]

            if current_value < best_value:
                best_value = current_value
                best_x = x

        # Explore from elite solutions
        for _, elite_x in elite_solutions:
            if self.evaluations >= self.budget:
                break

            x = elite_x
            current_value = func(x)
            self.evaluations += 1

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
                    step_size *= 1.2
                    perturbation_strength *= 0.8
                else:
                    step_size *= 0.5
                    perturbation_strength *= 1.1

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x