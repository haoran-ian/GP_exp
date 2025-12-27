import numpy as np

class HybridAdaptiveGradientDE:
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

    def differential_evolution(self, func, bounds, pop_size=10, F=0.8, CR=0.9):
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.evaluations += pop_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                CR_dynamic = 0.9 * (1 - (self.evaluations / self.budget))  # Dynamic CR
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < CR_dynamic  # Use dynamic CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                self.evaluations += 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial
        return best

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_x = None
        best_value = float('inf')

        # Use Differential Evolution for global exploration
        x = self.differential_evolution(func, bounds)
        current_value = func(x)

        step_size = 0.15 * (bounds[:, 1] - bounds[:, 0])
        perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])

        # Refine using gradient descent
        while self.evaluations < self.budget:
            grad = self.estimate_gradient(func, x)
            if self.evaluations >= self.budget:
                break

            perturbation = np.random.uniform(-perturbation_strength * (1 + self.evaluations / self.budget), perturbation_strength * (1 + self.evaluations / self.budget), size=self.dim)
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