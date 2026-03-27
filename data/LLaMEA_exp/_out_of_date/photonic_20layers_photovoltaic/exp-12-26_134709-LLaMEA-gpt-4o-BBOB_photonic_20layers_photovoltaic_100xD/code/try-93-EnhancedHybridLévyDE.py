import numpy as np

class EnhancedHybridLévyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def levy_flight(self, size, alpha=1.5):
        sigma = (np.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                 (np.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / alpha)
        return step

    def differential_evolution(self, func, bounds, pop_size=10, F_init=0.5, CR=0.9):
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.evaluations += pop_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        F = F_init

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                CR_dynamic = 0.9 * (1 - (self.evaluations / self.budget)) ** 0.5
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i]) + self.levy_flight(self.dim)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                f = func(trial)
                self.evaluations += 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial
                        F = min(1.0, F + 0.1)
            F = max(0.1, F * 0.9)
        return best

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_x = None
        best_value = float('inf')

        x = self.differential_evolution(func, bounds)
        current_value = func(x)

        step_size = 0.15 * (bounds[:, 1] - bounds[:, 0])
        perturbation_strength = 0.05 * (bounds[:, 1] - bounds[:, 0])

        while self.evaluations < self.budget:
            grad = self.estimate_gradient(func, x)
            if self.evaluations >= self.budget:
                break

            perturbation_strength = 0.05 * np.linalg.norm(grad)
            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
            x_new = x - step_size * grad + perturbation
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

            value = func(x_new)
            self.evaluations += 1

            if value < current_value:
                current_value = value
                x = x_new
                step_size = min(step_size * 1.2, 1.0)
                perturbation_strength *= 0.8
            else:
                step_size *= 0.5
                perturbation_strength *= 1.1

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x