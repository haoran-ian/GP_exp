import numpy as np

class EnhancedLearningDE:
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

    def update_mutation_strategy(self, success_rate, F, F_min=0.1, F_max=1.0):
        if success_rate > 0.2:
            F = min(F_max, F * 1.1)
        else:
            F = max(F_min, F * 0.9)
        return F

    def differential_evolution(self, func, bounds, pop_size=10, F_init=0.5, CR=0.9):
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.evaluations += pop_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        F = F_init

        while self.evaluations < self.budget:
            successful_mutations = 0
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                CR_dynamic = CR * (1 - (self.evaluations / self.budget)) ** 0.5
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                self.evaluations += 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    successful_mutations += 1
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial
            success_rate = successful_mutations / pop_size
            F = self.update_mutation_strategy(success_rate, F)
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