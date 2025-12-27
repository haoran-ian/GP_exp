import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, pop_size_init=10, F_init=0.5, CR=0.9):
        pop_size = pop_size_init
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
                CR_dynamic = 0.9 * (1 - (self.evaluations / self.budget))
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
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial
                        F = min(1.0, F + 0.1)  # Increase mutation factor if improvement
                else:
                    F = max(0.1, F * 0.9)  # Decrease mutation factor if no improvement

            # Dynamic population size; increase or decrease based on budget usage
            if self.evaluations < self.budget / 2:
                pop_size = min(pop_size_init * 2, self.budget - self.evaluations)
            else:
                pop_size = max(pop_size_init // 2, 1)

            # Opposition-based learning
            opp_pop = bounds[:, 0] + bounds[:, 1] - pop
            opp_fitness = np.array([func(ind) for ind in opp_pop])
            self.evaluations += pop_size

            combined_pop = np.vstack((pop, opp_pop))
            combined_fitness = np.hstack((fitness, opp_fitness))
            best_indices = np.argsort(combined_fitness)[:pop_size]
            pop = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

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

            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=self.dim)
            x_new = x - step_size * grad + perturbation
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

            value = func(x_new)
            self.evaluations += 1

            if value < current_value:
                current_value = value
                x = x_new
                step_size = min(step_size * 1.2, 1.0)  # Adaptive step size
                perturbation_strength *= 0.8
            else:
                step_size *= 0.5
                perturbation_strength *= 1.1

            if current_value < best_value:
                best_value = current_value
                best_x = x

        return best_x