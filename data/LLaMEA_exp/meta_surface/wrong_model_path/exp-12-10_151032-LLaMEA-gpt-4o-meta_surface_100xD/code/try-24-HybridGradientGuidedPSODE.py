import numpy as np

class HybridGradientGuidedPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F = 0.8
        self.CR = 0.9
        self.gradient_step_size = 1e-5

    def approximate_gradient(self, func, x):
        grad = np.zeros(self.dim)
        fx = func(x)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += self.gradient_step_size
            grad[i] = (func(x_step) - fx) / self.gradient_step_size
        return grad

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.population_size
        
        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - 1.5) * evaluations / self.budget)
            c2 = self.c2_initial + ((2.0 - self.c2_initial) * evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, lb, ub)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if evaluations >= self.budget:
                    break

            # Gradient-based local search intensification
            for i in range(self.population_size):
                grad = self.approximate_gradient(func, personal_best[i])
                local_search_candidate = personal_best[i] - self.gradient_step_size * grad
                local_search_candidate = np.clip(local_search_candidate, lb, ub)
                local_search_value = func(local_search_candidate)
                evaluations += 1

                if local_search_value < personal_best_values[i]:
                    personal_best[i] = local_search_candidate
                    personal_best_values[i] = local_search_value
                    if local_search_value < global_best_value:
                        global_best = local_search_candidate
                        global_best_value = local_search_value

                if evaluations >= self.budget:
                    break

        return global_best