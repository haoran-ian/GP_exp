import numpy as np

class EDE_GIM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        # Gradient estimation variables
        delta = 1e-5
        gradient_factor = 0.01

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Select three distinct individuals for mutation
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Gradient-informed mutation
                gradient = self.estimate_gradient(func, pop[i], delta)
                enhanced_mutant = mutant + gradient_factor * gradient
                enhanced_mutant = np.clip(enhanced_mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, enhanced_mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Dynamic adaptation
                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = 0.3 + 0.7 * diversity
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)
                    gradient_factor = 0.01 + 0.1 * (1 - diversity)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    def estimate_gradient(self, func, x, delta):
        grad = np.zeros(self.dim)
        for j in range(self.dim):
            x_forward = np.copy(x)
            x_forward[j] += delta
            grad[j] = (func(x_forward) - func(x)) / delta
        return grad