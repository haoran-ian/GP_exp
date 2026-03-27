import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Changed population size
        self.F = 0.5
        self.CR = 0.9
        self.restart_threshold = 0.1 * dim  # Added restart threshold

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        last_best = np.inf  # Track last best fitness for stagnation detection

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + 0.3 * np.random.random()  # Adaptive F
                mutant = np.clip(a + F * (b - c), lb, ub)  # Use adaptive F
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                if num_evaluations >= self.budget:
                    break
            # Restart mechanism if stagnation detected
            if np.abs(min(fitness) - last_best) < self.restart_threshold:
                new_population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
                fitness = np.array([func(ind) for ind in new_population])
                num_evaluations += self.pop_size
            last_best = min(fitness)
            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]