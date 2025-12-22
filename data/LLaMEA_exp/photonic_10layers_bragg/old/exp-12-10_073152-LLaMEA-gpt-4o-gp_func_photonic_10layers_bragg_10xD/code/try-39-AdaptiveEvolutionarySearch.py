import numpy as np

class AdaptiveEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        population_size = max(4, self.dim + 1)  # At least a small diverse population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        values = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        archive = []

        while self.evaluations < self.budget:
            indices = np.arange(population_size)
            np.random.shuffle(indices)
            
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break

                idxs = [idx for idx in indices if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), lb, ub)

                # Crossover with adaptive blending
                crossover_prob = 0.9 * np.exp(-0.1 * self.evaluations / self.budget)
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])
                
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < values[i]:
                    population[i] = trial
                    values[i] = trial_value

                    if trial_value < best_value:
                        best_solution = trial
                        best_value = trial_value

                # Archive elite solutions
                archive.append((trial, trial_value))
                archive = sorted(archive, key=lambda x: x[1])[:population_size]

            # Periodically introduce elite solutions back into the population
            if self.evaluations < self.budget and self.evaluations % (population_size * 2) == 0:
                elite = np.array([sol for sol, val in archive[:population_size]])
                population = elite + np.random.normal(0, 0.1 * (ub - lb), elite.shape)

        return best_solution