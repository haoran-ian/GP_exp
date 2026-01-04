import numpy as np

class LSHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f = 0.8
        self.temperature = 100
        self.cooling_rate = 0.99
        self.eval_count = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                # Local Search: Fine-tuning using gradient-free approach
                if self.eval_count < self.budget:
                    local_search_candidate = trial + np.random.normal(0, 0.1, self.dim)
                    local_search_candidate = np.clip(local_search_candidate, lb, ub)
                    local_fitness = func(local_search_candidate)
                    self.eval_count += 1
                    if local_fitness < trial_fitness:
                        new_population[-1] = local_search_candidate
                        fitness[i] = local_fitness

                if self.eval_count >= self.budget:
                    break

            population = np.array(new_population)
            self.temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]