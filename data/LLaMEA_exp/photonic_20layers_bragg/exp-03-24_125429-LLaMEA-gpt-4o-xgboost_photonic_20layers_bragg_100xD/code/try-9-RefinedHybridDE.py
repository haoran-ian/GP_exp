import numpy as np

class RefinedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_min = 0.4  # Adjusted minimum differential weight
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.local_search_prob_start = 0.15
        self.adaptive_rate = 0.1
        self.elitism_rate = 0.1  # Added elitism rate

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        return bounds.lb + pop * (bounds.ub - bounds.lb)

    def _mutate(self, pop, idx, bounds):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), idx), 3, replace=False)
        F = np.random.uniform(self.F_min, self.F_max)
        mutant = pop[a] + F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _local_search(self, candidate, bounds):
        scale = np.random.uniform(0.05, 0.15)
        perturbed = candidate + scale * np.random.normal(0, 1, self.dim) * (bounds.ub - bounds.lb)
        return np.clip(perturbed, bounds.lb, bounds.ub)

    def _chaotic_map(self, x):
        return 4 * x * (1 - x)  # Logistic map

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        local_search_prob = self.local_search_prob_start
        chaotic_factor = 0.7  # Initial chaotic factor

        for gen in range(self.budget - self.population_size):
            chaotic_factor = self._chaotic_map(chaotic_factor)
            
            for i in range(int(self.population_size * (1 - self.elitism_rate))):
                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant)

                if np.random.rand() < local_search_prob:
                    trial = self._local_search(trial, bounds)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best):
                        best = trial

            # Elitism replacement
            elite_indices = np.argsort(fitness)[:int(self.elitism_rate * self.population_size)]
            pop = np.vstack((pop[elite_indices], pop[int(self.elitism_rate * self.population_size):]))

            # Adaptive rate adjustment based on improvement
            self.F_max = min(1.0, self.F_max + self.adaptive_rate * chaotic_factor)
            self.CR_max = min(1.0, self.CR_max + self.adaptive_rate * chaotic_factor)
            self.F_min = max(0.1, self.F_min - self.adaptive_rate * chaotic_factor)
            self.CR_min = max(0.0, self.CR_min - self.adaptive_rate * chaotic_factor)

            # Dynamically adjust local search probability
            local_search_prob = max(0.05, local_search_prob - self.adaptive_rate / self.population_size)

        return best