import numpy as np

class DynamicEnsembleDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.local_search_prob_start = 0.15
        self.adaptive_rate = 0.1
        self.temperature = 1.0  # Starting temperature for simulated annealing

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

    def _simulated_annealing(self, current, candidate, current_fitness, candidate_fitness):
        if candidate_fitness < current_fitness:
            return candidate, candidate_fitness
        else:
            acceptance_prob = np.exp((current_fitness - candidate_fitness) / self.temperature)
            if np.random.rand() < acceptance_prob:
                return candidate, candidate_fitness
            else:
                return current, current_fitness

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        local_search_prob = self.local_search_prob_start

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant)

                if np.random.rand() < local_search_prob:
                    trial = self._local_search(trial, bounds)

                trial_fitness = func(trial)
                pop[i], fitness[i] = self._simulated_annealing(pop[i], trial, fitness[i], trial_fitness)

                if fitness[i] < func(best):
                    best = pop[i]

                # Adaptive rate adjustment
                improvement = (fitness[i] < trial_fitness)
                self.F_max = min(1.0, self.F_max + self.adaptive_rate * 0.5 * improvement)
                self.CR_max = min(1.0, self.CR_max + self.adaptive_rate * 0.5 * improvement)
                self.F_min = max(0.1, self.F_min - self.adaptive_rate * (1 - improvement))
                self.CR_min = max(0.0, self.CR_min - self.adaptive_rate * (1 - improvement))
                
                # Dynamically adjust local search and temperature
                local_search_prob = max(0.05, local_search_prob - self.adaptive_rate / self.population_size)
                self.temperature *= 0.99  # Gradually cool down

        return best