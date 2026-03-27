import numpy as np

class EnhancedAdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.max_population_size = 40
        self.min_population_size = 10
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.local_search_prob_start = 0.15
        self.adaptive_rate = 0.1
        self.chaotic_sequence = self._generate_chaotic_sequence()

    def _generate_chaotic_sequence(self):
        # Logistic map for chaotic sequence generation
        sequence = np.empty(self.budget)
        x = 0.7  # Initial value
        for i in range(self.budget):
            x = 4.0 * x * (1.0 - x)
            sequence[i] = x
        return sequence

    def _initialize_population(self, bounds, size):
        pop = np.random.rand(size, self.dim)
        return bounds.lb + pop * (bounds.ub - bounds.lb)

    def _mutate(self, pop, idx, bounds):
        a, b, c = np.random.choice(np.delete(np.arange(len(pop)), idx), 3, replace=False)
        F = self.chaotic_sequence[idx % self.budget] * (self.F_max - self.F_min) + self.F_min
        mutant = pop[a] + F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant, idx):
        CR = self.chaotic_sequence[(idx + 1) % self.budget] * (self.CR_max - self.CR_min) + self.CR_min
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _local_search(self, candidate, bounds):
        scale = np.random.uniform(0.05, 0.15)
        perturbed = candidate + scale * np.random.normal(0, 1, self.dim) * (bounds.ub - bounds.lb)
        return np.clip(perturbed, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        pop_size = self.initial_population_size
        pop = self._initialize_population(bounds, pop_size)
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        local_search_prob = self.local_search_prob_start
        
        for t in range(self.budget - pop_size):
            for i in range(pop_size):
                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant, i)
                
                if np.random.rand() < local_search_prob:
                    trial = self._local_search(trial, bounds)
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best):
                        best = trial
                        
                    self.F_max = min(1.0, self.F_max + self.adaptive_rate * 0.5)
                    self.CR_max = min(1.0, self.CR_max + self.adaptive_rate * 0.5)
                else:
                    self.F_min = max(0.1, self.F_min - self.adaptive_rate)
                    self.CR_min = max(0.0, self.CR_min - self.adaptive_rate)

                local_search_prob = max(0.05, local_search_prob - self.adaptive_rate / pop_size)
            
            # Dynamic resizing of the population
            if t % 10 == 0:  # Every 10 iterations
                if trial_fitness < np.mean(fitness):  # If there's improvement
                    pop_size = min(self.max_population_size, pop_size + 1)
                else:
                    pop_size = max(self.min_population_size, pop_size - 1)
                pop = self._initialize_population(bounds, pop_size)
                fitness = np.array([func(ind) for ind in pop])
        
        return best