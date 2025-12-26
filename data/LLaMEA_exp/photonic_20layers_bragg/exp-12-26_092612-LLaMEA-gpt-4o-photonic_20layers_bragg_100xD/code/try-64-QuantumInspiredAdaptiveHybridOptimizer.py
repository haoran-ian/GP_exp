import numpy as np

class QuantumInspiredAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25
        self.initial_temp = 100.0
        self.cooling_rate = 0.9
        self.bounds = None
        self.elitism_rate = 0.15
        self.groups = 4  # Hierarchical grouping
        self.levy_alpha = 1.5  # Lévy flight parameter
        self.tunneling_strength = 0.01

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _adaptive_mutation(self, fitness, f_min=0.2, f_max=0.7):
        norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-6)
        return f_min + (f_max - f_min) * (1 - norm_fitness)

    def _levy_flight(self, size):
        return np.random.standard_cauchy(size) * self.levy_alpha

    def _quantum_superposition(self, population):
        # Simulate superposition by averaging with random perturbations
        mean_position = np.mean(population, axis=0)
        return mean_position + np.random.normal(0, self.tunneling_strength, self.dim)

    def _mutate(self, target_idx, population, fitness):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._adaptive_mutation(fitness[target_idx])
        mutant = population[a] + f * (population[b] - population[c]) + self._levy_flight(self.dim)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.6):
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate, t):
        # Stochastic tunneling acceptance
        delta = candidate - current
        return 1.0 if delta < 0 else np.exp(-delta / (t * self.tunneling_strength))

    def _anneal(self, candidate, current, func, temperature):
        candidate_fitness = func(candidate)
        if self._acceptance_probability(func(current), candidate_fitness, temperature) > np.random.rand():
            return candidate, candidate_fitness
        return current, func(current)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temp
        elite_size = int(self.elitism_rate * self.population_size)

        while evaluations < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Dynamic grouping based on current evaluations
            current_groups = max(2, int(self.groups * (1 - evaluations / self.budget)))

            for g in range(current_groups):
                group_indices = range(g, self.population_size, current_groups)

                for i in group_indices:
                    if i < elite_size:
                        continue  # Preserve elite

                    quantum_position = self._quantum_superposition(population)
                    mutant = self._mutate(i, population, fitness) * quantum_position
                    trial = self._crossover(population[i], mutant)
                    new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            combined_pop = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]