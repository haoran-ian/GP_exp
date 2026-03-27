import numpy as np

class MultiPhasedAdaptiveExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.current_phase = 1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness), evaluations)
                offspring.append(self.mutate(child, lb, ub, np.std(fitness), evaluations))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            # Dynamically adjust population size and exploration phase
            population_size = self.dynamic_population_size(evaluations)
            self.current_phase = self.determine_phase(evaluations)

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, diversity, evaluations):
        cr = 0.5 + (0.3 * np.sin(np.pi * evaluations / self.budget) * diversity)
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity, evaluations):
        mutation_strength = (0.05 + 0.05 * np.cos(np.pi * evaluations / self.budget)) * diversity
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)

    def dynamic_population_size(self, evaluations):
        phases = [1, 0.8, 0.6, 0.4]
        return max(10, int(self.initial_population_size * phases[self.current_phase]))

    def determine_phase(self, evaluations):
        phase_thresholds = [0.2, 0.5, 0.8, 1.0]
        for i, threshold in enumerate(phase_thresholds):
            if evaluations / self.budget < threshold:
                return i
        return len(phase_thresholds) - 1