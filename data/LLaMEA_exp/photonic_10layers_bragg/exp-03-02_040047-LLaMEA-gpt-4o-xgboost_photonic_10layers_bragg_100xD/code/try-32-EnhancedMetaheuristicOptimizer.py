import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.lévy_exponent = 1.5
        self.chaos_control_factor = 0.5
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.contextual_awareness_factor = 0.9
        self.memory_size = 5
        self.subgroup_factor = 0.1  # New parameter for subgrouping
        self.local_search_intensity = 0.1  # Local search intensity factor

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def lévy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.lévy_exponent)
        return step

    def dynamic_adjustment(self, evaluations, budget):
        progress = evaluations / budget
        return self.contextual_awareness_factor * (1 - progress)

    def adaptive_memory(self, fitness_history):
        if len(fitness_history) < self.memory_size:
            return np.mean(fitness_history)
        else:
            return np.mean(fitness_history[-self.memory_size:])

    def local_search(self, candidate, lb, ub):
        perturbation = np.random.normal(0, self.local_search_intensity, self.dim)
        return np.clip(candidate + perturbation, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        fitness_history = [best_fitness]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            chaotic_seq = self.chaotic_sequence(population_size)
            lévy_steps = self.lévy_flight(population_size)

            for i in range(population_size):
                if chaotic_seq[i] < self.chaos_control_factor:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    mutant = best_solution + lévy_steps[i] * np.random.normal(0, 0.1, self.dim)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                trial = self.local_search(trial, lb, ub)  # Apply local search

                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            sorted_indices = np.argsort(fitness)
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = sorted_indices[:elite_count]

            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_population[np.argmin(new_fitness)]
                self.scaling_factor += self.dynamic_adjustment(evaluations, self.budget)

            fitness_history.append(best_fitness)
            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = self.adaptive_memory(fitness_history)

            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                subgroup_size = int(population_size * self.subgroup_factor)
                for j in range(0, population_size, subgroup_size):
                    subgroup = population[j:j + subgroup_size]
                    subgroup_fitness = fitness[j:j + subgroup_size]
                    new_members = np.random.uniform(lb, ub, (subgroup_size, self.dim))
                    population[j:j + subgroup_size] = np.where(
                        new_fitness[j:j + subgroup_size][:, None] < subgroup_fitness[:, None],
                        new_population[j:j + subgroup_size],
                        subgroup
                    )
                    fitness[j:j + subgroup_size] = np.apply_along_axis(func, 1, population[j:j + subgroup_size])
                    evaluations += subgroup_size

            self.scaling_factor = np.std(fitness) * 0.5

        return best_solution