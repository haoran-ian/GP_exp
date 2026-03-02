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
        self.competition_intensity = 0.3  # New parameter for competition

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

    def add_stochastic_diversity(self, population, lb, ub):
        noise = np.random.normal(0, 0.1, population.shape)
        return np.clip(population + noise, lb, ub)

    def competitive_selection(self, fitness, population, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.hstack((fitness, new_fitness))
        indices = np.argsort(combined_fitness)
        selected_indices = indices[:len(population)]
        return combined_population[selected_indices], combined_fitness[selected_indices]

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
                new_population[i] = np.clip(trial, lb, ub)

            new_population = self.add_stochastic_diversity(new_population, lb, ub)
            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            population, fitness = self.competitive_selection(fitness, population, new_population, new_fitness)

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]
                self.scaling_factor += self.dynamic_adjustment(evaluations, self.budget)

            fitness_history.append(best_fitness)
            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = self.adaptive_memory(fitness_history)

            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                evaluations += population_size - len(fitness)

            self.scaling_factor = np.std(fitness) * 0.5 * self.competition_intensity

        return best_solution