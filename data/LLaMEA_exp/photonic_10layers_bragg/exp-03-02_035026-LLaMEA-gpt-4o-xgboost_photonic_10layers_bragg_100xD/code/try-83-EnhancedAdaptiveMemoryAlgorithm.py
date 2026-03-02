import numpy as np

class EnhancedAdaptiveMemoryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 10
        learning_rate = 0.1
        memory = []

        while evaluations < self.budget:
            phase = evaluations / self.budget

            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.8 * self._dynamic_scaling(phase, memory, best_fitness) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate, crossover_rate=0.8, mutation_rate=0.2)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate, crossover_rate=0.5, mutation_rate=0.3)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate, crossover_rate=0.3, mutation_rate=0.1)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 10:
                memory.pop(0)

            # Adjust population size and learning rate dynamically based on convergence speed and diversity
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate, crossover_rate, mutation_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        solutions = np.clip(solutions, lb, ub)
        
        # Crossover
        for i in range(population_size // 2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                parent1, parent2 = solutions[2 * i], solutions[2 * i + 1]
                solutions[2 * i, :crossover_point] = parent2[:crossover_point]
                solutions[2 * i + 1, crossover_point:] = parent1[crossover_point:]
        
        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_vector = np.random.normal(0, scale, size=self.dim)
                solutions[i] = np.clip(solutions[i] + mutation_vector, lb, ub)

        return solutions

    def _fitness_variance(self, memory):
        if not memory:
            return 0.1
        return max(0.1, np.std(memory) / 10)

    def _dynamic_scaling(self, phase, memory, best_fitness):
        if not memory:
            return max(0.1, np.abs(best_fitness) / (10 * (1 + phase)))
        return max(0.1, np.mean(memory) / (10 * (1 + phase)))