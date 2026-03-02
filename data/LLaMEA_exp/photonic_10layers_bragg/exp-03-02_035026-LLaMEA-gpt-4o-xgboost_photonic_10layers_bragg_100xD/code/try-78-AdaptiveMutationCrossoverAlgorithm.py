import numpy as np

class AdaptiveMutationCrossoverAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 20
        memory = []

        while evaluations < self.budget:
            phase = evaluations / self.budget

            # Adjust mutation scale based on phase
            mutation_scale = self._adaptive_mutation_scale(phase, memory, best_fitness)
            crossover_probability = self._adaptive_crossover_probability(memory)

            candidate_solutions = self._generate_solutions(best_solution, lb, ub, mutation_scale, crossover_probability, population_size)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 10:
                memory.pop(0)

        return best_solution

    def _generate_solutions(self, center, lb, ub, mutation_scale, crossover_probability, population_size):
        perturbations = np.random.normal(0, mutation_scale, size=(population_size, self.dim))
        offspring = center + perturbations
        solutions = np.where(np.random.rand(population_size, self.dim) < crossover_probability, offspring, center)
        return np.clip(solutions, lb, ub)

    def _adaptive_mutation_scale(self, phase, memory, best_fitness):
        scale_factor = 0.1 + 0.4 * (1 - phase)
        if memory:
            scale_factor *= np.std(memory) / (np.abs(best_fitness) + 1e-10)
        return scale_factor / np.sqrt(self.dim)

    def _adaptive_crossover_probability(self, memory):
        if not memory:
            return 0.7
        return min(0.9, 0.5 + np.std(memory) / 10)
