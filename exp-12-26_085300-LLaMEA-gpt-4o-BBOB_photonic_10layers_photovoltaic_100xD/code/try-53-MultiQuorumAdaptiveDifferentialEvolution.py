import numpy as np

class MultiQuorumAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, population, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def adapt_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.mutation_factor = 0.5 + 0.3 * (1 - progress)
        self.crossover_probability = 0.9 - 0.5 * (1 - progress)

    def multi_quorum_sensing(self, population, fitness):
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        mean_fitness = np.mean(fitness)
        for i in range(self.population_size):
            if fitness[i] > mean_fitness * (1 + self.quorum_threshold):
                population[i] = best_solution + np.random.normal(0, 0.1, self.dim)

    def gradient_local_search(self, solution, func, bounds):
        grad_step = 1e-2
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            pos_solution = np.copy(solution)
            pos_solution[i] += grad_step
            grad[i] = (func(pos_solution) - func(solution)) / grad_step
        new_solution = solution - grad_step * grad
        new_solution = np.clip(new_solution, bounds.lb, bounds.ub)
        return new_solution

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)
            for i in range(self.population_size):
                mutant_vector = self.mutate(i, self.population, func.bounds)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.multi_quorum_sensing(self.population, fitness)

            # Apply gradient-based local search
            for i in range(self.population_size):
                local_search_solution = self.gradient_local_search(self.population[i], func, func.bounds)
                local_search_fitness = func(local_search_solution)
                evaluations += 1

                if local_search_fitness < fitness[i]:
                    self.population[i] = local_search_solution
                    fitness[i] = local_search_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return self.population[best_idx]