import numpy as np

class EnhancedHybridDEASA_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 10 * self.dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        F_init = 0.8  # Initial differential mutation factor
        CR_init = 0.9  # Initial crossover probability
        T0 = 1000  # Initial temperature for simulated annealing
        Tf = 1e-2  # Final temperature for simulated annealing

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        def adjust_population_size(current_size):
            return int(max(4, current_size * (0.9 + 0.2 * np.random.rand())))

        # Main optimization loop
        while self.evaluations < self.budget:
            # Adaptive adjustment based on diversity and convergence rate
            diversity = np.mean(np.std(population, axis=0))
            F = F_init * np.exp(-0.1 * self.evaluations / self.budget)
            CR = CR_init * np.exp(-0.1 * diversity)
            population_size = adjust_population_size(population_size)

            # Differential Evolution
            for i in range(population_size):
                # Mutation: select three distinct individuals
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.evaluations >= self.budget:
                    break

            # Adaptive Simulated Annealing
            T = T0 * (Tf / T0) ** (self.evaluations / self.budget)
            for i in range(population_size):
                # Gaussian perturbation with adaptive variance
                variance = max(1e-5, diversity)
                neighbor = population[i] + np.random.normal(0, variance, self.dim)
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = func(neighbor)
                self.evaluations += 1

                # Metropolis criterion
                if neighbor_fitness < fitness[i] or np.random.rand() < np.exp(-(neighbor_fitness - fitness[i]) / T):
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness

                if self.evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]