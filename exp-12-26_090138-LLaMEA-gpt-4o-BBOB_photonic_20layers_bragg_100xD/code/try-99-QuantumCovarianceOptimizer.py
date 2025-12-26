import numpy as np

class QuantumCovarianceOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for the algorithm
        population_size = 20 * self.dim
        F_base = 0.5
        CR_base = 0.9
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions
        best_mem = []
        covariance_matrices = [np.eye(self.dim) for _ in range(population_size)]
        
        # Quantum-inspired dynamics
        quantum_population = population.copy()
        quantum_fitness = fitness.copy()

        while eval_count < self.budget:
            if eval_count % (2 * population_size) == 0:  # Dynamic resizing
                diversity = np.mean(np.std(population, axis=0))
                if diversity < 0.05:
                    population_size = max(self.dim, int(0.9 * population_size))
                else:
                    population_size = min(25 * self.dim, int(1.1 * population_size))
                quantum_population = np.copy(population)
                quantum_fitness = np.copy(fitness)

            for i in range(population_size):
                # Adaptive parameter selection based on historical population
                F = F_base + np.random.normal(0, 0.1)
                CR = CR_base + np.random.uniform(-0.1, 0.1)

                # Covariance-based mutation
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                cov_matrix = covariance_matrices[i]
                mutation_vector = F * np.dot(np.random.randn(self.dim), cov_matrix)
                mutant = np.clip(a + mutation_vector, bounds[:, 0], bounds[:, 1])
                
                # Quantum-inspired population crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial candidate
                f_trial = func(trial)
                eval_count += 1

                # Selection and update covariance strategy
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    covariance_matrices[i] = np.cov(population.T)
                    if len(best_mem) < 5 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:5]

                # Dynamic quantum-inspired update
                if eval_count < self.budget and quantum_fitness[i] > f_trial:
                    quantum_population[i] = trial
                    quantum_fitness[i] = f_trial

                # Differential elitism adjustment
                if eval_count % population_size == 0:
                    elite_index = np.argmin(fitness)
                    for j in range(population_size):
                        if j != elite_index:
                            population[j] += 0.1 * (population[elite_index] - population[j])
                            population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                            covariance_matrices[j] = np.cov(population.T)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]