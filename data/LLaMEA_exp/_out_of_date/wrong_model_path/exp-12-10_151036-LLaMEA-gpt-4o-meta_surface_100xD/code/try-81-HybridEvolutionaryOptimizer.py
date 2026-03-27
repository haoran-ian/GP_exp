import numpy as np

class HybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_individuals = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.adaptive_factor = 0.5
        self.diversity_threshold = 0.1
        self.convergence_threshold = 0.01
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        individuals = np.random.uniform(lb, ub, (self.num_individuals, self.dim))
        fitness_values = np.array([func(x) for x in individuals])
        best_individual = individuals[np.argmin(fitness_values)]
        best_value = min(fitness_values)
        
        eval_count = self.num_individuals
        
        while eval_count < self.budget:
            new_individuals = []
            for i in range(self.num_individuals):
                if np.random.rand() < self.crossover_rate:
                    idx1, idx2 = np.random.choice(self.num_individuals, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dim)
                    new_individual = np.concatenate((individuals[idx1][:crossover_point],
                                                     individuals[idx2][crossover_point:]))
                else:
                    new_individual = individuals[i].copy()

                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.randn(self.dim) * self.adaptive_factor
                    new_individual = np.clip(new_individual + mutation_vector, lb, ub)

                new_individuals.append(new_individual)

            new_fitness_values = np.array([func(x) for x in new_individuals])
            eval_count += len(new_individuals)

            for i in range(len(new_individuals)):
                if new_fitness_values[i] < fitness_values[i]:
                    individuals[i] = new_individuals[i]
                    fitness_values[i] = new_fitness_values[i]
                    
                    if fitness_values[i] < best_value:
                        best_individual = individuals[i]
                        best_value = fitness_values[i]

            if np.std(fitness_values) < self.diversity_threshold:
                noise = np.random.randn(*individuals.shape) * self.adaptive_factor
                individuals = np.clip(individuals + noise, lb, ub)
                new_fitness_values = np.array([func(x) for x in individuals])
                eval_count += len(individuals)

                for i in range(len(individuals)):
                    if new_fitness_values[i] < best_value or np.random.rand() < 0.1:
                        best_individual = individuals[i]
                        best_value = new_fitness_values[i]
                        break

            if best_value < self.convergence_threshold:
                break

        return best_individual