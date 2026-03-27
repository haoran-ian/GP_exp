import numpy as np

class EnhancedAdaptiveEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        # Initial learning rates and mutation scale
        learning_rates = np.random.uniform(0.6, 0.9, pop_size)
        mutation_scale = np.random.uniform(0.5, 1.0)

        success_rates = np.zeros(pop_size)  # Track success rates for adaptive crossover
        diversity_measure = lambda pop: np.std(pop, axis=0).mean()  # Population diversity

        while evals < self.budget:
            initial_population_fitness = np.mean(fitness)
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + mutation_scale * (b - c), lb, ub)

                # Dynamic crossover based on success rate and diversity
                CR = 0.3 + 0.7 * success_rates[i] + 0.1 * diversity_measure(pop)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                
                # Adaptive Gaussian perturbation
                if np.random.rand() < 0.15:
                    trial += np.random.normal(0, 0.1 * (1 - evals / self.budget), self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    learning_rates[i] = learning_rates[i] + 0.1 * (1 - learning_rates[i])
                    mutation_scale = mutation_scale * 0.9 + 0.1 * np.abs(fitness[i] - trial_fitness)
                    success_rates[i] = success_rates[i] * 0.9 + 0.1  # Update success rate
                else:
                    learning_rates[i] = learning_rates[i] - 0.1 * learning_rates[i]
                    success_rates[i] = success_rates[i] * 0.9  # Decrease success rate

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
            
            # Adapt mutation scale based on population fitness improvement
            current_population_fitness = np.mean(fitness)
            if current_population_fitness < initial_population_fitness:
                mutation_scale *= 1.05
            else:
                mutation_scale *= 0.95

            # Dynamic adjustment of population size
            if evals / self.budget > 0.2:
                pop_size = max(5, int(self.initial_population_size * (1 - (evals / self.budget) ** 1.3)))

            # Periodically introduce new random individuals
            if np.random.rand() < 0.1:
                new_individual = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_individual)
                evals += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_individual

        return self.best_solution, self.best_fitness