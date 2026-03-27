import numpy as np

class EnhancedAdaptiveEMADEv2:
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
        
        mutation_scale = np.random.uniform(0.5, 1.0)
        adaptive_scale = 0.5
        
        while evals < self.budget:
            initial_population_fitness = np.mean(fitness)
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + mutation_scale * (b - c), lb, ub)
                
                # Adaptive crossover probability based on diversity and progress
                diversity = np.std(pop, axis=0).mean()
                progress_ratio = evals / self.budget
                CR = np.clip(0.6 * (1 - diversity / (diversity + 1e-9)) * (1 - progress_ratio), 0.3, 0.9)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                
                if np.random.rand() < 0.15:
                    trial += np.random.normal(0, 0.1 * (1 - progress_ratio), self.dim)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    adaptive_scale = min(1.0, adaptive_scale + 0.05)
                else:
                    adaptive_scale = max(0.1, adaptive_scale - 0.05)

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
            
            # Update mutation scale based on fitness improvement and adaptive scaling
            current_population_fitness = np.mean(fitness)
            if current_population_fitness < initial_population_fitness:
                mutation_scale *= 1.05 * adaptive_scale
            else:
                mutation_scale *= 0.95 * adaptive_scale

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