import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.opposition_probability = 0.3  # Probability to perform opposition-based learning
        self.dynamic_pop_adaptation_rate = 0.1  # Rate at which the population size can adapt

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = pop_size

        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Opposition-based learning
                if np.random.rand() < self.opposition_probability:
                    opposition_candidate = lb + ub - pop[i]
                    opposition_fitness = func(opposition_candidate)
                    eval_count += 1
                    if opposition_fitness < fitness[i]:
                        pop[i] = opposition_candidate
                        fitness[i] = opposition_fitness

                # Mutation
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                
                # Adaptive F and CR
                if trial_fitness < fitness[i]:
                    self.F = min(1.0, self.F + 0.1 * (np.random.rand() - 0.5))
                    self.CR = min(1.0, self.CR + 0.1 * (np.random.rand() - 0.5))
                else:
                    self.F = max(0.1, self.F - 0.1 * (np.random.rand() - 0.5))
                    self.CR = max(0.1, self.CR - 0.1 * (np.random.rand() - 0.5))

            # Dynamic population adaptation
            if eval_count < self.budget:
                fitness_std = np.std(fitness)
                if fitness_std < 0.01:
                    pop_size = max(4, int(pop_size * (1 - self.dynamic_pop_adaptation_rate)))
                    pop = pop[:pop_size]
                    fitness = fitness[:pop_size]
                elif fitness_std > 0.05:
                    new_individuals = np.random.uniform(lb, ub, (int(pop_size * self.dynamic_pop_adaptation_rate), self.dim))
                    pop = np.vstack((pop, new_individuals))
                    fitness = np.append(fitness, [func(ind) for ind in new_individuals])
                    pop_size = len(pop)
                    eval_count += len(new_individuals)

        return best_individual, best_fitness