import numpy as np

class QuantumInspiredHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 20
        F_init = 0.8
        CR_init = 0.9
        T0 = 1.0
        alpha = 0.98
        elitism_rate = 0.1
        quantum_prob = 0.1

        population = np.random.rand(pop_size, self.dim)
        for i in range(pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)

        fitness = np.array([func(ind) for ind in population])
        num_evaluations = pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0

        while num_evaluations < self.budget:
            F = F_init * (1 - num_evaluations / self.budget)
            CR = CR_init * (1 - num_evaluations / self.budget)

            elite_count = int(elitism_rate * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                if np.random.rand() < quantum_prob:
                    quantum_vector = np.random.rand(self.dim) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
                    trial = np.where(np.random.rand(self.dim) < CR, quantum_vector, population[i])
                else:
                    crossover_mask = np.random.rand(self.dim) < CR
                    trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break

            population[:elite_count] = elites
            temperature *= alpha

        return best_solution