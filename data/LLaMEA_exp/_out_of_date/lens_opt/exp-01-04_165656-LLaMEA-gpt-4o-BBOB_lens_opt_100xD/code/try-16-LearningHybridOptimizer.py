import numpy as np

class LearningHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)
        self.mutation_memory = np.full((self.initial_population_size, self.dim), 0.5)  # Memory for mutation factors

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(4, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def update_mutation_memory(self, idx, success):
        learning_rate = 0.1
        if success:
            self.mutation_memory[idx] = np.minimum(1.0, self.mutation_memory[idx] + learning_rate * (1.0 - self.mutation_memory[idx]))
        else:
            self.mutation_memory[idx] = np.maximum(0.0, self.mutation_memory[idx] - learning_rate * self.mutation_memory[idx])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]

        evaluations = population_size

        while evaluations < self.budget:
            population_size = self.dynamic_population_resizing(evaluations)
            new_population = []
            diversity = np.mean(np.std(population, axis=0))
            dynamic_crossover_prob = max(0.5, min(1.0, 1.5 * diversity))
            
            for i in range(population_size):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutation_factor = np.clip(self.mutation_memory[i] + 0.1 * np.random.randn(self.dim), 0.4, 1.0)
                mutant = np.clip(a + mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < dynamic_crossover_prob
                trial = np.where(crossover, mutant, population[i % len(population)])

                trial_fitness = func(trial)
                evaluations += 1
                success = False
                if trial_fitness < fitness[i % len(fitness)] or np.random.rand() < np.exp((fitness[i % len(fitness)] - trial_fitness) / self.temp_schedule(evaluations)):
                    new_population.append(self.local_search(trial, lb, ub))
                    if len(fitness) > i:
                        fitness[i] = trial_fitness
                    success = True
                else:
                    new_population.append(population[i % len(population)])

                self.update_mutation_memory(i % len(population), success)

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = fitness[:population_size]

        return global_best