import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_initial_size = 30
        self.population_variability = 5
        self.F_base = 0.8
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.population_initial_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, population.shape)
        
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        elite = global_best

        while evaluations < self.budget:
            # Update inertia weight
            w = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            # Dynamic mutation factor
            avg_fitness = np.mean(fitness)
            F = self.F_base * (1 - (global_best_fitness / avg_fitness))

            # Adjust population size dynamically
            if evaluations % (self.budget // 4) == 0:
                population_size = max(20, self.population_initial_size + np.random.randint(-self.population_variability, self.population_variability))
                if population_size > len(population):
                    new_individuals = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                    population = np.concatenate((population, new_individuals))
                    velocities = np.concatenate((velocities, np.random.uniform(-1, 1, new_individuals.shape)))
                    fitness = np.concatenate((fitness, [func(ind) for ind in new_individuals]))
                    personal_best = np.concatenate((personal_best, new_individuals))
                    personal_best_fitness = np.concatenate((personal_best_fitness, [func(ind) for ind in new_individuals]))
                    evaluations += len(new_individuals)
                else:
                    population = population[:population_size]
                    velocities = velocities[:population_size]
                    fitness = fitness[:population_size]
                    personal_best = personal_best[:population_size]
                    personal_best_fitness = personal_best_fitness[:population_size]

            # Differential Evolution Phase
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness

            # Particle Swarm Optimization Phase
            for i in range(population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                new_fitness = func(population[i])
                evaluations += 1
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < personal_best_fitness[i]:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = new_fitness
                        if new_fitness < global_best_fitness:
                            global_best = population[i]
                            global_best_fitness = new_fitness

            # Enhanced elitism strategy
            if global_best_fitness < func(elite):
                elite = global_best

        return elite, func(elite)