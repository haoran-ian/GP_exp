import numpy as np

class SynergisticAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 20
        inertia_weight = 0.7  # Inertia weight for velocity update
        cognitive_coeff = 1.5  # Cognitive component for personal best
        social_coeff = 1.5  # Social component for global best
        F_init = 0.8  # Differential weight
        CR_init = 0.9  # Crossover probability
        temperature = 1.0  # Initial temperature for simulated annealing
        alpha = 0.95  # Cooling rate
        elitism_rate = 0.2  # Elitism rate
        
        # Initialize population
        population = np.random.rand(initial_pop_size, self.dim)
        velocities = np.random.randn(initial_pop_size, self.dim) * (func.bounds.ub - func.bounds.lb) * 0.1
        for i in range(initial_pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = initial_pop_size

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        best_idx = np.argmin(fitness)
        global_best = population[best_idx].copy()
        global_best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            # Update velocities and positions
            r1 = np.random.rand(initial_pop_size, self.dim)
            r2 = np.random.rand(initial_pop_size, self.dim)
            new_velocities = (inertia_weight * velocities +
                              cognitive_coeff * r1 * (personal_best - population) +
                              social_coeff * r2 * (global_best - population))
            
            population = population + new_velocities
            velocities = new_velocities

            # Differential Evolution crossover
            for i in range(initial_pop_size):
                idxs = np.random.choice(initial_pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F_init * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR_init
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1

                # Simulated Annealing acceptance
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (temperature + 0.1))
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[i]:
                            personal_best[i] = trial
                            personal_best_fitness[i] = trial_fitness

                if num_evaluations >= self.budget - 1:
                    break

            # Elitism
            elite_count = int(elitism_rate * initial_pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]
            population[:elite_count] = elites

            temperature *= alpha  # Cool down the temperature

        return global_best