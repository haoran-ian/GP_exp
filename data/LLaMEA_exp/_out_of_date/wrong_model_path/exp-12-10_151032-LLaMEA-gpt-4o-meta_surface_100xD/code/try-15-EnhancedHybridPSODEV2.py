import numpy as np

class EnhancedHybridPSODEV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F = 0.8  # Mutation factor for DE
        self.CR = 0.9  # Crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        # Initialize the population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.initial_population_size
        
        while evaluations < self.budget:
            # Dynamic parameter adjustment
            progress_ratio = evaluations / self.budget
            w = self.w_max - ((self.w_max - self.w_min) * progress_ratio)
            c1 = self.c1_initial - ((self.c1_initial - 1.5) * progress_ratio)
            c2 = self.c2_initial + ((2.0 - self.c2_initial) * progress_ratio)
            # Adaptive population size
            population_size = int(self.initial_population_size * (1 + progress_ratio / 2))
            if population_size > self.initial_population_size:
                additional_pop = np.random.uniform(lb, ub, (population_size - self.initial_population_size, self.dim))
                additional_vel = np.random.uniform(-1, 1, (population_size - self.initial_population_size, self.dim))
                population = np.vstack((population, additional_pop))
                velocity = np.vstack((velocity, additional_vel))
                personal_best = np.vstack((personal_best, additional_pop))
                additional_best_values = np.array([func(ind) for ind in additional_pop])
                personal_best_values = np.concatenate((personal_best_values, additional_best_values))
                evaluations += (population_size - self.initial_population_size)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity

            # Boundary handling
            population = np.clip(population, lb, ub)

            # Differential Evolution update
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if evaluations >= self.budget:
                    break

        return global_best