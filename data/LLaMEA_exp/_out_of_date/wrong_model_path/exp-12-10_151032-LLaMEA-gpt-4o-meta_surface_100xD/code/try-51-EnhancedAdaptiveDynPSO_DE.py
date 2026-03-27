import numpy as np

class EnhancedAdaptiveDynPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.max_population_size = 40
        self.w_initial = 0.9  # Initial inertia weight
        self.w_final = 0.4  # Final inertia weight
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F_initial = 0.8  # Initial Mutation factor for DE
        self.CR_initial = 0.9  # Initial Crossover probability for DE
        self.evaluations = 0

    def logistic_map(self, x):
        return 4.0 * x * (1 - x)

    def chaotic_local_search(self, individual, lb, ub):
        # Add chaotic perturbation for local search
        chaos_val = np.random.rand(self.dim)
        return np.clip(individual + 0.1 * chaos_val, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        self.evaluations += self.population_size
        chaos_param = np.random.rand()

        while self.evaluations < self.budget:
            # Dynamic parameter adjustment
            chaos_param = self.logistic_map(chaos_param)
            progress_ratio = self.evaluations / self.budget
            w = self.w_initial - (self.w_initial - self.w_final) * progress_ratio
            c1 = self.c1_initial * (1 - progress_ratio) + chaos_param * (3.0 - self.c1_initial)
            c2 = self.c2_initial * (1 - progress_ratio) + chaos_param * (3.0 - self.c2_initial)
            F = self.F_initial * chaos_param
            CR = self.CR_initial * chaos_param
            
            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, lb, ub)

            # Differential Evolution update
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial = self.chaotic_local_search(trial, lb, ub)
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if self.evaluations >= self.budget:
                    break

            # Adaptive population size adjustment
            if self.evaluations / self.budget < 0.5:
                self.population_size = min(self.max_population_size, self.population_size + 1)

        return global_best