import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F = 0.8  # Mutation factor for DE
        self.CR = 0.9  # Crossover probability for DE

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v)**(1 / beta)
        return L * step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize the population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Dynamic parameter adjustment
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - 1.5) * evaluations / self.budget)
            c2 = self.c2_initial + ((2.0 - self.c2_initial) * evaluations / self.budget)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity

            # Boundary handling
            population = np.clip(population, lb, ub)

            # Differential Evolution update
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
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

            # Lévy flight local search
            if evaluations < self.budget:
                L = 0.01  # Scale factor for Lévy flight
                for i in range(self.population_size):
                    levy_step = self.levy_flight(L)
                    candidate = personal_best[i] + levy_step
                    candidate = np.clip(candidate, lb, ub)
                    candidate_value = func(candidate)
                    evaluations += 1
                    if candidate_value < personal_best_values[i]:
                        personal_best[i] = candidate
                        personal_best_values[i] = candidate_value
                        if candidate_value < global_best_value:
                            global_best = candidate
                            global_best_value = candidate_value

        return global_best