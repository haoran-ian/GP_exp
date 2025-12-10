import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Size of the population
        self.inertia_weight = 0.7  # Initial inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability
        self.inertia_damp = 0.99  # Damping factor for inertia weight
        self.restart_threshold = 0.001  # Threshold for triggering a restart

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        
        evaluations = self.population_size
        stagnation_counter = 0
        previous_best = func(global_best)

        while evaluations < self.budget:
            # Update velocities and positions for PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

            # Evaluate new positions
            for i in range(self.population_size):
                new_fitness = func(pop[i])
                evaluations += 1
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < func(global_best):
                        global_best = pop[i]

            # Apply inertia weight damping
            self.inertia_weight *= self.inertia_damp

            # Differential Evolution step
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < func(global_best):
                        global_best = trial

            # Check for stagnation
            current_best = func(global_best)
            if abs(previous_best - current_best) < self.restart_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            previous_best = current_best

            # Restart mechanism if stagnation persists
            if stagnation_counter > 5:
                pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
                velocity = np.zeros((self.population_size, self.dim))
                personal_best = pop.copy()
                personal_best_fitness = np.array([func(ind) for ind in personal_best])
                global_best = personal_best[np.argmin(personal_best_fitness)]
                stagnation_counter = 0

        return global_best