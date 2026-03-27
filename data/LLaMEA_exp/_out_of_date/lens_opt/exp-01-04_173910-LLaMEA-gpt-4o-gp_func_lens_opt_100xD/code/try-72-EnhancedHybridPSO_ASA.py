import numpy as np

class EnhancedHybridPSO_ASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.initial_temperature = 1.0
        self.temperature_decay = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_value = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + vel
            pop = np.clip(pop, lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            # Update personal and global bests
            improvement = new_values < personal_best_value
            personal_best[improvement] = pop[improvement]
            personal_best_value[improvement] = new_values[improvement]

            if np.min(personal_best_value) < global_best_value:
                global_best = personal_best[np.argmin(personal_best_value)]
                global_best_value = np.min(personal_best_value)

            # Adaptive Simulated Annealing (ASA)
            if evaluations < self.budget:
                for i in range(self.population_size):
                    a, b, c = pop[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = np.clip(a + self.w * (b - c), lb, ub)
                    trial = np.copy(pop[i])
                    crossover = np.random.rand(self.dim) < np.exp(-1.0 / temperature)
                    trial[crossover] = mutant[crossover]
                    trial_value = func(trial)
                    evaluations += 1

                    # Annealing acceptance criterion
                    if trial_value < new_values[i] or np.random.rand() < np.exp((new_values[i] - trial_value) / temperature):
                        pop[i] = trial
                        new_values[i] = trial_value
                        if trial_value < personal_best_value[i]:
                            personal_best[i] = trial
                            personal_best_value[i] = trial_value
                            if trial_value < global_best_value:
                                global_best = trial
                                global_best_value = trial_value

            temperature *= self.temperature_decay

        return global_best