import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.w = 0.9  # initial inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.c1 = 2.0  # initial cognitive component
        self.c2 = 2.0  # initial social component
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.chaotic_sequence = self.generate_chaotic_sequence(self.population_size * 100)  # Generate a chaotic sequence for randomness

    def generate_chaotic_sequence(self, length):
        x = 0.7  # Initial value
        r = 3.9  # Chaotic parameter
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_pos = np.copy(population)
        personal_best_val = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]

        evaluations = self.population_size
        chaotic_index = 0

        while evaluations < self.budget:
            # Dynamic adaptation of parameters
            self.w = self.w_min + (0.5 * (self.budget - evaluations) / self.budget)
            self.c1 = 2 - evaluations / self.budget
            self.c2 = 2 + evaluations / self.budget

            # PSO Component with chaotic randomness
            r1 = self.chaotic_sequence[chaotic_index:chaotic_index+self.population_size*self.dim].reshape((self.population_size, self.dim))
            chaotic_index += self.population_size * self.dim
            r2 = self.chaotic_sequence[chaotic_index:chaotic_index+self.population_size*self.dim].reshape((self.population_size, self.dim))
            chaotic_index += self.population_size * self.dim

            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_pos - population)
                          + self.c2 * r2 * (global_best_pos - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component with chaotic randomness
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = self.chaotic_sequence[chaotic_index:chaotic_index+self.dim] < self.crossover_rate
                chaotic_index += self.dim
                trial = np.where(crossover, mutant, population[i])
                trial_val = func(trial)
                evaluations += 1

                if trial_val < personal_best_val[i]:
                    personal_best_pos[i] = trial
                    personal_best_val[i] = trial_val

                    if trial_val < personal_best_val[global_best_idx]:
                        global_best_idx = i
                        global_best_pos = trial

                if evaluations >= self.budget:
                    break

        return global_best_pos