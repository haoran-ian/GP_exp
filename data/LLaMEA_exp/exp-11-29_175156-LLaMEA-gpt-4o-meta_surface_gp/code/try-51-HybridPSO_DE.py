import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                value = func(self.population[i])
                evaluations += 1
                if value < self.personal_best_value[i]:
                    self.personal_best_value[i] = value
                    self.personal_best_position[i] = self.population[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.population[i]

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coeff * r1 * (self.personal_best_position[i] - self.population[i]) +
                                    self.social_coeff * r2 * (self.global_best_position - self.population[i])) * (1 + 0.1 * np.sin(evaluations))
                self.population[i] = np.clip(self.population[i] + self.velocity[i] * (1 + 0.05 * np.cos(evaluations)), self.lower_bound, self.upper_bound)

            self.inertia_weight = max(0.3, self.inertia_weight * (0.93 + 0.03 * np.sin(evaluations * 3)))  # Minor adjustment
            self.F = 0.6 + 0.4 * np.cos(evaluations * np.pi / self.budget)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                trial_value = func(trial)
                evaluations += 1
                if trial_value < self.personal_best_value[i]:
                    self.population[i] = trial
                    self.personal_best_value[i] = trial_value
                    self.personal_best_position[i] = trial
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial

            self.social_coeff = 1.5 + 0.1 * np.sin(evaluations * np.pi / self.budget)  # Changed to sine scheduling
            self.CR = 0.9 - 0.4 * evaluations / self.budget

        return self.global_best_position, self.global_best_value