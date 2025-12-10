import numpy as np

class HybridChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def differential_mutation(self, target_idx, f):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + f * (self.positions[b] - self.positions[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def chaotic_mutation(self, position, factor):
        mutation_scale = 0.01
        chaotic_value = self.logistic_map(factor)
        return position + mutation_scale * chaotic_value * np.random.uniform(-1, 1, size=self.dim)

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand(self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            for i in range(self.population_size):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                f = 0.5 + chaotic_factor[i] * 0.5  # Chaotic adaptation of differential weight
                mutant = self.differential_mutation(i, f)
                trial = self.chaotic_mutation(mutant, chaotic_factor[i])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.positions[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        return self.global_best_position, self.global_best_score