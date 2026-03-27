import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.init_population()
        # Adaptive parameters
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.2, 0.9
        self.current_gen = 0

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)

    def adaptive_mutation_factor(self):
        return self.F_min + (self.F_max - self.F_min) * np.random.rand()

    def adaptive_crossover_rate(self):
        return self.CR_min + (self.CR_max - self.CR_min) * np.random.rand()

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self.adaptive_mutation_factor()
        donor_vector = (
            self.population[a] + F * (self.population[b] - self.population[c])
        )
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor):
        trial = np.copy(target)
        CR = self.adaptive_crossover_rate()
        crossover_points = np.random.rand(self.dim) < CR
        trial[crossover_points] = donor[crossover_points]
        return trial

    def local_greedy_search(self, individual):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            self.current_gen += 1
            for i in range(self.population_size):
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)

                if np.random.rand() < (0.1 + 0.9 * self.current_gen / self.budget):
                    trial = self.local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if self.budget <= 0:
                    break

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution