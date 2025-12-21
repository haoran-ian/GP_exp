import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_min, self.F_max = 0.5, 0.9
        self.inertia_weight_max, self.inertia_weight_min = 0.9, 0.4
        self.cognitive = 1.5
        self.social = 1.5
        self.population1 = None
        self.population2 = None
        self.velocities1 = None
        self.velocities2 = None
        self.best_positions1 = None
        self.best_positions2 = None
        self.best_scores1 = None
        self.best_scores2 = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def levy_flight(self, size, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def self_adaptive_crossover_rate(self, evaluations):
        return 0.7 + 0.3 * (evaluations / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population1 = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.population2 = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities1 = np.zeros((self.population_size, self.dim))
        self.velocities2 = np.zeros((self.population_size, self.dim))
        self.best_positions1 = np.copy(self.population1)
        self.best_positions2 = np.copy(self.population2)
        self.best_scores1 = np.full(self.population_size, np.inf)
        self.best_scores2 = np.full(self.population_size, np.inf)
        evaluations = 0

        while evaluations < self.budget:
            generation = evaluations // (2 * self.population_size)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Lévy Flight for global exploration
                L = self.levy_flight(self.dim)
                mutant_vector1 = np.clip(self.population1[i] + L * (self.population1[i] - self.global_best_position), lb, ub)
                mutant_vector2 = np.clip(self.population2[i] + L * (self.population2[i] - self.global_best_position), lb, ub)

                # Self-Adaptive Crossover
                CR = self.self_adaptive_crossover_rate(evaluations)
                crossover_mask1 = np.random.rand(self.dim) < CR
                crossover_mask2 = np.random.rand(self.dim) < CR

                trial_vector1 = np.where(crossover_mask1, mutant_vector1, self.population1[i])
                trial_vector2 = np.where(crossover_mask2, mutant_vector2, self.population2[i])

                trial_score1 = func(trial_vector1)
                trial_score2 = func(trial_vector2)
                evaluations += 2

                if trial_score1 < self.best_scores1[i]:
                    self.best_scores1[i] = trial_score1
                    self.best_positions1[i] = trial_vector1

                if trial_score2 < self.best_scores2[i]:
                    self.best_scores2[i] = trial_score2
                    self.best_positions2[i] = trial_vector2

                if trial_score1 < self.global_best_score:
                    self.global_best_score = trial_score1
                    self.global_best_position = trial_vector1

                if trial_score2 < self.global_best_score:
                    self.global_best_score = trial_score2
                    self.global_best_position = trial_vector2

            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component1 = self.cognitive * r1 * (self.best_positions1 - self.population1)
            social_component1 = self.social * r2 * (self.global_best_position - self.population1)
            inertia_weight1 = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            self.velocities1 = inertia_weight1 * self.velocities1 + cognitive_component1 + social_component1
            self.population1 = np.clip(self.population1 + self.velocities1, lb, ub)

            r3, r4 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component2 = self.cognitive * r3 * (self.best_positions2 - self.population2)
            social_component2 = self.social * r4 * (self.global_best_position - self.population2)
            inertia_weight2 = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            self.velocities2 = inertia_weight2 * self.velocities2 + cognitive_component2 + social_component2
            self.population2 = np.clip(self.population2 + self.velocities2, lb, ub)

        return self.global_best_position, self.global_best_score