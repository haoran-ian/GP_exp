import numpy as np

class RefinedHybridDEPSOWithChaoticInit:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.CR = 0.9
        self.F_min, self.F_max = 0.5, 0.9  # Adaptive DE parameters
        self.inertia_weight_max, self.inertia_weight_min = 0.9, 0.4  # Dynamic PSO parameters
        self.cognitive = 1.5
        self.social = 1.5
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_initialization(self, lb, ub):
        chaotic_seq = np.linspace(0.1, 0.9, self.population_size)
        chaotic_seq = np.cos(2 * np.pi * chaotic_seq)
        chaotic_seq = (chaotic_seq + 1) / 2  # Normalize to [0, 1]
        return lb + chaotic_seq[:, np.newaxis] * (ub - lb)

    def opposition_based_learning(self, population, lb, ub):
        opp_population = lb + ub - population
        opp_population = np.clip(opp_population, lb, ub)
        return opp_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = self.chaotic_initialization(lb, ub)
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)
        evaluations = 0

        while evaluations < self.budget:
            # Opposition-Based Learning
            opp_population = self.opposition_based_learning(self.population, lb, ub)
            for i in range(self.population_size):
                opp_score = func(opp_population[i])
                if opp_score < self.best_scores[i]:
                    self.best_scores[i] = opp_score
                    self.best_positions[i] = opp_population[i]
                if opp_score < self.global_best_score:
                    self.global_best_score = opp_score
                    self.global_best_position = opp_population[i]

            # Update inertia weight dynamically
            inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive Differential Evolution Mutation
                F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + F * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                # Crossover with dynamic adaptation
                self.CR = 0.5 + 0.5 * (self.global_best_score / (self.global_best_score + 1e-10))
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                # Selection
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.best_scores[i]:
                    self.best_scores[i] = trial_score
                    self.best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive * r1 * (self.best_positions - self.population)
            social_component = self.social * r2 * (self.global_best_position - self.population)
            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, lb, ub)

        return self.global_best_position, self.global_best_score