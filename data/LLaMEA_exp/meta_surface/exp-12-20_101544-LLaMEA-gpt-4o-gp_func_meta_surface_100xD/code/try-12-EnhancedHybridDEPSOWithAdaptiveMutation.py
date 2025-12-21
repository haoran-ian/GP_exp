import numpy as np

class EnhancedHybridDEPSOWithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.CR = 0.9
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

    def opposition_based_learning(self, population, lb, ub):
        opp_population = lb + ub - population
        opp_population = np.clip(opp_population, lb, ub)
        return opp_population

    def mutate_and_select(self, population, best_positions, lb, ub, global_best_score):
        for i in range(self.population_size):
            F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = a + F * (b - c)
            mutant_vector = np.clip(mutant_vector, lb, ub)
            crossover_mask = np.random.rand(self.dim) < self.CR

            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, self.dim)] = True

            trial_vector = np.where(crossover_mask, mutant_vector, population[i])
            trial_score = func(trial_vector)

            if trial_score < best_scores[i]:
                best_scores[i] = trial_score
                best_positions[i] = trial_vector

            if trial_score < global_best_score:
                global_best_score = trial_score
                self.global_best_position = trial_vector

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
            opp_population1 = self.opposition_based_learning(self.population1, lb, ub)
            opp_population2 = self.opposition_based_learning(self.population2, lb, ub)

            self.mutate_and_select(opp_population1, self.best_positions1, lb, ub, self.global_best_score)
            self.mutate_and_select(opp_population2, self.best_positions2, lb, ub, self.global_best_score)

            inertia_weight1 = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            inertia_weight2 = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component1 = self.cognitive * r1 * (self.best_positions1 - self.population1)
            social_component1 = self.social * r2 * (self.global_best_position - self.population1)
            self.velocities1 = (0.5 * (inertia_weight1 + inertia_weight2)) * self.velocities1 + cognitive_component1 + social_component1
            self.population1 = np.clip(self.population1 + self.velocities1, lb, ub)

            r3, r4 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component2 = self.cognitive * r3 * (self.best_positions2 - self.population2)
            social_component2 = self.social * r4 * (self.global_best_position - self.population2)
            self.velocities2 = (0.5 * (inertia_weight1 + inertia_weight2)) * self.velocities2 + cognitive_component2 + social_component2
            self.population2 = np.clip(self.population2 + self.velocities2, lb, ub)

            evaluations += 2 * self.population_size

        return self.global_best_position, self.global_best_score