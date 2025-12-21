import numpy as np

class QuantumChaosDEPSO:
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

    def chaotic_map(self, size):
        x = np.random.rand(size)
        return 4 * x * (1 - x)

    def quantum_walk(self, position, lb, ub):
        step_size = np.random.normal(0, 0.1, size=position.shape)
        new_position = position + step_size
        return np.clip(new_position, lb, ub)

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

                # Quantum-inspired mutation
                mutant_vector1 = self.quantum_walk(self.population1[i], lb, ub)
                mutant_vector2 = self.quantum_walk(self.population2[i], lb, ub)

                # Chaotic local search
                chaos1 = self.chaotic_map(self.dim)
                chaos2 = self.chaotic_map(self.dim)
                trial_vector1 = np.clip(mutant_vector1 + chaos1, lb, ub)
                trial_vector2 = np.clip(mutant_vector2 + chaos2, lb, ub)

                # Evaluate both populations
                trial_score1 = func(trial_vector1)
                trial_score2 = func(trial_vector2)
                evaluations += 2

                # Update best positions and scores
                if trial_score1 < self.best_scores1[i]:
                    self.best_scores1[i] = trial_score1
                    self.best_positions1[i] = trial_vector1
                if trial_score2 < self.best_scores2[i]:
                    self.best_scores2[i] = trial_score2
                    self.best_positions2[i] = trial_vector2

                # Update global best position
                if trial_score1 < self.global_best_score:
                    self.global_best_score = trial_score1
                    self.global_best_position = trial_vector1
                if trial_score2 < self.global_best_score:
                    self.global_best_score = trial_score2
                    self.global_best_position = trial_vector2

            # Update inertia weight and velocity for PSO update
            inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component1 = self.cognitive * r1 * (self.best_positions1 - self.population1)
            social_component1 = self.social * r2 * (self.global_best_position - self.population1)
            self.velocities1 = inertia_weight * self.velocities1 + cognitive_component1 + social_component1
            self.population1 = np.clip(self.population1 + self.velocities1, lb, ub)

            r3, r4 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component2 = self.cognitive * r3 * (self.best_positions2 - self.population2)
            social_component2 = self.social * r4 * (self.global_best_position - self.population2)
            self.velocities2 = inertia_weight * self.velocities2 + cognitive_component2 + social_component2
            self.population2 = np.clip(self.population2 + self.velocities2, lb, ub)

        return self.global_best_position, self.global_best_score