import numpy as np

class AdaptiveSwarmMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 5
        self.swarm = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        self.swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.swarm)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = np.copy(self.swarm[0])

    def update_velocity(self, inertia, personal_coefficient, global_coefficient):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive = personal_coefficient * r1 * (self.personal_best - self.swarm)
        social = global_coefficient * r2 * (self.global_best - self.swarm)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, lb, ub):
        self.swarm += self.velocity
        self.swarm = np.clip(self.swarm, lb, ub)

    def non_uniform_mutation(self, particle, lb, ub, t, T, b=5):
        tau = (1 - t/T) ** b
        delta = (ub - lb) * tau * np.random.uniform(-1, 1, self.dim)
        mutant = np.clip(particle + delta, lb, ub)
        return mutant

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.swarm[i])
            if fitness < self.personal_best_values[i]:
                self.personal_best_values[i] = fitness
                self.personal_best[i] = self.swarm[i].copy()
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.swarm[i].copy()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            inertia = 0.9 - 0.9 * (evaluations / self.budget)
            personal_coefficient = 1.5 + np.random.rand()
            global_coefficient = 1.5 + np.random.rand()

            self.update_velocity(inertia, personal_coefficient, global_coefficient)
            self.update_position(lb, ub)
            self.evaluate_population(func)
            
            elite_indices = np.argsort(self.personal_best_values)[:self.elite_size]
            elites = self.swarm[elite_indices]

            for i in range(self.population_size):
                t = evaluations // self.population_size
                T = self.budget // self.population_size
                mutant = self.non_uniform_mutation(self.swarm[i], lb, ub, t, T)
                mutant_fitness = func(mutant)
                evaluations += 1
                if mutant_fitness < self.personal_best_values[i]:
                    self.swarm[i] = mutant
                    self.personal_best_values[i] = mutant_fitness
                    self.personal_best[i] = mutant
                if mutant_fitness < self.global_best_value:
                    self.global_best_value = mutant_fitness
                    self.global_best = mutant

                if evaluations % 10 == 0:
                    for elite in elites:
                        adaptive_scale = 0.05 * (1 - evaluations / self.budget)
                        enhanced_neighbor = self.non_uniform_mutation(elite, lb, ub, t, T)
                        enhanced_fitness = func(enhanced_neighbor)
                        evaluations += 1
                        if enhanced_fitness < self.global_best_value:
                            self.global_best_value = enhanced_fitness
                            self.global_best = enhanced_neighbor

                if evaluations >= self.budget:
                    break

        return self.global_best_value, self.global_best