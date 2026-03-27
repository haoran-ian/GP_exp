import numpy as np

class AdvancedHybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.individuals = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.individuals)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.learning_rate = 0.6
        self.mutation_rate = 0.05
        self.elitism_rate = 0.1
        self.resize_interval = self.budget // 10

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.individuals = lb + (ub - lb) * self.individuals
        evaluations = 0
        resize_trigger = self.resize_interval

        def chaotic_sequence(x, a=3.9):
            return a * x * (1 - x)

        chaos_factor = np.random.rand(self.population_size, self.dim)
        
        while evaluations < self.budget:
            for i, individual in enumerate(self.individuals):
                score = func(individual)
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = individual
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = individual

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.personal_best_scores)[:elite_count]
            elite_individuals = self.individuals[elite_indices]
            
            new_population = []
            for _ in range(self.population_size):
                parents = np.random.choice(elite_individuals, 2, replace=False)
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
                mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                child[mutation_mask] = lb[mutation_mask] + (ub - lb)[mutation_mask] * np.random.rand(np.sum(mutation_mask))
                new_population.append(child)

            self.individuals = np.array(new_population)
            chaos_factor = chaotic_sequence(chaos_factor)
            self.individuals += self.learning_rate * chaos_factor
            self.individuals = np.clip(self.individuals, lb, ub)

            if evaluations >= resize_trigger:
                self.population_size = max(10, int(self.initial_population_size * (1.0 - evaluations / self.budget)))
                resize_trigger += self.resize_interval
                self.individuals = self.individuals[:self.population_size]
                self.personal_best_positions = self.personal_best_positions[:self.population_size]
                self.personal_best_scores = self.personal_best_scores[:self.population_size]

        return self.global_best_position, self.global_best_score