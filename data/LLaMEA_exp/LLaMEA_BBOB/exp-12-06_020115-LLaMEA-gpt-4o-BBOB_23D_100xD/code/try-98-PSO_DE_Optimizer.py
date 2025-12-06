import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.fitness_evaluations = 0
        self.inertia_weight = 0.9
        self.cognitive_coef = 2.0
        self.social_coef = 2.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def initialize_population(self):
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocity = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        return position, velocity

    def evaluate_fitness(self, func, position):
        fitness = np.array([func(p) for p in position])
        self.fitness_evaluations += len(position)
        return fitness

    def update_velocity(self, velocity, position, personal_best, global_best):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        self.inertia_weight = 0.9 - (0.6 * (self.fitness_evaluations / self.budget))
        self.cognitive_coef = 2.0 - (1.0 * (self.fitness_evaluations / self.budget))
        self.social_coef = 2.0 + (1.0 * (self.fitness_evaluations / self.budget))
        cognitive_velocity = self.cognitive_coef * r1 * (personal_best - position)
        social_velocity = self.social_coef * r2 * (global_best - position)
        new_velocity = self.inertia_weight * velocity + cognitive_velocity + social_velocity
        return new_velocity

    def differential_evolution(self, position, fitness, func):
        for i in range(self.num_particles):
            if self.fitness_evaluations >= self.budget:
                break
            candidates = [n for n in range(self.num_particles) if n != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            self.mutation_factor = 0.8 + 0.3 * (self.fitness_evaluations / self.budget)
            mutant_vector = position[a] + self.mutation_factor * (position[b] - position[c])
            mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
            self.crossover_prob = 0.9 - 0.3 * (self.fitness_evaluations / self.budget)  # Altered
            trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, position[i])
            trial_fitness = func(trial_vector)
            self.fitness_evaluations += 1
            if trial_fitness < fitness[i]:
                position[i] = trial_vector
                fitness[i] = trial_fitness
        return position, fitness

    def update_global_best(self, fitness, personal_best_fitness, personal_best):
        global_best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_index]
        return global_best

    def __call__(self, func):
        position, velocity = self.initialize_population()
        fitness = self.evaluate_fitness(func, position)
        personal_best = position.copy()
        personal_best_fitness = fitness.copy()
        global_best = self.update_global_best(fitness, personal_best_fitness, personal_best)

        while self.fitness_evaluations < self.budget:
            velocity = self.update_velocity(velocity, position, personal_best, global_best)
            position = position + velocity
            position = np.clip(position, self.lower_bound, self.upper_bound)
            fitness = self.evaluate_fitness(func, position)

            update_indices = fitness < personal_best_fitness
            personal_best[update_indices] = position[update_indices]
            personal_best_fitness[update_indices] = fitness[update_indices]

            global_best = self.update_global_best(fitness, personal_best_fitness, personal_best)

            position, fitness = self.differential_evolution(position, fitness, func)

        return global_best