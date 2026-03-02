import numpy as np

class AdvancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.lévy_exponent = 1.5
        self.chaos_control_factor = 0.5
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.contextual_awareness_factor = 0.9
        self.memory_size = 5
        self.enhanced_exploration_factor = 0.2
        self.swarm_inertia = 0.5     # New swarm intelligence factor
        self.cognitive_coeff = 1.5   # New cognitive coefficient
        self.social_coeff = 1.5      # New social coefficient

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def lévy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.lévy_exponent)
        return step

    def dynamic_adjustment(self, evaluations, budget):
        progress = evaluations / budget
        return self.contextual_awareness_factor * (1 - progress)

    def adaptive_memory(self, fitness_history):
        if len(fitness_history) < self.memory_size:
            return np.mean(fitness_history)
        else:
            return np.mean(fitness_history[-self.memory_size:])

    def enhanced_exploration(self, progress):
        return self.enhanced_exploration_factor / (1 + progress)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim)) * (ub - lb) * 0.1
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        personal_best_pos = np.copy(population)
        personal_best_fit = np.copy(fitness)
        fitness_history = [best_fitness]

        while evaluations < self.budget:
            chaotic_seq = self.chaotic_sequence(population_size)
            lévy_steps = self.lévy_flight(population_size)

            for i in range(population_size):
                if chaotic_seq[i] < self.chaos_control_factor:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    enhanced_factor = self.enhanced_exploration(evaluations / self.budget)
                    mutant = (best_solution + lévy_steps[i] * np.random.normal(0, 0.1, self.dim)
                              * enhanced_factor)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_candidate = np.clip(trial, lb, ub)
                
                new_fitness = func(new_candidate)
                if new_fitness < personal_best_fit[i]:
                    personal_best_pos[i] = new_candidate
                    personal_best_fit[i] = new_fitness

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_candidate

                velocities[i] = (self.swarm_inertia * velocities[i] +
                                 self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_pos[i] - population[i]) +
                                 self.social_coeff * np.random.rand(self.dim) * (best_solution - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

            evaluations += population_size
            fitness_history.append(best_fitness)

            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = self.adaptive_memory(fitness_history)

            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                velocities = np.vstack((velocities, np.random.uniform(-1, 1, (population_size - len(population), self.dim)) * (ub - lb) * 0.1))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                personal_best_pos = np.vstack((personal_best_pos, new_members))
                personal_best_fit = np.hstack((personal_best_fit, fitness[-len(new_members):]))
                evaluations += population_size - len(fitness)

            self.scaling_factor = np.std(fitness) * 0.5

        return best_solution