import numpy as np

class MultiAgentCooperativeAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.agent_learning_rate = 0.1  # Initial learning rate for agents
        self.chaotic_map_factor = 0.75  # Chaotic map influence factor
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def chaotic_map(self, x):
        return self.chaotic_map_factor * x * (1 - x)

    def levy_flight(self, size, scale=1.0):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = scale * u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0  # Initial temperature for Simulated Annealing
        agent_learning_rates = np.full(self.population_size, self.agent_learning_rate)
        chaotic_sequence = np.random.rand(self.population_size)

        while eval_budget < self.budget:
            for i in range(self.population_size):
                # Update chaotic maps
                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])

                # Differential Evolution mutation and crossover
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])  # Dynamic crossover operator

                # Simulated Annealing acceptance criterion
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (T + 1e-10))  # Adjusted selection pressure
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best_position = trial

            # Cooling schedule for Simulated Annealing
            T *= self.alpha
            
            # Cooperative learning and adaptive learning rate adjustment
            for j in range(self.population_size):
                if np.random.rand() < 0.1:  # 10% chance to perform cooperative learning
                    distance = np.linalg.norm(population[j] - self.global_best_position)
                    exploration_scale = 1.0 if distance > 0.1 else 0.5
                    exploration_adjustment = np.exp(-agent_learning_rates[j] * distance)
                    population[j] += exploration_adjustment * (self.global_best_position - population[j]) + self.levy_flight(self.dim, scale=exploration_scale)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
                # Adaptive learning rate influenced by chaotic maps
                agent_learning_rates[j] *= (1 + chaotic_sequence[j])

            # Dynamic population resizing based on progress and diversity
            if np.random.rand() < 0.05:  # Occasionally adjust population size
                improvement = np.max(fitness) - np.min(fitness)
                if improvement < 1e-6:  # If minimal progress, reduce population
                    self.population_size = max(int(self.population_size * 0.9), 5)
                else:  # Otherwise, maintain or slightly increase
                    self.population_size = min(int(self.population_size * 1.1), self.initial_population_size)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]