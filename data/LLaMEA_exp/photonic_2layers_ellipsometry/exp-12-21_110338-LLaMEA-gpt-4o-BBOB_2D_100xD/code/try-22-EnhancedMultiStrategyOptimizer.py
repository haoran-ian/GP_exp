import numpy as np

class EnhancedMultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.num_subpopulations = 5
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.strategies = [
            self.differential_evolution, 
            self.particle_swarm, 
            self.simulated_annealing
        ]
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.performance_history = np.zeros(len(self.strategies))
        self.subpopulation_assignments = np.random.choice(len(self.strategies), self.num_subpopulations)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for sp in range(self.num_subpopulations):
                chosen_strategy_idx = self.subpopulation_assignments[sp]
                chosen_strategy = self.strategies[chosen_strategy_idx]
                subpop_indices = np.arange(sp * self.subpopulation_size, (sp + 1) * self.subpopulation_size)
                subpop = population[subpop_indices]
                subpop_fitness = fitness[subpop_indices]
                new_subpop, new_subpop_fitness = chosen_strategy(subpop, subpop_fitness, func, bounds)
                
                excess = evals + len(new_subpop_fitness) - self.budget
                if excess > 0:
                    new_subpop = new_subpop[:-excess]
                    new_subpop_fitness = new_subpop_fitness[:-excess]

                population[subpop_indices] = new_subpop
                fitness[subpop_indices] = new_subpop_fitness
                self.update_strategy_weights(new_subpop_fitness, chosen_strategy_idx)
                evals += len(new_subpop_fitness)
                if evals >= self.budget:
                    break

            # Dynamic reassignment of strategies to subpopulations
            if np.random.rand() < 0.1:
                self.subpopulation_assignments = np.random.choice(len(self.strategies), self.num_subpopulations, p=self.strategy_weights)

        return population[np.argmin(fitness)]

    def select_strategy(self):
        c = np.random.rand()
        chaotic_factor = (2 ** c) % 1
        chaos_strategy = np.argmax(chaotic_factor * self.strategy_weights)
        return chaos_strategy

    def update_strategy_weights(self, new_fitness, strategy_idx):
        improvement = np.maximum(0, np.min(new_fitness) - np.min(self.performance_history)) / np.min(new_fitness)
        self.performance_history[strategy_idx] += improvement
        self.strategy_weights = self.performance_history / self.performance_history.sum()

    def differential_evolution(self, population, fitness, func, bounds):
        F = 0.5 + 0.5 * np.random.rand() * (1 - np.mean(fitness) / np.max(fitness))
        CR = 0.9
        new_population = np.empty_like(population)
        new_fitness = np.empty(len(population))
        for i in range(len(population)):
            indices = np.random.choice(len(population), 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant = np.clip(x1 + F * (x2 - x3), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, population[i])
            new_population[i] = trial
            new_fitness[i] = func(trial)
        return new_population, new_fitness

    def particle_swarm(self, population, fitness, func, bounds):
        c1, c2 = 2.05, 2.05
        w = 0.729
        v_max = 0.2 * (bounds[:, 1] - bounds[:, 0])
        velocities = np.random.uniform(-v_max, v_max, (len(population), self.dim))
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best = population[np.argmin(fitness)]

        new_population = np.empty_like(population)
        new_fitness = np.empty(len(population))
        for i in range(len(population)):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best[i] - population[i]) +
                             c2 * r2 * (global_best - population[i]))
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            new_population[i] = np.clip(population[i] + velocities[i], bounds[:, 0], bounds[:, 1])
            fit = func(new_population[i])
            new_fitness[i] = fit
            if fit < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = new_population[i], fit
            if fit < func(global_best):
                global_best = new_population[i]
        return new_population, new_fitness

    def simulated_annealing(self, population, fitness, func, bounds):
        T0 = 1000
        alpha = 0.99
        new_population = np.empty_like(population)
        new_fitness = np.empty(len(population))
        for i, x in enumerate(population):
            T = T0
            current_solution = x
            current_fitness = fitness[i]
            for _ in range(10):
                new_solution = current_solution + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
                new_fitness_candidate = func(new_solution)
                if new_fitness_candidate < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness_candidate) / T):
                    current_solution, current_fitness = new_solution, new_fitness_candidate
                T *= alpha
            new_population[i] = current_solution
            new_fitness[i] = current_fitness
        return new_population, new_fitness