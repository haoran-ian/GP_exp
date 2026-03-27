import numpy as np

class DynamicMultiSwarmMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.ensemble_factor = 0.2
        self.reshape_probability = 0.3
        self.entropy_threshold = 0.5
        self.phase_transition_factor = 0.12
        self.swarm_count = 3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        swarms = self.create_swarms(population)
        
        while budget_spent < self.budget:
            for swarm in swarms:
                self.optimize_swarm(swarm, fitness, lb, ub, func)
                budget_spent += len(swarm)

            if budget_spent >= self.budget:
                break

            if np.random.rand() < self.reshape_probability:
                self.reshuffle_population(swarms, fitness, lb, ub, func)
                budget_spent += len(swarms) * (self.population_size // self.swarm_count)

        best_index = np.argmin(fitness)
        return population[best_index]

    def create_swarms(self, population):
        swarm_size = self.population_size // self.swarm_count
        return [population[i * swarm_size:(i + 1) * swarm_size] for i in range(self.swarm_count)]

    def optimize_swarm(self, swarm, fitness, lb, ub, func):
        for i in range(len(swarm)):
            indices = np.random.choice(len(swarm), 3, replace=False)
            while i in indices:
                indices = np.random.choice(len(swarm), 3, replace=False)
            x0, x1, x2 = swarm[indices]
            mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)

            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, swarm[i])

            dynamic_pt_factor = self.phase_transition_factor * (1 - fitness[i] / max(fitness))
            trial += dynamic_pt_factor * np.random.normal(0, 0.1, self.dim)

            entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
            if entropy_measure < self.entropy_threshold:
                trial += np.random.normal(0, 0.05, self.dim)

            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                swarm[i] = trial
                fitness[i] = trial_fitness

    def reshuffle_population(self, swarms, fitness, lb, ub, func):
        worst_indices = np.argsort(fitness)[-len(swarms[0]):]
        new_individuals = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
        for i, idx in enumerate(worst_indices):
            swarms[i // len(swarms[0])][i % len(swarms[0])] = new_individuals[i]
            fitness[idx] = func(new_individuals[i])