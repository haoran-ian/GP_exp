import numpy as np

class EnhancedQuorumDifferentialEvolutionMultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.num_swarms = 3
        self.population = None
        self.swarms = []

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, swarm, bounds):
        indices = [i for i in range(len(swarm)) if i != idx]
        a, b, c = swarm[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def adapt_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.mutation_factor = 0.5 + 0.3 * (1 - progress)
        self.crossover_probability = 0.9 - 0.5 * (1 - progress)

    def quorum_sensing(self, population, fitness):
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for i in range(len(population)):
            if fitness[i] > fitness[best_idx] * (1 + self.quorum_threshold):
                population[i] = best_solution + np.random.normal(0, 0.1, self.dim)

    def initialize_swarms(self):
        swarm_size = self.population_size // self.num_swarms
        np.random.shuffle(self.population)
        self.swarms = [self.population[i*swarm_size:(i+1)*swarm_size] for i in range(self.num_swarms)]

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        self.initialize_swarms()

        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.adapt_parameters(evaluations)

            for swarm in self.swarms:
                swarm_fitness = np.array([func(ind) for ind in swarm])
                for i in range(len(swarm)):
                    mutant_vector = self.mutate(i, swarm, func.bounds)
                    trial_vector = self.crossover(swarm[i], mutant_vector)
                    trial_fitness = func(trial_vector)
                    evaluations += 1

                    if trial_fitness < swarm_fitness[i]:
                        swarm[i] = trial_vector
                        swarm_fitness[i] = trial_fitness

                    if evaluations >= self.budget:
                        break

                self.quorum_sensing(swarm, swarm_fitness)

            self.population = np.vstack(self.swarms)
            fitness = np.hstack([np.array([func(ind) for ind in swarm]) for swarm in self.swarms])

        best_idx = np.argmin(fitness)
        return self.population[best_idx]