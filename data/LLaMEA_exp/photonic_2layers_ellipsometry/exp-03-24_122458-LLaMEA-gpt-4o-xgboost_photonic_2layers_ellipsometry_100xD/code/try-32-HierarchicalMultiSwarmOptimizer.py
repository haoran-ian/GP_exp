import numpy as np

class HierarchicalMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temp_initial = 1.0
        self.temp_final = 0.01
        self.evaluations = 0
        self.elite_fraction = 0.1
        self.num_swarms = 3
        self.velocity_scaling = 0.5

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def mutate(self, individual, bounds, mutation_rate=0.1):
        dynamic_mutation_rate = mutation_rate * (1 - self.evaluations / self.budget) ** 2
        dynamic_scale = 0.5 + 0.5 * (self.evaluations / self.budget)
        if np.random.rand() < dynamic_mutation_rate:
            mutation_vector = np.random.normal(0, dynamic_scale, size=self.dim)
            new_individual = individual + mutation_vector
            return np.clip(new_individual, bounds.lb, bounds.ub)
        return individual

    def crossover(self, parent1, parent2):
        beta = np.random.rand()
        return beta * parent1 + (1 - beta) * parent2

    def differential_evolution_strategy(self, target, donor, bounds, F=0.8, CR=0.9):
        dynamic_CR = CR * (1 - self.evaluations / self.budget)
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < dynamic_CR:
                trial[i] = target[i] + F * (donor[i] - target[i])
        return np.clip(trial, bounds.lb, bounds.ub)

    def simulated_annealing(self, candidate, func, bounds, temp):
        perturbed = self.mutate(candidate, bounds, mutation_rate=0.5)
        candidate_eval = func(candidate)
        perturbed_eval = func(perturbed)
        self.evaluations += 2
        if perturbed_eval < candidate_eval:
            return perturbed
        else:
            prob = np.exp((candidate_eval - perturbed_eval) / temp)
            return perturbed if np.random.rand() < prob else candidate

    def swarm_update(self, population, global_best, bounds, learning_rate=0.5):
        for i in range(len(population)):
            inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
            personal_best = population[i]
            velocity = np.random.rand(self.dim) * (personal_best - population[i]) + np.random.rand(self.dim) * (global_best - population[i])
            velocity *= inertia_weight * self.velocity_scaling
            population[i] += learning_rate * velocity
            population[i] = np.clip(population[i], bounds.lb, bounds.ub)
        return population

    def elite_migration(self, sub_swarms, bounds):
        elites = [sub_swarms[i][0] for i in range(self.num_swarms)]
        for sub_swarm in sub_swarms:
            for i in range(len(sub_swarm)):
                sub_swarm[i] = self.crossover(sub_swarm[i], elites[np.random.randint(self.num_swarms)])
                sub_swarm[i] = np.clip(sub_swarm[i], bounds.lb, bounds.ub)
        return sub_swarms
    
    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += self.population_size
        
        sub_swarms = [population[i::self.num_swarms] for i in range(self.num_swarms)]

        while self.evaluations < self.budget:
            new_swarms = []
            for sub_swarm in sub_swarms:
                new_population = []
                current_elite_fraction = self.elite_fraction * (1 - self.evaluations / self.budget)
                elites = int(current_elite_fraction * len(sub_swarm))
                sorted_indices = np.argsort([fitness[population.index(ind)] for ind in sub_swarm])
                elites_indices = sorted_indices[:elites]
                elites_population = [sub_swarm[i] for i in elites_indices]

                for _ in range(len(sub_swarm) - elites):
                    parent1 = self.select_parents(sub_swarm, [fitness[population.index(ind)] for ind in sub_swarm])
                    parent2 = self.select_parents(sub_swarm, [fitness[population.index(ind)] for ind in sub_swarm])
                    donor = self.select_parents(sub_swarm, [fitness[population.index(ind)] for ind in sub_swarm])
                    offspring = self.crossover(parent1, parent2)
                    offspring = self.differential_evolution_strategy(offspring, donor, bounds)
                    new_population.append(offspring)

                new_population.extend(elites_population)
                new_fitness = self.evaluate_population(new_population, func)
                self.evaluations += len(sub_swarm)

                for i in range(len(new_population)):
                    temperature = self.temp_initial * ((self.temp_final / self.temp_initial) ** (self.evaluations / self.budget))
                    new_population[i] = self.simulated_annealing(new_population[i], func, bounds, temperature)
                    new_fitness[i] = func(new_population[i])

                sub_swarm.extend(new_population)
                sub_swarm_fitness = fitness + new_fitness
                selected_indices = np.argsort(sub_swarm_fitness)[:len(sub_swarm)]
                sub_swarm = [sub_swarm[i] for i in selected_indices]

                current_best = min(sub_swarm_fitness)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_individual = sub_swarm[np.argmin(sub_swarm_fitness)]

                sub_swarm = self.swarm_update(sub_swarm, best_individual, bounds, learning_rate=0.5)
                new_swarms.append(sub_swarm)

            sub_swarms = self.elite_migration(new_swarms, bounds)

        return best_individual, best_fitness