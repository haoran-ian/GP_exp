import numpy as np
from sklearn.cluster import KMeans

class EGLAHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.learning_rate = 0.1
        self.automata_rewards = np.ones(2) / 2  # Two actions: global or local

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            num_clusters = min(5, population_size)
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                new_individuals = self.dynamic_mutation_strategy(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)

                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]

                self.update_automata_rewards(cluster_fitness, new_fitness)
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def dynamic_mutation_strategy(self, individual, func):
        action = np.random.choice([0, 1], p=self.automata_rewards)
        if action == 0:
            scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        else:
            scale = max(0.05, 0.5 - self.evaluations / self.budget)

        perturbations = scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(10, self.dim)
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population

    def update_automata_rewards(self, old_fitness, new_fitness):
        improvement = np.sum(new_fitness < np.min(old_fitness))
        if improvement > 0:
            reward = self.learning_rate * improvement / len(new_fitness)
            self.automata_rewards += reward * (1 - self.automata_rewards)
        else:
            penalty = self.learning_rate * (1 - improvement / len(new_fitness))
            self.automata_rewards -= penalty * self.automata_rewards
        self.automata_rewards /= np.sum(self.automata_rewards)