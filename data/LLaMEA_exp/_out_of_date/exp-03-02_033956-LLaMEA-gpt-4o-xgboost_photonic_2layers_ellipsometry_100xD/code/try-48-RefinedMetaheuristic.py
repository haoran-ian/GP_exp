import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RefinedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.rf_alpha = 0.1  # Random forest learning rate factor
        self.reshape_probability = 0.3  # Probability for dynamic population reshaping

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size

        rf_model = RandomForestRegressor(n_estimators=10)
        history = np.hstack((population, fitness.reshape(-1, 1)))

        while budget_spent < self.budget:
            rf_model.fit(history[:, :-1], history[:, -1])
            predicted_fitness = rf_model.predict(population)
            fitness_order = np.argsort(predicted_fitness)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Random Forest Guided Search
                if np.random.rand() < self.rf_alpha:
                    trial += 0.1 * (predicted_fitness[i] - predicted_fitness.mean())

                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            # Dynamic Clustering for Exploration
            cluster_centers = self.dynamic_clustering(population, fitness_order)
            if np.random.rand() < self.reshape_probability:
                for center in cluster_centers:
                    closest = min(population, key=lambda ind: np.linalg.norm(ind - center))
                    trial = closest + np.random.normal(0, 0.1, self.dim)
                    trial_fitness = func(trial)
                    budget_spent += 1
                    if trial_fitness < fitness[np.argmin(fitness)]:
                        population[np.argmax(fitness)] = trial
                        fitness[np.argmax(fitness)] = trial_fitness

        best_index = np.argmin(fitness)
        return population[best_index]

    def dynamic_clustering(self, population, fitness_order):
        cluster_centers = []
        for start in range(0, self.population_size, 5):
            cluster = population[fitness_order[start:start+5]]
            cluster_centers.append(np.mean(cluster, axis=0))
        return np.array(cluster_centers)