import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

class EnhancedMetaheuristicOptimizerV13:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 3 * dim
        self.initial_mutation_factor = 0.6
        self.crossover_probability = 0.85
        self.elite_fraction = 0.1
        self.memory = []
        self.lsh_radius = 0.1
        self.phase_switch_threshold = 0.5  # Threshold to switch between exploration and exploitation

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        phase = 'exploration'

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty(self.population_size)

            if phase == 'exploration':
                elite_size = max(1, int(self.population_size * self.elite_fraction))
                elite_indices = np.argsort(fitness)[:elite_size]
                new_population[:elite_size], new_fitness[:elite_size] = population[elite_indices], fitness[elite_indices]

                for i in range(elite_size, self.population_size):
                    if evaluations >= self.budget:
                        break

                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)

                    best_idx = np.argmin(fitness)
                    orthogonal_base = (population[a] + population[best_idx]) / 2
                    adaptive_mutation_factor = max(0.1, self.initial_mutation_factor * (1 - (evaluations / self.budget)))

                    mutation_decay = 0.15 + 0.85 * (fitness[i] - fitness[best_idx]) / (fitness.max() - fitness.min() + 1e-12)
                    adaptive_mutation_factor *= mutation_decay
                    adaptive_mutation_factor *= (1 + 0.4 * np.cos(0.4 * np.pi * evaluations / self.budget))

                    mutant = orthogonal_base + adaptive_mutation_factor * (population[b] - population[c])
                    mutant = np.clip(mutant, bounds[0], bounds[1])

                    cross_prob = self.crossover_probability * (np.std(fitness) / (np.mean(fitness) + 1e-12))
                    cross_points = np.random.rand(self.dim) < cross_prob
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])

                    trial_fitness = func(trial)
                    evaluations += 1

                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

            else:  # exploitation phase
                num_clusters = max(2, min(10, self.population_size // 3))
                kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state=0).fit(population)
                cluster_centers = kmeans.cluster_centers_
                for i, center in enumerate(cluster_centers):
                    if len(self.memory) > 0:
                        tree = cKDTree(self.memory)
                        distances, indexes = tree.query(center, k=1, distance_upper_bound=self.lsh_radius)
                        if distances < self.lsh_radius:
                            continue

                    closest_idx = np.argmin([np.linalg.norm(ind - center) for ind in population])
                    center_fitness = func(center)
                    self.memory.append(center)

                    if fitness[closest_idx] > center_fitness:
                        population[closest_idx] = center
                        fitness[closest_idx] = center_fitness
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

            fitness_std = np.std(new_fitness)
            if fitness_std > 0:
                scaled_fitness = (new_fitness - new_fitness.min()) / fitness_std
                probabilities = np.exp(-scaled_fitness)
                probabilities /= probabilities.sum()
                selected_indices = np.random.choice(self.population_size, self.population_size, p=probabilities)
                population, fitness = new_population[selected_indices], new_fitness[selected_indices]
            else:
                population, fitness = new_population, new_fitness

            # Switch phase based on progress
            if evaluations / self.budget > self.phase_switch_threshold:
                phase = 'exploitation' if phase == 'exploration' else 'exploration'
            
            if phase == 'exploitation' and evaluations % (self.budget // 10) == 0:
                self.memory.clear()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]