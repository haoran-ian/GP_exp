import numpy as np

class EnhancedMetaheuristicOptimizerV7:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 3 * dim
        self.initial_mutation_factor = 0.6
        self.crossover_probability = 0.85
        self.elite_fraction = 0.1
        self.memory = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty(self.population_size)
            
            dynamic_elite_fraction = 0.1 + 0.03 * (evaluations / self.budget)
            elite_size = max(1, int(self.population_size * dynamic_elite_fraction))
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
                
                # Change: Introduce a phase-based mutation factor
                adaptive_mutation_factor = self.initial_mutation_factor * (0.5 + 0.5 * np.cos(2 * np.pi * evaluations / self.budget))

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

            self.memory.append((population, fitness))
            self.memory = self.memory[-5:]

            fitness_std = np.std(new_fitness)
            if fitness_std > 0:
                scaled_fitness = (new_fitness - new_fitness.min()) / fitness_std
                probabilities = np.exp(-scaled_fitness)
                probabilities /= probabilities.sum()
                selected_indices = np.random.choice(self.population_size, self.population_size, p=probabilities)
                population, fitness = new_population[selected_indices], new_fitness[selected_indices]
            else:
                population, fitness = new_population, new_fitness

            diversity_threshold = np.std(population, axis=0).mean()
            if diversity_threshold < 0.05:
                past_population, past_fitness = max(self.memory, key=lambda x: np.std(x[0], axis=0).mean())
                perturbation_strength = 0.25 * (1 - evaluations/self.budget) * (1 + 0.15 * np.sin(np.pi * evaluations / self.budget))
                for i in range(self.population_size):
                    perturbation = np.random.normal(0, perturbation_strength, self.dim)
                    candidate = past_population[i] + perturbation
                    candidate = np.clip(candidate, bounds[0], bounds[1])
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < past_fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness

            elite_indices = np.argsort(fitness)[:elite_size]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]