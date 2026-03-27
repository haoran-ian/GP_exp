import numpy as np

class EnhancedHybridDE_SA_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.9
        self.beta = 0.99
        self.eexplore_weight = 0.1

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0
        
        while eval_budget < self.budget:
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            velocity = np.zeros_like(population)

            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / T)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                # Swarm intelligence-inspired local search
                inertia = np.random.rand() * 0.5
                social = np.random.rand() * 0.5 * (global_best - population[i])
                cognitive = np.random.rand() * 0.5 * (population[i] - global_best)
                velocity[i] = inertia * velocity[i] + cognitive + social
                population[i] = np.clip(population[i] + velocity[i], bounds[:, 0], bounds[:, 1])

                # Simulated Annealing refinement
                if np.random.rand() < 0.1:
                    refined_candidate = population[i] + self.levy_flight(self.dim)
                    refined_candidate = np.clip(refined_candidate, bounds[:, 0], bounds[:, 1])
                    refined_fitness = func(refined_candidate)
                    eval_budget += 1
                    if refined_fitness < fitness[i]:
                        population[i] = refined_candidate
                        fitness[i] = refined_fitness

            T *= self.alpha * 0.95
            
            if np.random.rand() < 0.2:
                self.F = self.F * self.beta + self.eexplore_weight * np.random.rand()
                self.CR = self.CR * (self.beta + np.random.rand() * 0.05)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]