import numpy as np

class MultiSwarmAdaptiveDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.sub_swarm_count = 3  # Number of sub-swarms
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1  # Rate of elite preservation
        self.communication_rate = 0.2  # Rate of inter-swarm communication

    def adapt_parameters(self):
        # Adaptive parameter adjustment using diversity metrics
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.8 + 0.2 * np.random.rand()

    def calculate_diversity(self, population):
        # Calculate population diversity
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize multiple swarms
        swarms = [np.random.rand(self.population_size // self.sub_swarm_count, self.dim) * (ub - lb) + lb
                  for _ in range(self.sub_swarm_count)]
        fitnesses = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        self.num_evaluations += sum(len(fit) for fit in fitnesses)
        
        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            for swarm_idx, (population, fitness) in enumerate(zip(swarms, fitnesses)):
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]
                elite_size = max(1, int(self.elite_rate * len(population)))
                
                for i in range(elite_size, len(population)):
                    idxs = [idx for idx in range(len(population)) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                    
                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    
                    trial_fitness = func(trial)
                    self.num_evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i], fitness[i] = trial, trial_fitness
                    
                    if self.num_evaluations < self.budget:
                        diversity = self.calculate_diversity(population)
                        sigma = 0.1 * (ub - lb) * (1 + diversity)
                        local_candidates = trial + np.random.randn(3, self.dim) * sigma
                        local_candidates = np.clip(local_candidates, lb, ub)
                        local_fitnesses = np.array([func(cand) for cand in local_candidates])
                        self.num_evaluations += 3
                        best_local_idx = np.argmin(local_fitnesses)
                        if local_fitnesses[best_local_idx] < trial_fitness:
                            population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]
                
                # Update swarm with new solutions
                swarms[swarm_idx] = population
                fitnesses[swarm_idx] = fitness
            
            # Inter-swarm communication
            if np.random.rand() < self.communication_rate and self.num_evaluations < self.budget:
                best_solutions = [swarm[np.argmin(fit)] for swarm, fit in zip(swarms, fitnesses)]
                best_fitnesses = [np.min(fit) for fit in fitnesses]
                global_best_idx = np.argmin(best_fitnesses)
                
                for swarm_idx in range(self.sub_swarm_count):
                    if swarm_idx != global_best_idx:
                        rand_idx = np.random.randint(0, len(swarms[swarm_idx]))
                        swarms[swarm_idx][rand_idx] = best_solutions[global_best_idx]
                        fitnesses[swarm_idx][rand_idx] = best_fitnesses[global_best_idx]

        best_fitness = np.min([np.min(fit) for fit in fitnesses])
        best_solution = swarms[np.argmin([np.min(fit) for fit in fitnesses])][np.argmin([np.min(fit) for fit in fitnesses])]

        return best_solution, best_fitness