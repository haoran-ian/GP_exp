import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.2
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.vel = np.random.uniform(-self.vel_max, self.vel_max, (self.population_size, self.dim))
        self.p_best = self.pop.copy()
        self.p_best_scores = np.full(self.population_size, np.inf)
        self.g_best = None
        self.g_best_score = np.inf

    def __call__(self, func):
        eval_count = 0
        
        # Evaluate initial population
        scores = np.array([func(ind) for ind in self.pop])
        eval_count += self.population_size
        
        # Update personal and global bests
        for i in range(self.population_size):
            if scores[i] < self.p_best_scores[i]:
                self.p_best_scores[i] = scores[i]
                self.p_best[i] = self.pop[i]
        
            if scores[i] < self.g_best_score:
                self.g_best_score = scores[i]
                self.g_best = self.pop[i]

        while eval_count < self.budget:
            # Hybrid step: Perform PSO
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.vel = (self.vel +
                        self.c1 * r1 * (self.p_best - self.pop) +
                        self.c2 * r2 * (self.g_best - self.pop))
            self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)

            new_pop_pso = self.pop + self.vel
            new_pop_pso = np.clip(new_pop_pso, self.lower_bound, self.upper_bound)

            # Hybrid step: Perform DE
            new_pop_de = np.empty_like(self.pop)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.pop[a] + self.mutation_factor * (self.pop[b] - self.pop[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                self.crossover_prob = 0.5 + 0.5 * np.random.rand()  # Adaptive crossover
                crossover = np.random.rand(self.dim) < self.crossover_prob
                new_pop_de[i] = np.where(crossover, mutant, self.pop[i])

            # Combine both populations
            new_pop = (new_pop_pso + new_pop_de) / 2
            new_pop = np.clip(new_pop, self.lower_bound, self.upper_bound)

            # Evaluate new population
            new_scores = np.array([func(ind) for ind in new_pop])
            eval_count += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if new_scores[i] < self.p_best_scores[i]:
                    self.p_best_scores[i] = new_scores[i]
                    self.p_best[i] = new_pop[i]

                if new_scores[i] < self.g_best_score:
                    self.g_best_score = new_scores[i]
                    self.g_best = new_pop[i]

            self.pop = new_pop

        return self.g_best