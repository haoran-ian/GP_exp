import numpy as np

class IADE_FBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.strategy_pool = [('rand/1', 0.5, 0.9), ('best/1', 0.7, 0.7)]
        self.adaptation_period = 50

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        strategy_scores = np.zeros(len(self.strategy_pool))

        while eval_count < self.budget:
            for i in range(self.population_size):
                strategy_idx = np.argmax(np.random.multinomial(1, strategy_scores + 1e-9))
                strategy, F, CR = self.strategy_pool[strategy_idx]
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                if strategy == 'rand/1':
                    mutant = np.clip(a + F * (b - c), lb, ub)
                elif strategy == 'best/1':
                    best_idx = np.argmin(fitness)
                    mutant = np.clip(pop[best_idx] + F * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    strategy_scores[strategy_idx] += 1

                if eval_count >= self.budget:
                    break

            if eval_count % self.adaptation_period == 0:
                strategy_scores = strategy_scores / np.sum(strategy_scores)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]