import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.strategy_prob = [0.5, 0.5]  # Probability for DE/rand/1 and DE/best/1
        self.success_rate = [0, 0]  # Success rate for each strategy

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = pop_size
        
        while eval_count < self.budget:
            new_pop = []
            new_fitness = []
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Choose strategy based on probability
                strategy = np.random.choice([0, 1], p=self.strategy_prob)
                
                # Mutation 
                idxs = [idx for idx in range(pop_size) if idx != i]
                if strategy == 0:  # DE/rand/1
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                else:  # DE/best/1
                    b, c = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(best_individual + self.F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
               
                # Update success rate
                if trial_fitness < fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                    self.success_rate[strategy] += 1
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])
            
            # Update population and fitness
            pop = np.array(new_pop)
            fitness = np.array(new_fitness)
            pop_size = len(pop)
            
            # Self-adaptive F and CR
            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)
            
            # Adapt strategy probabilities
            total_success = sum(self.success_rate)
            if total_success > 0:
                self.strategy_prob = [rate / total_success for rate in self.success_rate]

            # Adjust F and CR based on population diversity
            diversity = np.mean(np.std(pop, axis=0))
            self.F = np.clip(self.F + 0.1 * (0.5 - diversity), 0.1, 1.0)  # Adjust F
            self.CR = np.clip(self.CR + 0.1 * (0.5 - diversity), 0.1, 1.0)  # Adjust CR
        
        return best_individual, best_fitness