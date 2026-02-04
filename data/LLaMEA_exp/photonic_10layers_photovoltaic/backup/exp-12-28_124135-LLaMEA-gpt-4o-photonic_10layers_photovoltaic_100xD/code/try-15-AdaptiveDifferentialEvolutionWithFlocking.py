import numpy as np

class AdaptiveDifferentialEvolutionWithFlocking:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.cr = 0.9  # Crossover rate
        self.f_range = (0.5, 1.0)  # Differential weight factor range
        self.flock_factor = 0.1  # Flocking attraction factor

    def __call__(self, func):
        lb = np.array(func[0].bounds.lb)
        ub = np.array(func[0].bounds.ub)
        search_range = ub - lb

        population = lb + np.random.rand(self.population_size, self.dim) * search_range
        scores = np.array([func[0](ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            f = np.random.uniform(*self.f_range)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = population[a] + f * (population[b] - population[c])

                r = np.random.rand()
                trial_vector = np.where(r < self.cr, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_score = func[0](trial_vector)
                evaluations += 1

                if trial_score < scores[i]:
                    scores[i] = trial_score
                    population[i] = trial_vector
            
            # Flocking strategy to encourage convergence
            global_best_index = np.argmin(scores)
            global_best_position = population[global_best_index]

            for i in range(self.population_size):
                if i != global_best_index:
                    attraction = self.flock_factor * (global_best_position - population[i])
                    population[i] += attraction
                    population[i] = np.clip(population[i], lb, ub)

        global_best_index = np.argmin(scores)
        return population[global_best_index], scores[global_best_index]