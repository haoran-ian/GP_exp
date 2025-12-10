class DMDE_PSO:
    def __init__(self, budget, dim):
        # ... [rest of the code remains unchanged]
        self.CR = 0.9  # initial crossover probability

    def __call__(self, func):
        # ... [rest of the code remains unchanged]
        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                self.CR = 0.9 - 0.4 * (self.current_evaluations / self.budget)  # dynamic crossover probability
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                # ... [rest of the loop remains unchanged]