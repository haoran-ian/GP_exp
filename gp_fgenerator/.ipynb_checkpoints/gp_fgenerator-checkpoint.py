
import os
import random
import operator
import numpy as np
import pandas as pd
from functools import partial
from .pset import create_pset
from .symb_regression import symb_regr
from .utils import write_file, logger2csv, fitness2df
from deap import algorithms, base, creator, tools, gp
from multiprocessing import Pool, cpu_count




#%%
class GP_func_generator:
    def __init__(self,
                 doe_x,
                 target_vector,
                 minimization: bool = True,
                 bs_ratio: float = 0.8,
                 bs_repeat: int = 2,
                 list_ela: list = [],
                 ela_min: dict = {},
                 ela_max: dict = {},
                 ela_weight: dict = {},
                 dist_metric: str = 'cityblock',
                 problem_label: str = '',
                 filepath_save: str = '',
                 tree_size: tuple = (8,12),
                 population: int = 100,
                 cxpb: float = 0.5, 
                 mutpb: float = 0.1,
                 ngen: int = 10,
                 nhof: int = 1,
                 seed: int = 1,
                 verbose: bool = True
                 ):
        # optimization
        self.doe_x = doe_x
        self.target_vector = target_vector
        self.minimization = minimization
        self.bs_ratio: float = bs_ratio
        self.bs_repeat: int = bs_repeat
        self.list_ela: list = list_ela
        self.ela_min: dict = ela_min
        self.ela_max: dict = ela_max
        self.ela_weight: dict = ela_weight
        self.dist_metric: str = dist_metric
        self.problem_label: str = problem_label if problem_label else 'problem'
        self.filepath_save: str = filepath_save if filepath_save else os.path.join(os.getcwd(), f'results_gpfg_{self.doe_x.shape[1]}d_{self.problem_label}')
        self.tree_size: tuple = tree_size
        self.population: int = population
        self.cxpb: float = cxpb
        self.mutpb: float = mutpb
        self.ngen: int = ngen
        self.nhof: int = nhof
        self.seed: int = seed
        self.verbose: bool = verbose
        self.neval = 0
        self.weight = -1.0 if self.minimization else 1.0
        self.fopt = np.inf if self.minimization else -1.0*np.inf
        self.result = pd.DataFrame()
        self.id_best = np.inf
        if not (os.path.isdir(self.filepath_save)):
            os.makedirs(self.filepath_save)
    
    #%%
    def evalSymbReg(self, individual, points):
        self.neval += 1
        f_ = partial(symb_regr, self.pset, self.target_vector, self.bs_ratio, self.bs_repeat, 
                     self.list_ela, self.ela_min, self.ela_max, self.ela_weight, self.dist_metric, self.verbose)
        fitness_ = f_(individual, points)
        self.result = pd.concat([self.result, fitness2df(fitness_, label=f'{self.neval}')], axis=0, ignore_index=True)
        if (fitness_[1] < self.fopt):
            self.fopt = fitness_[1]
            self.id_best = self.neval
        if (self.verbose):
            print(f'neval: {self.neval}, fitness: {fitness_[1]}; fopt: {self.fopt}; id_best: {self.id_best}')
        return fitness_[1],
    
    #%%    
    def __call__(self):
        np.random.seed(self.seed)
        random.seed(self.seed+10)
        self.pset = create_pset(self.doe_x)()
        creator.create("FitnessMin", base.Fitness, weights=(self.weight,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        pool = Pool(cpu_count())
        toolbox.register("map", pool.map)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.tree_size[0], max_=self.tree_size[1])
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        self.toolbox.register("evaluate", self.evalSymbReg, points=self.doe_x)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        
        pop = self.toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.nhof)
    
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
    
        pop, logger = algorithms.eaSimple(pop, self.toolbox, self.cxpb, self.mutpb, self.ngen, stats=mstats,
                                          halloffame=hof, verbose=self.verbose)
        self.result.to_csv(os.path.join(self.filepath_save, 'gpfg_opt_data.csv'), index=False)
        logger2csv(os.path.join(self.filepath_save, 'gpfg_gen.csv'), logger)
        write_file(os.path.join(self.filepath_save, 'tree_id_best.txt'), [f'id_best: {self.id_best}'])
        if (self.verbose):
            print('[GPFG] Optimization for symbolic regression done.')
        return hof, pop
# END CLASS