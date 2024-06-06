import logging
import sys
from typing import List, Tuple, Union

import numpy as np
from pop import Population
from env import evaluate, select, cross, mutate
from net import Network

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class GeneticAlgorithm(object):
    """Genetic algorithm
    Chromosomes are encoded as routes and fitness is based on a general
    objective funcion (GOF)'s labels to each wavelength index supported.
    Chromosome creation and mutation procedures are based on depth-first search
    (DFS) operation, crossover is based on the one-point strategy, and
    selection takes place under a k=3-size tournament rule.
    Attributes:
        _population_size: number of individuals that comprise a population
        _num_generations: number of generations a population has to evolve
        _crossover_rate: percentage of individuals to undergo crossover
        _mutation_rate: percentage of individuals to undergo mutation
        _best_fits: collection of best fitness values across generations
    """

    def __init__(self, pop_size: int, num_gen: int,
                 cross_rate: float, mut_rate: float) -> None:
        """Constructor
        Args:
            pop_size: size of the population
            num_gen: number of evolution generations
            cross_rate: crossover rate
            mut_rate: mutation rate
        """
        self._population_size: int = pop_size
        self._num_generations: int = num_gen
        self._crossover_rate: float = cross_rate
        self._mutation_rate: float = mut_rate
        self._best_fits: List[int] = []

    @property
    def bestfit(self) -> List[int]:
        """A lisf of the best fitness values across all generations"""
        return self._best_fits

    # FIXME this ain't seem right
    @bestfit.setter
    def bestfit(self, value: int) -> None:
        self._best_fits.append(value)

    def run(self, net: Network, k: int) -> Tuple[List[int], Union[int, None]]:

        """Run the main genetic algorithm's evolution pipeline
        Args:
            net: Network instance object
            k: number of alternative paths (ignored)
        Returns:
            :obj: `tuple`: route as a list of router indices and wavelength
                index upon RWA success
        """

        # generates initial population with random but valid chromosomes
        population = Population()
        trial = 0

        # logger.debug('Creating population')
        while len(population) < self._population_size and trial < 300:  # FIXME
            allels = set(range(net.nnodes))  # router indices

            chromosome = population.make_chromosome(net.a, net.s, net.d,
                                                    allels, net.nnodes)
            if chromosome is not None:
                population.add_chromosome(chromosome)
                trial = 0
            else:
                trial += 1

        # nets = [net] * len(population.individuals)

        if (rank == 0):
            if (size == 2):
                comm.send(population.individuals, dest=1)

            else:
                splited = [population.individuals[i::(size - 1)] for i in range((size - 1))]

                for x in range(size):
                    if (x == size - 1):
                        break
                    comm.send(splited[x], dest=(x + 1))

        for generation in range(self._num_generations + 1):

            if (generation == 0):
                x = 0
            else:

                if (rank == 0):

                    if (size == 2):
                        comm.send(population.individuals, dest=1)
                    else:
                        splited = [population.individuals[i::(size - 1)] for i in range((size - 1))]

                        for x in range(size):
                            if (x == size - 1):
                                break
                            comm.send(splited[x], dest=(x + 1))

            if (rank != 0):

                data = comm.recv(source=0)
                test = []
                for chromosome in data:
                    chromosome.fit = evaluate(net, chromosome)
                    test.append(chromosome)
                comm.send(test, dest=0)

            pop_fit = Population()

            if (rank == 0):
                if (size == 2):
                    for x in comm.recv(source=1):
                        pop_fit.add_chromosome(x)
                else:
                    for x in range(size):
                        if (x == size - 1):
                            break
                        for y in comm.recv(source=(x + 1)):
                            pop_fit.add_chromosome(y)

                self.bestfit = pop_fit.sort()

                if generation == self._num_generations:
                    break

                if (size == 2):
                    comm.send(pop_fit.individuals, dest=1)
                else:
                    splited2 = [pop_fit.individuals[i::(size - 1)] for i in range((size - 1))]

                    for x in range(size):
                        if (x == size - 1):
                            break
                        comm.send(splited2[x], dest=(x + 1))

                pop_fit2 = Population()

                if (size == 2):
                    test1 = comm.recv(source=1)
                    for x in test1.individuals:
                        pop_fit2.add_chromosome(x)
                else:
                    tests = []
                    for x in range(size):
                        if (x == size - 1):
                            break
                        tests.append(comm.recv(source=(x + 1)))

                    for x in range(len(tests)):
                        if (x == size - 1):
                            break
                        for y in tests[x].individuals:
                            pop_fit2.add_chromosome(y)

                offspring = cross(pop_fit2, self._population_size,
                                  self._crossover_rate)

                pop_fit = mutate(offspring, self._population_size,
                                 self._mutation_rate, net)
            else:
                if generation == self._num_generations:
                    break
                if (rank != 0):
                    data = Population()
                    for x in comm.recv(source=0):
                        data.add_chromosome(x)

                    mating = select(data, len(data.individuals))

                    comm.send(mating, dest=0)

        if (rank == 0):
            route = pop_fit.best.genes
            try:
                wavelength = pop_fit.best.fit.labels.tolist().index(1)
            except ValueError:
                wavelength = None
            return route, wavelength
        else:

            route = None
            wavelength = None
            return route, wavelength
