import sys

from mpi4py import MPI
from pop import Population
from net import Network
import nsf
from env import evaluate, select

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

population = Population()
net = nsf.NationalScienceFoundation(4)
trial = 0
while len(population) < 100 and trial < 300:  # FIXME
    allels = set(range(net.nnodes))  # router indices

    chromosome = population.make_chromosome(net.a, net.s, net.d,
                                            allels, net.nnodes)
    if chromosome is not None:
        population.add_chromosome(chromosome)
        trial = 0
    else:
        trial += 1

splited = [population.individuals[i::3] for i in range(3)]
list_size_1 = len(splited[0])
list_size_2 = len(splited[1])
list_size_3 = len(splited[2])

if (rank == 0):
    comm.send(splited[0], dest=1)
    comm.send(splited[1], dest=2)
    comm.send(splited[2], dest=3)

if (rank == 1):
    data1 = comm.recv(source=0)
    test = []
    for chromosome in data1:
        chromosome.fit = evaluate(net, chromosome)
        test.append(chromosome)
    comm.send(test, dest=0)

if (rank == 2):
    data2 = comm.recv(source=0)
    test = []
    for chromosome in data2:
        chromosome.fit = evaluate(net, chromosome)
        test.append(chromosome)
    comm.send(test, dest=0)

if (rank == 3):
    data3 = comm.recv(source=0)
    test = []
    for chromosome in data3:
        chromosome.fit = evaluate(net, chromosome)
        test.append(chromosome)
    comm.send(test, dest=0)

pop_fit = Population()

if(rank==0):
    for x in comm.recv(source=1):
        pop_fit.add_chromosome(x)
    for x in comm.recv(source=2):
        pop_fit.add_chromosome(x)
    for x in comm.recv(source=3):
        pop_fit.add_chromosome(x)

else:
    MPI.Is_finalized()


print(pop_fit.individuals)


"""""
if(rank == 1):
    pop_temp = Population()
    for chromosome in population.individuals[0:list_size_1-1]:
        chromosome.fit = evaluate(net,chromosome)
        comm.send(chromosome, dest=0)

if(rank == 2):
    pop_temp = Population()
    for chromosome in population.individuals[0:list_size_1 - 1]:
        chromosome.fit = evaluate(net, chromosome)
        pop_temp.add_chromosome(chromosome)
        comm.send(chromosome, dest=0)

if(rank == 3):
    pop_temp = Population()
    for chromosome in population.individuals[0:list_size_1 - 1]:
        pop_temp.add_chromosome(chromosome)
        comm.send(chromosome, dest=0)

print(comm.gather())
"""""