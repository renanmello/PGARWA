import mpi4py.MPI
import numpy as np
import glob
import os
import rwa as rw
from timeit import default_timer
import nsf
from datetime import datetime
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def simulator() -> None:
    """Main RWA simulation routine over WDM networks
    The loop levels of the simulator iterate over the number of repetitions,
    (simulations), the number of Erlangs (load), and the number of connection
    requests (calls) to be either allocated on the network or blocked if no
    resources happen to be available.
    Args:
        args: set of arguments provided via CLI to argparse module
    """

    # simulation params
    calls = 150  # number of calls to arrive
    loads = 30  # network load
    num_sim = 20  # number of simulations
    waves = 4  # wavelenghts(4 or 8)
    # print header for pretty stdout console logging

    if (rank == 0):
        print('Load:   ', end='')
        for i in range(1, loads + 1):
            print('%4d' % i, end=' ')
        print()

    time_per_simulation = []

    for simulation in range(num_sim):
        sim_time = default_timer()
        net = nsf.NationalScienceFoundation(waves)
        rwa = rw.genetic_algorithm(120, 30, 0.6,
                                   0.02)  # Ag Params(total population, generations, cross rate, mutation rate)

        blocklist = []
        blocks_per_erlang = []

        # ascending loop through Erlangs
        for load in range(1, loads + 1):
            blocks = 0
            for call in range(calls):
                if (rank == 0):
                    print('\rBlocks: ', end='', flush=True)

                for b in blocklist:
                    if (rank == 0):
                        print('%04d ' % b, end='', flush=True)
                if (rank == 0):
                    print(' %04d' % call, end='')

                # Poisson arrival is modelled as an exponential distribution
                # of times, according to Pawełczak's MATLAB package [1]:
                # @until_next: time until the next call arrives
                # @holding_time: time an allocated call occupies net resources
                until_next = -np.log(1 - np.random.rand()) / load
                holding_time = -np.log(1 - np.random.rand())

                # Call RWA algorithm, which returns a lightpath if successful
                # or None if no λ can be found available at the route's first
                # link

                if (rank == 0):
                    if (size == 2):
                        comm.isend(net, dest=1)
                    else:
                        for x in range(size):
                            if (x == (size - 1)):
                                break
                            comm.send(net, dest=(x + 1))

                    lightpath = rwa(net, waves)
                elif rank != 0:

                    net2 = comm.recv(source=0)
                    lightpath = rwa(net2, waves)

                # If lightpath is non None, the first link between the source
                # node and one of its neighbours has a wavelength available,
                # and the RWA algorithm running at that node thinks it can
                # allocate on that λ. However, we still need to check whether
                # that same wavelength is available on the remaining links
                # along the path in order to reach the destination node. In
                # other words, despite the RWA was successful at the first
                # node, the connection can still be blocked on further links
                # in the future hops to come, nevertheless.

                if (rank == 0):
                    if lightpath is not None:
                        # check if the color chosen at the first link is available
                        # on all remaining links of the route
                        for (i, j) in lightpath.links:
                            if not net.n[i][j][lightpath.w]:
                                lightpath = None
                                break

                    # Check if λ was not available either at the first link from
                    # the source or at any other further link along the route.
                    # Otherwise, allocate resources on the network for the
                    # lightpath.
                    if lightpath is None:
                        blocks += 1
                    else:
                        lightpath.holding_time = holding_time
                        net.t.add_lightpath(lightpath)
                        for (i, j) in lightpath.links:
                            net.n[i][j][lightpath.w] = 0  # lock channel
                            net.t[i][j][lightpath.w] = holding_time

                            # make it symmetric
                            net.n[j][i][lightpath.w] = net.n[i][j][lightpath.w]
                            net.t[j][i][lightpath.w] = net.t[i][j][lightpath.w]

                    # FIXME The following two routines below are part of the same
                    # one: decreasing the time network resources remain allocated
                    # to connections, and removing finished connections from the
                    # traffic matrix. This, however, should be a single routine
                    # iterating over lightpaths links instead of all edges, so when
                    # the time is up on all links of a lightpath, the lightpath
                    # might be popped from the matrix's list. I guess the problem
                    # is the random initialisation of the traffic matrix's holding
                    # times during network object instantiation, but if this is
                    # indeed the fact it needs some consistent testing.
                    for lightpath in net.t.lightpaths:
                        if lightpath.holding_time > until_next:
                            lightpath.holding_time -= until_next
                        else:
                            # time's up: remove conn from traffic matrix's list
                            net.t.remove_lightpath_by_id(lightpath.id)

                    # Update *all* channels that are still in use
                    for (i, j) in net.get_edges():
                        for w in range(net.nchannels):
                            if net.t[i][j][w] > until_next:
                                net.t[i][j][w] -= until_next
                            else:
                                # time's up: free channel
                                net.t[i][j][w] = 0
                                if not net.n[i][j][w]:
                                    net.n[i][j][w] = 1  # free channel

                            # make matrices symmetric
                            net.t[j][i][w] = net.t[i][j][w]
                            net.n[j][i][w] = net.n[j][i][w]

            if (rank == 0):
                blocklist.append(blocks)
                blocks_per_erlang.append(100.0 * blocks / calls)

        if (rank == 0):
            sim_time = default_timer() - sim_time
            time_per_simulation.append(sim_time)

        if (rank == 0):
            print('\rBlocks: ', end='', flush=True)
            for b in blocklist:
                print('%04d ' % b, end='', flush=True)
            print('\n%-7s ' % 'BP (%):', end='')
            print(' '.join(['%4.1f' % b for b in blocks_per_erlang]), end=' ')
            print('[sim %d: %.2f secs]' % (simulation + 1, sim_time))

            with open('simulacoes.txt', 'a') as f:
                for bp in blocks_per_erlang:
                    f.write(' %7.3f' % bp)
                f.write('\n')

            with open('time.txt', 'a') as f:
                for bp in time_per_simulation:
                    f.write(' %7.3f' % bp)
                f.write('\n')
        else:
            print("fim do rank: ", rank)


# function to plot the results from a .txt file
def plot():
    filelist = []

    for f in glob.glob('simulacoes4.txt'):
        filelist.append(os.path.basename(f))
        data = np.loadtxt(f)
        # print("data", data)
        if data.ndim == 1:
            max_load = data.shape[0] + 1
            plt.plot(np.arange(1, max_load), data, '--')
        else:
            max_load = data.shape[1] + 1
            plt.plot(np.arange(1, max_load), data.mean(axis=0), '--')
        plt.xlim(0, max_load)  # load in erlangs
        plt.ylim(0, 100)  # blocking probs in %
        plt.yticks(np.arange(0, 101, 10))

    plt.grid()
    plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=18)
    plt.xlabel('Carga (Erlangs)', fontsize=18)
    plt.title('Média das Probabilidades de Bloqueio', fontsize=20)
    # plt.legend(filelist)
    plt.show(block=True)


def plots():
    filelist = []
    filelist2 = []
    filelist3 = []
    filelist4 = []

    for f in glob.glob('simulacoes1.txt'):
        filelist.append(os.path.basename(f))
        data = np.loadtxt(f)
        # print("data", data)
        if data.ndim == 1:
            max_load = data.shape[0] + 1
            plt.plot(np.arange(1, max_load), data, '--')
        else:
            max_load = data.shape[1] + 1
            plt.plot(np.arange(1, max_load), data.mean(axis=0), '--', color='b', label='sequencial')
            plt.legend()
        plt.xlim(0, max_load)  # load in erlangs
        plt.ylim(0, 100)  # blocking probs in %
        plt.yticks(np.arange(0, 101, 10))

    for f in glob.glob('simulacoes2.txt'):
        filelist2.append(os.path.basename(f))
        data = np.loadtxt(f)
        # print("data", data)
        if data.ndim == 1:
            max_load = data.shape[0] + 1
            plt.plot(np.arange(1, max_load), data, '--')
        else:
            max_load = data.shape[1] + 1
            plt.plot(np.arange(1, max_load), data.mean(axis=0), '--', color='r', label='2 núcleos')
            plt.legend()
        plt.xlim(0, max_load)  # load in erlangs
        plt.ylim(0, 100)  # blocking probs in %
        plt.yticks(np.arange(0, 101, 10))

    for f in glob.glob('simulacoes3.txt'):
        filelist3.append(os.path.basename(f))
        data = np.loadtxt(f)
        # print("data", data)
        if data.ndim == 1:
            max_load = data.shape[0] + 1
            plt.plot(np.arange(1, max_load), data, '--')
        else:
            max_load = data.shape[1] + 1
            plt.plot(np.arange(1, max_load), data.mean(axis=0), '--', color='g', label='3 núcleos')
            plt.legend()
        plt.xlim(0, max_load)  # load in erlangs
        plt.ylim(0, 100)  # blocking probs in %
        plt.yticks(np.arange(0, 101, 10))

    for f in glob.glob('simulacoes4.txt'):
        filelist4.append(os.path.basename(f))
        data = np.loadtxt(f)
        # print("data", data)
        if data.ndim == 1:
            max_load = data.shape[0] + 1
            plt.plot(np.arange(1, max_load), data, '--')
        else:
            max_load = data.shape[1] + 1
            plt.plot(np.arange(1, max_load), data.mean(axis=0), '--', color='y', label='4 núcleos')
            plt.legend()
        plt.xlim(0, max_load)  # load in erlangs
        plt.ylim(0, 100)  # blocking probs in %
        plt.yticks(np.arange(0, 101, 10))

    plt.grid()
    plt.ylabel('Probabilidade de Bloqueio (%)', fontsize=18)
    plt.xlabel('Carga (Erlangs)', fontsize=18)
    plt.title('Média das Probabilidades de Bloqueio', fontsize=20)
    # plt.legend(filelist)
    plt.show(block=True)


if __name__ == '__main__':

    inicio = datetime.now()
    print("Inicio do rank: ", rank)
    simulator()
    print("fim do rank: ", rank)
    fim = datetime.now() - inicio
    if (rank == 0):
        # plot()
        print("Time spend: ", fim)
        print("fim do rank: ", rank)

    # print("sim time: ", datetime.now() - start)
    # plot()
