from mpi4py import futures
from datetime import datetime
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank==0):
    for x in range(size):
        print(x)


else:
    for x in range(size):
        print(x + 10)


'''''
inicio = datetime.now()

lista1  = []
lista2 = []
#sequencial 15 segundos

def calc():
    for j in range(500):
        for x in range(100000):
            lista1.append(2 ^ x)

    #print(lista1)

    #for z in range(100):
    #    for y in range(100000):
    #        lista2.append(2 ^ y)
    return lista1[100000]
    #print(lista2)

#for x in range(2):
#    print(calc())
with futures.ThreadPoolExecutor() as executor:
    tasks = [executor.submit(calc) for x in range(2)]
    for x in futures.as_completed(tasks):
        print(x.result())



print(datetime.now() - inicio)

'''''