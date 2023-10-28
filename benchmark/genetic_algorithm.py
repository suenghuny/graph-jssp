import os
import time
import random
import numpy as np
import pandas as pd


class TourManager:
    def __init__(self):
        self.destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)


class Tour:
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0.0
        self.distance = 0
        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        geneString = 'Start -> '
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + ' -> '
        geneString += 'End'
        return geneString

    def generateIndividual(self):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        random.shuffle(self.tour)

    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.fitness_function())
        return self.fitness

    def tourSize(self):
        return len(self.tour)

    def containsCity(self, city):
        return city in self.tour

    def fitness_function(self):

        # data = pd.DataFrame(
        #     columns=["plate_weld", "saw_front", "turn_over", "saw_back", "longi_attach", "longi_weld"])
        # for i in range(self.tourmanager.numberOfCities()):
        #     data.loc[i] = self.tour[i]
        data = self.tour
        block_num = len(data)
        process_num = len(data[0])

        blocks = np.array(data)
        sequence = [i for i in range(block_num)]

        temp = np.zeros((block_num + 1, process_num + 1))
        for i in range(1, block_num + 1):
            temp[i, 0] = 0
            for j in range(1, process_num + 1):
                if i == 1:
                    temp[0, j] = 0

                if temp[i - 1, j] > temp[i, j - 1]:
                    temp[i, j] = temp[i - 1, j] + blocks[sequence[i - 1], j - 1]
                else:
                    temp[i, j] = temp[i, j - 1] + blocks[sequence[i - 1], j - 1]
        C_max = temp[block_num, process_num]

        return C_max


class Population:
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)

        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
        self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    def populationSize(self):
        return len(self.tours)


class GA:
    def __init__(self, tourmanager, mutationRate=0.05, tournamentSize=5, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism

    def evolvePopulation(self, pop):
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1

        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            child = self.crossover(parent1, parent2)
            newPopulation.saveTour(i, child)

        for i in range(elitismOffset, newPopulation.populationSize()):
            self.mutate(newPopulation.getTour(i))

        return newPopulation

    def crossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        startPos = int(random.random() * parent1.tourSize())
        endPos = int(random.random() * parent1.tourSize())

        for i in range(0, child.tourSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setCity(i, parent1.getCity(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setCity(i, parent1.getCity(i))

        for i in range(0, parent2.tourSize()):
            if not child.containsCity(parent2.getCity(i)):
                for ii in range(0, child.tourSize()):
                    if child.getCity(ii) == None:
                        child.setCity(ii, parent2.getCity(i))
                        break

        return child

    def mutate(self, tour):
        for tourPos1 in range(0, tour.tourSize()):
            if random.random() < self.mutationRate:
                tourPos2 = int(tour.tourSize() * random.random())

                city1 = tour.getCity(tourPos1)
                city2 = tour.getCity(tourPos2)

                tour.setCity(tourPos2, city1)
                tour.setCity(tourPos1, city2)

    def tournamentSelection(self, pop):
        tournament = Population(self.tourmanager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            randomId = int(random.random() * pop.populationSize())
            tournament.saveTour(i, pop.getTour(randomId))
        fittest = tournament.getFittest()
        return fittest


def result():

    np_data = data.to_numpy()
    list_data = np_data.tolist()

    # Setup cities and tour
    tourmanager = TourManager()

    for i in range(len(list_data)):
        tourmanager.addCity(list_data[i])

    # Initialize population
    ga_start = time.time()
    pop = Population(tourmanager, populationSize=population_size, initialise=True)
    # print("Initial Makespan:", "#################################################################################",
    #       pop.getFittest().fitness_function())

    # Evolve population
    ga = GA(tourmanager)

    evolve_time.append(time.time()-ga_start)
    make_span.append(pop.getFittest().fitness_function())
    for i in range(n_generations):
        pop = ga.evolvePopulation(pop)
        evolve_time.append(time.time()-ga_start)
        make_span.append(pop.getFittest().fitness_function())

    # import matplotlib.pyplot as plt
    # from pylab import rcParams
    # rcParams['figure.figsize']=5,5
    # plt.plot(evolve_time,make_span)
    # plt.xlabel('Time')
    # plt.ylabel('Makespan')
    # plt.title('Genetic Algorithm')
    # plt.show()
    # plt.savefig('./GA_result')

    # Print final results
    print("Congratulation! Finished:")
    print(pop.getFittest().fitness_function())


    return pop.getFittest().fitness_function()


population_size = 10
n_generations = 500
np.random.seed(seed=233423)
random.seed(100)
filename = ["../environment/data/PBS_5_75.xlsx",
            "../environment/data/PBS_5_125.xlsx"]
            # "../environment/data/PBS_6_75.xlsx",
            # "../environment/data/PBS_6_100.xlsx",
            # "../environment/data/PBS_6_125.xlsx",
            # "../environment/data/PBS_10_50.xlsx",
            # "../environment/data/PBS_10_100.xlsx",
            # "../environment/data/PBS_10_200.xlsx",
            # "../environment/data/PBS_15_25.xlsx",
            # "../environment/data/PBS_15_50.xlsx",
            # "../environment/data/PBS_15_100.xlsx",
            # "../environment/data/PBS_15_200.xlsx",
            # "../environment/data/PBS_20_25.xlsx",
            # "../environment/data/PBS_20_50.xlsx",
            # "../environment/data/PBS_20_100.xlsx",
            # "../environment/data/PBS_20_200.xlsx"]
evolve_time=[]
make_span=[]

res_list = []
time_list = []
for j in filename:
    start = time.time()
    total_result = []
    for i in range(30):
        data = pd.read_excel(j, sheet_name=i, engine="openpyxl")
        total_result.append(result())
    finish = time.time()

    np.array(total_result)
    print('Population = ',population_size)
    print('Generation = ',n_generations)
    print(j, "파일의 결과는 : ", total_result)
    print(j, "파일의평균은 : ", np.mean(total_result))
    print("time : ",(finish-start)/30)
    res_list.append(np.mean(total_result))
    time_list.append((finish-start)/30)

df = pd.DataFrame({"RES": res_list, "TIME": time_list})
df.to_excel("ga.xlsx")
# total_result=[]
# for i in range(10):
#     data = pd.read_excel(filename, sheet_name=i)
#     total_result.append(result())
# print(filename, "파일의 결과는 : ", total_result)
# print(filename, "파일의평균은 : ", sum(total_result)/len(total_result))
# print("time : ", time.time() - start)
