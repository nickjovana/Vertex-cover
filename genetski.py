import numpy as np
from random import randint, uniform, choice, shuffle
import matplotlib.pyplot as plt
from networkx.algorithms import approximation
import networkx as nx
import time

#parametri
populacija =30
mutacija = 0.03
iteracije = 5
elitnaPopulacija = 8

#generisanje grafa
start_time = time.time()
G = nx.connected_watts_strogatz_graph(50,20,0.5,seed=1)
print(len(G.nodes), len(G.edges))


#odabir sledece generacije
def odabir(odabir, tezine):
    normalizovanaTezine= np.array([tezina for tezina in tezine]) / np.sum(tezine)
    raspodela = uniform(0, 1)
    total = 1
    for index, normalizovanaTezina in enumerate(normalizovanaTezine):
        total -= normalizovanaTezina
        if total < raspodela:
            return odabir[index]

#klasa grafa sa svim parametrima
class PokrivacGrafa:
#inicijalizacija
    def __init__(self, pocetnaPopulacija=None):
        self.pocetnaPopulacija = pocetnaPopulacija
        self.hromozomi = [randint(0, 1) for _ in range(len(self.pocetnaPopulacija.graph.nodes()))]
        self.nizCvorova = np.array([False for _ in range(len(self.pocetnaPopulacija.graph.nodes()))])
        self.brojHromozoma = 0
        self.listaCvorova = np.array([])
        self.index = -1
        self.fitness = 0.0
        self.diverzitet= 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1
        self.evoluiranFitness = False
#izracunavanje fitenes funckije
    def fitnesFunkcija(self):
        if not self.evoluiranFitness:
            graf = self.pocetnaPopulacija.graph.copy()

            self.nizCvorova = np.array([False for _ in range(len(graf.nodes()))])
            self.brojHromozoma = 0

            while len(graf.edges) > 0:
               
                cvorovi = list(graf.nodes)
                shuffle(cvorovi)

                cvorStepenaJedanPronadjena = False
                for vertex in cvorovi:
                    if graf.degree[vertex] == 1:
                        cvorStepenaJedanPronadjena = True

                        neighbors = list(graf.neighbors(vertex))
                        adjvertex = neighbors[0]
                        self.nizCvorova[adjvertex] = True
                        removed_subgraph = neighbors
                        removed_subgraph.append(vertex)
                        graf.remove_nodes_from(removed_subgraph)
                        break

                if not cvorStepenaJedanPronadjena:
                    vertex = choice(list(graf.nodes))
                    if graf.degree[vertex] >= 2:
                        
                        if self.hromozomi[self.brojHromozoma] == 0:
                           
                            for othervertex in graf.neighbors(vertex):
                                self.nizCvorova[othervertex] = True

                            removed_subgraph = list(graf.neighbors(vertex))
                            removed_subgraph.append(vertex)
                            graf.remove_nodes_from(removed_subgraph)

                        elif self.hromozomi[self.brojHromozoma] == 1:
                          
                            self.nizCvorova[vertex] = True
                            graf.remove_node(vertex)
                        self.brojHromozoma = self.brojHromozoma + 1
                        continue

            self.listaCvorova = np.where(self.nizCvorova == True)[0]
            self.fitness = len(self.pocetnaPopulacija.graph.nodes()) / (1 + len(self.listaCvorova))
            self.evoluiranFitness = True

        return self.fitness
#mutacija
    def mutate(self):
        if self.brojHromozoma > 0:
            index = randint(0, self.brojHromozoma)
        else:
            index = 0

        if self.hromozomi[index] == 0:
            self.hromozomi[index] = 1
        elif self.hromozomi[index] == 1:
            self.hromozomi[index] = 0

        self.evaluated_fitness = False
        self.fitnesFunkcija()

    def __len__(self):
        return len(self.listaCvorova)

    def __iter__(self):
        return iter(self.listaCvorova)


#klasa sa generisanje pokrivaca
class Population:
    def __init__(self, G, velicina):
        self.pokrivac = []
        self.velicina = velicina
        self.graph = G.copy()

        for pokrivacIndeks in range(self.velicina):
            pokrivacGrana = PokrivacGrafa(self)
            pokrivacGrana.fitnesFunkcija()

            self.pokrivac.append(pokrivacGrana)
            self.pokrivac[pokrivacIndeks].index = pokrivacIndeks

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.srednjiFitness = 0
       
        self.velicina_pokrivaca = 0
        self.average_vertices = np.zeros((len(self.graph.nodes()), 1))

#rangiranje prema vrednosti fitness funkcije
    def fitnessRank(self):
        if not self.evaluated_fitness_ranks:
            for pokrivacGrana in self.pokrivac:
                pokrivacGrana.fitness = pokrivacGrana.fitnesFunkcija()
                self.srednjiFitness += pokrivacGrana.fitness
                self.velicina_pokrivaca += len(pokrivacGrana)

            self.srednjiFitness /= self.velicina
            self.velicina_pokrivaca /= self.velicina
            self.pokrivac.sort(key=lambda pokrivacGrana: pokrivacGrana.fitness, reverse=True)

            for rank_number in range(self.velicina):
                self.pokrivac[rank_number].fitness_rank = rank_number
            self.evaluated_fitness_ranks = True
#raznolikost generacije
    def evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            for pokrivacGrana in self.pokrivac:
                self.average_vertices[pokrivacGrana.listaCvorova] += 1

            self.average_vertices /= self.velicina

            for pokrivacGrana in self.pokrivac:
                pokrivacGrana.diverzitet= np.sum(np.abs(pokrivacGrana.listaCvorova - self.average_vertices))/self.velicina
                
            self.pokrivac.sort(key=lambda pokrivacGrana: pokrivacGrana.diverzitet, reverse=True)

            for rank_number in range(self.velicina):
                self.pokrivac[rank_number].diverzitetRang = rank_number

            self.evaluated_diversity_ranks = True

#stvaranje nove populacije  
    def breed(self):
    
        self.pokrivac.sort(key=lambda pokrivacGrana: pokrivacGrana.fitness_rank)
        novaPopulacija = []
        for index in range(elitnaPopulacija):
            novaPopulacija.append(self.pokrivac[index])
        tezina = [1 / (1 + pokrivacGrana.fitness_rank + pokrivacGrana.diverzitetRang) for pokrivacGrana in self.pokrivac]

       
        while len(novaPopulacija) < populacija:
            roditelj1 = odabir(list(range(populacija)), tezina)
            roditelj2 = odabir(list(range(populacija)), tezina)
            while roditelj1 == roditelj2:
                roditelj1 = odabir(list(range(populacija)), tezina)
                roditelj2 = odabir(list(range(populacija)), tezina)

            dete1, dete2 = crossover(self.pokrivac[roditelj1], self.pokrivac[roditelj2])
            novaPopulacija.append(dete1)
            novaPopulacija.append(dete2)

        self.pokrivac = novaPopulacija
        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
#mutacija
    def mutate(self):
        for pokrivacGrana in self.pokrivac:
            test_probability = uniform(0, 1)
            if test_probability < mutacija:
                pokrivacGrana.mutate()
                pokrivacGrana.fitnesFunkcija()

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False
#ukrstanje
def crossover(roditelj1, roditelj2):
    if roditelj1.pocetnaPopulacija != roditelj2.pocetnaPopulacija:
        raise ValueError("Vertex covers belong to different populations.")
    dete1 = PokrivacGrafa(roditelj1.pocetnaPopulacija)
    dete2 = PokrivacGrafa(roditelj2.pocetnaPopulacija)
    tackaPodele = randint(0, min(roditelj1.brojHromozoma, roditelj2.brojHromozoma))
    dete1.hromozomi = roditelj1.hromozomi[:tackaPodele] + roditelj2.hromozomi[tackaPodele:]
    dete2.hromozomi = roditelj2.hromozomi[:tackaPodele] + roditelj1.hromozomi[tackaPodele:]
    dete1.fitnesFunkcija()
    dete2.fitnesFunkcija()
    return dete1, dete2

#main funkcija
populacijaJedinki = Population(G, populacija)
populacijaJedinki.fitnessRank()
populacijaJedinki.evaluate_diversity_ranks()
plot_fitness = [populacijaJedinki.srednjiFitness]
print("Inicijalna populacija")
print("Srednji fitness =", populacijaJedinki.srednjiFitness)
print("Srednja velicina pokrivaca =", populacijaJedinki.velicina_pokrivaca)
print()
for iteration in range(1, iteracije + 1):
    populacijaJedinki.breed()
    populacijaJedinki.mutate()
    populacijaJedinki.fitnessRank()
    populacijaJedinki.evaluate_diversity_ranks()
    plot_fitness.append(populacijaJedinki.srednjiFitness)
    print("Iteracija", iteration)
    print("Srednji fitness =", populacijaJedinki.srednjiFitness)
    print("Srednja velicina pokrivaca =", populacijaJedinki.velicina_pokrivaca)
    print(" ")
  
najboljiPokrivac = None
najboljiFitness = 0
for pokrivacGrana in populacijaJedinki.pokrivac:
    if pokrivacGrana.fitness > najboljiFitness:
        najboljiPokrivac = pokrivacGrana

print("Vreme izvrsavanja:%s sekundi ---" % (time.time() - start_time))
print("Velicina pokrivaca = ", len(najboljiPokrivac))
print("Najbolji pokrivac = ", najboljiPokrivac.listaCvorova)
print("Broj cvorova = ", len(najboljiPokrivac.listaCvorova))
print("Broj grana = ", len(approximation.min_weighted_vertex_cover(G)))
plt.title("statistika")
plt.plot(range(iteracije + 1), plot_fitness, 'b--',)
plt.ylabel('fitnes')
plt.xlabel("broj iteracija")
plt.show()
