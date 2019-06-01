import random
import math

POP_SIZE = 200
GEN_MAX = 100
N_VARIABLES = 2
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.25  
PRECISION = 10**2
A_VALUE = 10

def fitness(target):
  return 1/(1+target)
  # return target

def funcObjective(x):
  value = A_VALUE * N_VARIABLES
  for i in range(0, N_VARIABLES):
    value += (x[i]**2 - A_VALUE * math.cos(2*math.pi*x[i]))
  return value

def randomNumber():
  return (random.randint(-512, 512)/PRECISION)
  # return random.uniform(-5.12, 5.12)

def probability(target, total):
  return target / total

def spawn_starting_population(amount):
  return [spawn_individual() for x in range (0,amount)]

def spawn_individual():
  return [randomNumber() for x in range (0,N_VARIABLES)]

def calcNewPopulation(oldPopulation, cumulativeProbabilities):
  newPopulation = []
  for i in range(0,len(cumulativeProbabilities)):
    r = random.random()
    for j in range(0,len(cumulativeProbabilities)):
      if(r < cumulativeProbabilities[j]):
        newPopulation.append(oldPopulation[j])
        break
      if(r > cumulativeProbabilities[j] and r < cumulativeProbabilities[j+1]):
        newPopulation.append(oldPopulation[j+1]) 
        break
  return newPopulation

def crossover(oldPopulation):
  if(N_VARIABLES > 1):
    parents = []
    newPopulation = oldPopulation
    for i in range(0,len(oldPopulation)):
      r = random.random()
      if(r < CROSSOVER_RATE):
        parents.append(i)

    for index, i in enumerate(parents):
      crossoverPoint = random.randint(1, N_VARIABLES-1)
      if(index == len(parents) - 1):
        j = parents[0]
      else:
        j = parents[index + 1]
      newPopulation[i] = oldPopulation[i][:crossoverPoint] + oldPopulation[j][crossoverPoint:]
    return newPopulation
  return oldPopulation

def mutate(population):
  totalGen = N_VARIABLES * POP_SIZE
  numberOfMutations = round(totalGen * MUTATION_RATE)
  for i in range(0, numberOfMutations):
    randomIndex = random.randint(0, totalGen-1)
    row = math.floor(randomIndex / N_VARIABLES)
    column = randomIndex % N_VARIABLES
    randomValue = randomNumber()
    population[row][column] = randomValue
  return population

def main():
  population = spawn_starting_population(POP_SIZE)
  for x in population:
    print("Chromosome: %s. Generation:  %i. Function value: %s" % (x, 0, str(funcObjective(x))))
  print('----')
  for gen in range(0, GEN_MAX):
    fValues = [funcObjective(x) for x in population]
    fitnesses = [fitness(x) for x in fValues]
    totalFitnessValue = 0
    for x in fitnesses:
      totalFitnessValue += x
    probabilities = [probability(x, totalFitnessValue) for x in fitnesses]
    cumulativeProbabilities = []
    for index, x in enumerate(probabilities):
      if(index == 0):
        cumulativeProbabilities.append(x)
      else:
        cumulativeProbabilities.append(x + cumulativeProbabilities[index-1])

    population = calcNewPopulation(population, cumulativeProbabilities)
    population = crossover(population)
    population = mutate(population)

  population = sorted(population, key=lambda x: funcObjective(x), reverse=False)
  for x in population:
    print("Chromosome: %s. Generation:  %i. Function value: %s" % (x, gen, str(funcObjective(x))))

  print('Answer: ')
  print("Chromosome: %s. Generation:  %i. Function value: %s" % (population[0], gen, str(funcObjective(population[0]))))

if __name__ == "__main__":
  main()