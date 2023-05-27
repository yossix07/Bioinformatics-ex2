import random
import sys
from matplotlib import pyplot as plt

# Global variables
POPULATION_SIZE = 100
MAX_GENERATIONS = 90
MUTATION_RATE = 1
REPLACEMENT_RATE = 0.15
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001
NO_IMPROVEMENT_THRESHOLD = 12
OPTIMAL = False
LOCAL_SWAPS = 8
OPTIMAL_THRESHOLD = 0.98
DISPLAY_AVG_GRAPH = False
DISPLAY_BEST_GRAPH = False

fitness_calls_coutner = 0

with open("dict.txt", "r") as file:
    word_list = file.read().splitlines()

with open("Letter_Freq.txt", "r") as file:
    letter_freq = file.read().splitlines()

with open("Letter2_Freq.txt", "r") as file:
    letter_pair_freq = file.read().splitlines()

# return the text from a file
def read_text_from_file(file_name):
    file = open(file_name, "r")
    text = file.read()
    file.close()
    return text.lower()

# return a mapping between letters and their frequency from a file
def load_frequency_map(data):
    freq_map = {}
    for line in data:
        freq, letter = line.split("\t")
        if freq != '':
            freq_map[letter.lower()] = float(freq)
    return freq_map

# Load frequency maps
letter_freq_map = load_frequency_map(letter_freq)
letter_pair_freq_map = load_frequency_map(letter_pair_freq)

# return a mapping between letters and their frequency from a text
def calculate_letter_frequency_from_text(text):
    freq_map = {}
    for letter in text:
        if letter in freq_map:
            freq_map[letter] += 1
        else:
            freq_map[letter] = 1

    text_len = len(text)
    for letter in freq_map:
        freq_map[letter] /= text_len
    return freq_map

# return a mapping between letter pairs and their frequency from a text
def calculate_letters_pairs_frequency_from_text(text):
    freq_map = {}
    for i in range(len(text)-1):
        letter_pair = text[i:i+2]
        if letter_pair in freq_map:
            freq_map[letter_pair] += 1
        else:
            freq_map[letter_pair] = 1
    text_len = len(text)
    for letter_pair in freq_map:
        freq_map[letter_pair] /= text_len

    return freq_map

# return the fitness of a decryption key for a given ciphertext
def calculate_fitness(decryption_key, ciphertext):
    global fitness_calls_coutner
    global OPTIMAL
    OPTIMAL = False
    fitness_calls_coutner += 1
    decrypted_text = decrypt_text(ciphertext, decryption_key).replace(".", "").replace(",", "").replace(";", "")

    decrypted_text_set = set(decrypted_text.split(" "))
    letter_freq = calculate_letter_frequency_from_text(decrypted_text)
    letter_pair_freq = calculate_letters_pairs_frequency_from_text(decrypted_text)
    words_matching = 0
    letters_matching = 0

    actual_word_list = [element for element in word_list if element != ""]
    num_of_word_in_text = len(decrypted_text_set)

    for word in actual_word_list:
        if word in decrypted_text_set:
            words_matching += 1

    for letter in letter_freq_map:
        if letter in letter_freq and letter in letter_freq_map:
            letters_matching += (1 - abs(letter_freq[letter] - letter_freq_map[letter])) ** 2

    for letter_pair in letter_pair_freq_map:
        if letter_pair in letter_pair_freq and letter_pair in letter_pair_freq_map:
            letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter_pair_freq_map[letter_pair])) ** 2.5

    if words_matching >= OPTIMAL_THRESHOLD * num_of_word_in_text:
        OPTIMAL = True

    return words_matching + letters_matching

# return the decrypted text for a given ciphertext and decryption key
def decrypt_text(ciphertext, decryption_key):
    decrypted_text = ""
    for letter in ciphertext:
        if letter in decryption_key:
            decrypted_text += decryption_key[letter]
        else:
            decrypted_text += letter
    return decrypted_text

# return a random decryption key
def generate_random_key():
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shuffled_alphabet = list(alphabet)
    random.shuffle(shuffled_alphabet)
    decryption_key = {}
    for i, letter in enumerate(alphabet):
        decryption_key[letter] = shuffled_alphabet[i]
    return decryption_key

# return a crossover between two decryption keys
def crossover(parent1, parent2):
    keys = list(parent1.keys())
    crossover_point = random.randint(0, len(keys)-1)
    child = {}
    
    # Copy the letters from parent1 up to the crossover point
    for i in range(crossover_point + 1):
        child[keys[i]] = parent1[keys[i]]

    # Fill the remaining letters from parent2, ensuring uniqueness
    for i in range(crossover_point + 1, len(keys)):
        letter = parent2[keys[i]]

        # Check if the letter is already present in the child
        while letter in child.values():

            # Find a letter that doesn't appear in the child
            available_letter = next(l for l in parent2.values() if l not in child.values())

            # Replace one of the duplicate letters with the available letter
            duplicate_letter = next(k for k, v in child.items() if v == letter)
            child[duplicate_letter] = available_letter

            letter = parent2[keys[i]]

        child[keys[i]] = letter

    return child

# return a mutated decryption key
def mutate(decryption_key):
    mutated_key = decryption_key.copy()
    keys = list(mutated_key.keys())
    index1, index2 = random.sample(range(len(keys)), 2)
    letter1, letter2 = keys[index1], keys[index2]
    mutated_key[letter1], mutated_key[letter2] = mutated_key[letter2], mutated_key[letter1]
    return mutated_key

# returns two parents from the population using tournament selection
def select_parents(population, fitness_scores, tournament_size=5):
    parents = []
    
    # Select 2 parents
    for _ in range(2):
        tournament_candidates = random.sample(range(len(population)), tournament_size)
        tournament_scores = [fitness_scores[i] for i in tournament_candidates]
        winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
        parents.append(population[winner_index])
    
    return parents

# replace the worst individuals in the population with the best individuals from the offspring and form the population
def replace_population(population, offspring, fitness_scores):
    worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:3*REPLACEMENT_SIZE]

    # Find indices of individuals with best fitness scores from the existing population
    best_population_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:REPLACEMENT_SIZE]

    # Find indices of individuals with best fitness scores from the offspring
    best_offspring_indices = sorted(range(len(offspring)), key=lambda i: calculate_fitness(offspring[i], ciphertext),
                                    reverse=True)[:REPLACEMENT_SIZE]

    # Replace worst fitness scores with best fitness scores from the existing population
    for i in range(REPLACEMENT_SIZE):
        population[worst_indices[i]] = population[best_population_indices[0]]
        population[worst_indices[i + REPLACEMENT_SIZE]] = population[best_population_indices[i]]
        population[worst_indices[i + 2 * REPLACEMENT_SIZE]] = offspring[best_offspring_indices[i]]

    return population

# make n swaps in the decryption key
def local_optimization(individual, n=LOCAL_SWAPS):
    mutated_key = individual.copy()
    for _ in range(n):
        mutated_key = mutate(mutated_key)
    return mutated_key

# make local optimization on the population and change the fitness scores when it improves
def darwin_mutation(population, fitness_scores, ciphertext):
    for i in range(len(population)):
        mutated_individual = local_optimization(population[i])
        mutated_fitness = calculate_fitness(mutated_individual, ciphertext)
        if mutated_fitness > fitness_scores[i]:
            fitness_scores[i] = mutated_fitness

# make local optimization on the population, change the decryption_key and fitness scores when it improves
def lamarck_mutation(population, fitness_scores, ciphertext):
    for i in range(len(population)):
        mutated_individual = local_optimization(population[i])
        mutated_fitness = calculate_fitness(mutated_individual, ciphertext)
        if mutated_fitness > fitness_scores[i]:
            fitness_scores[i] = mutated_fitness
            population[i] = mutated_individual

# move out of local maxima
def handle_local_max(ciphertext,best_decryption_key, best_decryption_key_fitness):
    for i in range(5):
        best_key, _, best_fitness = genetic_algorithm(ciphertext)
        if best_decryption_key_fitness < best_fitness:
            best_decryption_key_fitness = best_fitness
            best_decryption_key = best_key
        if OPTIMAL:
            return best_decryption_key
    return best_decryption_key

# generate graph with given text and scores
def generate_graph(main_text, y_text, num_generation, generation_scores ):
    plt.plot(num_generation, generation_scores)
    plt.title(main_text)
    plt.ylabel(y_text)
    plt.xlabel('Number of generations')
    plt.show()

# genetic algorithm for decrypting the ciphertext
def genetic_algorithm(ciphertext, mode=''):
    global generations_average_fitness_scores
    global generations_best_fitness_scores
    global num_of_generations
    num_of_generations = 0
    generations_best_fitness_scores = []
    generations_average_fitness_scores = []

    # Initialize random starting population
    population = [generate_random_key() for _ in range(POPULATION_SIZE)]
    no_improvement_counter = 0
    best_fitness = 0
    gen_num = 0

    for generation in range(MAX_GENERATIONS):
        fitness_scores = []
        offspring = []

        # Calculate fitness for each individual
        for decryption_key in population:
            fitness = calculate_fitness(decryption_key, ciphertext)
            fitness_scores.append(fitness)
            if OPTIMAL:
                gen_num = generation
                num_of_generations = [i for i in range(gen_num)]
                return decryption_key, 0, fitness

        # Select parents and create offspring
        for _ in range(POPULATION_SIZE//2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            offspring.append(child)

            
        # Replace population with offspring
        population = replace_population(population, offspring, fitness_scores)

        if mode == 'darwin':
            darwin_mutation(population, fitness_scores, ciphertext)
        if mode == 'lamarck':
            lamarck_mutation(population, fitness_scores, ciphertext)

        temp_best_fitness = best_fitness
        best_index = fitness_scores.index(max(fitness_scores))
        best_key = population[best_index]
        best_fitness = fitness_scores[best_index]
        average_gen_fitness = sum(fitness_scores) / len(fitness_scores)
        generations_average_fitness_scores.append(average_gen_fitness)
        generations_best_fitness_scores.append(best_fitness)

        if best_fitness - temp_best_fitness < EPSILON:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        if no_improvement_counter == NO_IMPROVEMENT_THRESHOLD:
            break

        print(f"Generation: {generation+1} | Best Fitness: {best_fitness} | Best Decryption Key: {best_key}")

        gen_num = generation + 1
    num_of_generations = [i for i in range(gen_num)]

    return best_key, no_improvement_counter, best_fitness

args = sys.argv[1:]
encFile = args[0]
mode = ''
if len(args) > 1:
    mode = args[1].lower()

ciphertext = read_text_from_file(encFile)

# initialize lists to generate graphs
num_of_generations = []
generations_best_fitness_scores = []
generations_average_fitness_scores = []

best_decryption_key, counter, fitness = genetic_algorithm(ciphertext, mode)

# handle local max
if counter == NO_IMPROVEMENT_THRESHOLD and not OPTIMAL:
    best_decryption_key = handle_local_max(ciphertext, best_decryption_key, fitness)

# generate graphs
if DISPLAY_AVG_GRAPH:
    generate_graph('Average fitness score','Average fitness score',num_of_generations,
                   generations_average_fitness_scores)
if DISPLAY_BEST_GRAPH:
    generate_graph('Best fitness score', 'Best fitness score', num_of_generations,
                   generations_best_fitness_scores)

decrypted_text = decrypt_text(ciphertext, best_decryption_key)

print("fitness calls: ", fitness_calls_coutner)

# creating files
with open("plain.txt", "w") as plain_file:
    plain_file.write(decrypted_text)

with open("perm.txt", "w") as perm_file:
    for letter, decrypted_letter in best_decryption_key.items():
        perm_file.write(f"{letter}\t{decrypted_letter}\n")