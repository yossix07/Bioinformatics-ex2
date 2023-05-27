import random

# Global variables
from matplotlib import pyplot as plt

POPULATION_SIZE = 100
MAX_GENERATIONS = 90
MUTATION_RATE = 1
REPLACEMENT_RATE = 0.15
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001
NO_IMPROVEMENT_THRESHOLD = 12
OPTIMAL = False
DARWIN = False
LAMARCK = False
LOCAL_SWAPS = 5
OPTIMAL_THRESHOLD = 0.98

fitness_calls_coutner = 0

with open("dict.txt", "r") as file:
    word_list = file.read().splitlines()

with open("Letter_Freq.txt", "r") as file:
    letter_freq = file.read().splitlines()

with open("Letter2_Freq.txt", "r") as file:
    letter_pair_freq = file.read().splitlines()

def read_text_from_file(file_name):
    file = open(file_name, "r")
    text = file.read()
    file.close()
    return text

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

def calculate_fitness(decryption_key, ciphertext):
    global fitness_calls_coutner
    global OPTIMAL
    OPTIMAL = False
    #optimal_solution=False
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
#    print(letters_matching)
    for letter_pair in letter_pair_freq_map:
        if letter_pair in letter_pair_freq and letter_pair in letter_pair_freq_map:
            letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter_pair_freq_map[letter_pair])) ** 2.5

    if words_matching >= OPTIMAL_THRESHOLD*num_of_word_in_text:
        OPTIMAL = True

    return words_matching + letters_matching


def decrypt_text(ciphertext, decryption_key):
    decrypted_text = ""
    for letter in ciphertext:
        if letter in decryption_key:
            decrypted_text += decryption_key[letter]
        else:
            decrypted_text += letter
    return decrypted_text


def generate_random_key():
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shuffled_alphabet = list(alphabet)
    random.shuffle(shuffled_alphabet)
    decryption_key = {}
    for i, letter in enumerate(alphabet):
        decryption_key[letter] = shuffled_alphabet[i]
    return decryption_key


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

            letter = parent2[keys[i]]  # Update the current letter with the new one

        child[keys[i]] = letter

    return child

def mutate(decryption_key):
    mutated_key = decryption_key.copy()
    keys = list(mutated_key.keys())
    index1, index2 = random.sample(range(len(keys)), 2)
    letter1, letter2 = keys[index1], keys[index2]
    mutated_key[letter1], mutated_key[letter2] = mutated_key[letter2], mutated_key[letter1]
    return mutated_key


def select_parents(population, fitness_scores, tournament_size=5):
    parents = []
    
    for _ in range(2):  # Select 2 parents
        tournament_candidates = random.sample(range(len(population)), tournament_size)
        tournament_scores = [fitness_scores[i] for i in tournament_candidates]
        winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
        parents.append(population[winner_index])
    
    return parents


def replace_population(population, offspring, fitness_scores):
    # # Find index of individual with worst fitness score
    # worst_fitness = min(fitness_scores)
    # worst_index = fitness_scores.index(worst_fitness)
    #
    # # Replace individual with worst fitness score with best offspring
    # best_offspring = max(offspring, key=lambda x: calculate_fitness(x, ciphertext))
    # population[worst_index] = best_offspring
    #
    # return population
    # Find indices of individuals with worst fitness scores
    # Find indices of individuals with worst fitness scores

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


def local_optimization(individual, n=LOCAL_SWAPS):
    keys = list(individual.keys())
    for _ in range(n):
        index1, index2 = random.sample(range(len(keys)), 2)
        letter1, letter2 = keys[index1], keys[index2]
        individual[letter1], individual[letter2] = individual[letter2], individual[letter1]
    return individual


def darwin_mutation(population, fitness_scores, ciphertext):
    for i in range(len(population)):
        mutated_individual = mutate(population[i])
        mutated_fitness = calculate_fitness(mutated_individual, ciphertext)
        if mutated_fitness > fitness_scores[i]:
            fitness_scores[i] = mutated_fitness


def lamarck_mutation(population, fitness_scores, ciphertext):
    for i in range(len(population)):
        mutated_individual = mutate(population[i])
        mutated_fitness = calculate_fitness(mutated_individual, ciphertext)
        if mutated_fitness > fitness_scores[i]:
            fitness_scores[i] = mutated_fitness
            population[i] = mutated_individual


def handle_local_max(ciphertext,best_decryption_key, best_decryption_key_fitness):
    print("handling local max")
    for i in range(5):
        best_key, _, best_fitness = genetic_algorithm(ciphertext)
        if best_decryption_key_fitness < best_fitness:
            print(best_decryption_key_fitness)
            print(best_fitness)
            best_decryption_key_fitness = best_fitness
            best_decryption_key = best_key
        if OPTIMAL:
            return best_decryption_key

    print(best_decryption_key_fitness)
    return best_decryption_key


def generate_graph(main_text,y_text,num_generation, generation_scores ):
    plt.plot(num_generation, generation_scores)
    plt.title(
        main_text)
    plt.ylabel(y_text)
    plt.xlabel('Number of generations')
    plt.show()


def genetic_algorithm(ciphertext):
    global generations_average_fitness_scores
    global generations_best_fitness_scores
    global num_of_generations
    generations_best_fitness_scores = []
    generations_average_fitness_scores = []
    # Initialize random population
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
                print('optimal')
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

        if DARWIN:
            darwin_mutation(population, fitness_scores, ciphertext)
        if LAMARCK:
            lamarck_mutation(population, fitness_scores, ciphertext)

        # Print best decryption key and fitness score for the current generation
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

        gen_num = generation+1
    print(gen_num)
    num_of_generations = [i for i in range(gen_num)]

    return best_key, no_improvement_counter, best_fitness



ciphertext = read_text_from_file('enc.txt')

#initialize lists to generate graphs
num_of_generations = []
generations_best_fitness_scores = []
generations_average_fitness_scores = []

#create average fitness score graph
create_graph_avg = True

#create best fitness score graph
create_graph_best = True

best_decryption_key, counter, fitness = genetic_algorithm(ciphertext)

#handle local max
if counter == NO_IMPROVEMENT_THRESHOLD and not OPTIMAL:
    best_decryption_key = handle_local_max(ciphertext, best_decryption_key, fitness)

#generate graphs
if create_graph_avg:
    generate_graph('Average fitness score','Average fitness score',num_of_generations,
                   generations_average_fitness_scores)
if create_graph_best:
    generate_graph('Best fitness score', 'Best fitness score', num_of_generations,
                   generations_best_fitness_scores)

decrypted_text = decrypt_text(ciphertext, best_decryption_key)
# print("Decrypted Text:", decrypted_text)

actual_decryption_key = {
    'a': 'y',
    'b': 'x',
    'c': 'i',
    'd': 'n',
    'e': 't',
    'f': 'o',
    'g': 'z',
    'h': 'j',
    'i': 'c',
    'j': 'e',
    'k': 'b',
    'l': 'l',
    'm': 'd',
    'n': 'u',
    'o': 'k',
    'p': 'm',
    'q': 's',
    'r': 'v',
    's': 'p',
    't': 'q',
    'u': 'r',
    'v': 'h',
    'w': 'w',
    'x': 'g',
    'y': 'a',
    'z': 'f'
}

correct = 0
print(best_decryption_key)
for letter in actual_decryption_key:
    if actual_decryption_key[letter] == best_decryption_key[letter]:
        correct += 1
print("Correct:", correct, " out of", len(actual_decryption_key))
print("fitness calls: ", fitness_calls_coutner)

# creating files
with open("plain.txt", "w") as plain_file:
    plain_file.write(decrypted_text)
with open("perm.txt", "w") as perm_file:
    for letter, decrypted_letter in best_decryption_key.items():
        perm_file.write(f"{letter}\t{decrypted_letter}\n")