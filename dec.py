import random

# Global variables
POPULATION_SIZE = 100
MAX_GENERATIONS = 90
MUTATION_RATE = 0.12
REPLACEMENT_RATE = 0.15
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001
NO_IMPROVEMENT_THRESHOLD = 12

DARWIN = False
LAMARCKN = False

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
    fitness_calls_coutner += 1
    decrypted_text = decrypt_text(ciphertext, decryption_key)
    decrypted_text_set = set(decrypted_text.split(" "))
    letter_freq = calculate_letter_frequency_from_text(decrypted_text)
    letter_pair_freq = calculate_letters_pairs_frequency_from_text(decrypted_text)
    words_matching = 0
    letters_matching = 0

    for word in word_list:
        if word == '':
            continue
        if word in decrypted_text_set:
            words_matching += 1

    for letter in letter_freq_map:
        if letter in letter_freq and letter in letter_freq_map:
            letters_matching += (1 - abs(letter_freq[letter] - letter_freq_map[letter])) ** 2

    for letter_pair in letter_pair_freq_map:
        if letter_pair in letter_pair_freq and letter_pair in letter_pair_freq_map:
            letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter_pair_freq_map[letter_pair])) ** 2.5

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


import random

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

def genetic_algorithm(ciphertext):
    # Initialize random population
    population = [generate_random_key() for _ in range(POPULATION_SIZE)]
    no_improvement_counter = 0
    best_fitness = 0
    for generation in range(MAX_GENERATIONS):
        fitness_scores = []
        offspring = []

        # Calculate fitness for each individual
        for decryption_key in population:
            fitness = calculate_fitness(decryption_key, ciphertext)
            fitness_scores.append(fitness)

        # Select parents and create offspring
        for _ in range(POPULATION_SIZE//2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            offspring.append(mutated_child)

        # Replace population with offspring
        population = replace_population(population, offspring, fitness_scores)

        # Print best decryption key and fitness score for the current generation
        temp_best_fitness = best_fitness
        best_index = fitness_scores.index(max(fitness_scores))
        best_key = population[best_index]
        best_fitness = fitness_scores[best_index]

        if best_fitness - temp_best_fitness < EPSILON:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        if no_improvement_counter == NO_IMPROVEMENT_THRESHOLD:
            break
        print(f"Generation: {generation+1} | Best Fitness: {best_fitness} | Best Decryption Key: {best_key}")

    return best_key

ciphertext = read_text_from_file('enc.txt')
best_decryption_key = genetic_algorithm(ciphertext)
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
for letter in actual_decryption_key:
    if actual_decryption_key[letter] == best_decryption_key[letter]:
        correct += 1
print("Correct:", correct, " out of", len(actual_decryption_key))
print("fitness calls: ", fitness_calls_coutner)
