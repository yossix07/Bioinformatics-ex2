import random

# Global variables
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.2

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
        # # Use regular expression to match whole words
        # pattern = r"\b" + re.escape(word) + r"\b"
        # if re.search(pattern, decrypted_text, re.IGNORECASE):
            words_matching += 1

    #print(words_matching)
    # for each letter, add more fitness as it gets closer to the expected frequency
    for letter in letter_freq_map:
        if letter in letter_freq and letter in letter_freq_map:
            #the letter_freq returns the number of times the letter appears
            #but letter_freq_map is float
            #need to divided the letter_freq by the number of letters in the text?
            letters_matching += (1 - abs(letter_freq[letter] - letter_freq_map[letter]))

    # # for each pair of letters, add more fitness as it gets closer to the expected frequency
    for letter_pair in letter_pair_freq_map:
        if letter_pair in letter_pair_freq and letter_pair in letter_pair_freq_map:
            letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter_pair_freq_map[letter_pair]))

    # return the fitness
    return words_matching+letters_matching
# def calculate_fitness(decryption_key, ciphertext):
#     decrypted_text = decrypt_text(ciphertext, decryption_key)
#     decrypted_text_set = set(decrypted_text.split(" "))
#     letter_freq = calculate_letter_frequency_from_text(decrypted_text)
#     letter_pair_freq = calculate_letters_pairs_frequency_from_text(decrypted_text)
#     words_matching = len(decrypted_text_set.intersection(word_list))
#
#     total_letters = sum(letter_freq.values())
#     normalized_letter_freq = {letter: count / total_letters for letter, count in letter_freq.items()}
#     letters_matching = 0
#
#     for letter in letter_freq_map:
#         if letter in normalized_letter_freq and letter in letter_freq_map:
#             letters_matching += (1 - abs(normalized_letter_freq[letter] - letter_freq_map[letter]))
#
#     for letter_pair in letter_pair_freq_map:
#         if letter_pair in letter_pair_freq and letter_pair in letter_pair_freq_map:
#             letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter_pair_freq_map[letter_pair]))
#
#     return words_matching + letters_matching

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
    print(parent1)
    print(parent2)
    keys = list(parent1.keys())
    crossover_point = random.randint(0, len(keys)-1)
    child = {}
    # for i in range(len(keys)):
    #     if i <= crossover_point:
    #         child[keys[i]] = parent1[keys[i]]
    #     else:
    #         child[keys[i]] = parent2[keys[i]]
    #
    #     #     while letter in child.values():
    #     #         # If the letter is already present, find the next available letter
    #     #         letter = parent2[keys[i]]
    #     #
    #     # child[keys[i]] = letter
    #     print(child)
    # Copy the letters from parent1 up to the crossover point
    for i in range(crossover_point + 1):
        child[keys[i]] = parent1[keys[i]]
    print("here")
    # Fill the remaining letters from parent2, ensuring uniqueness
    for i in range(crossover_point + 1, len(keys)):
        letter = parent2[keys[i]]

    # Check if the letter is already present in the child
        # Check if the letter is already present in the child
        while letter in child.values():
            # Find a letter that doesn't appear in the child
            available_letter = next(l for l in parent2.values() if l not in child.values())

            # Replace one of the duplicate letters with the available letter
            duplicate_letter = next(k for k, v in child.items() if v == letter)
            child[duplicate_letter] = available_letter

            letter = parent2[keys[i]]  # Update the current letter with the new one

        child[keys[i]] = letter

    #print(child)
    return child

def mutate(decryption_key):
    mutated_key = decryption_key.copy()
    for letter in mutated_key:
        if random.random() < MUTATION_RATE:
            random_letter = random.choice(list(mutated_key.keys()))
            mutated_key[letter], mutated_key[random_letter] = mutated_key[random_letter], mutated_key[letter]
    return mutated_key

def select_parents(population, fitness_scores):
    parents = random.choices(population, weights=fitness_scores, k=2)
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


    worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:40]

    # Find indices of individuals with best fitness scores from the existing population
    best_population_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:10]

    # Find indices of individuals with best fitness scores from the offspring
    best_offspring_indices = sorted(range(len(offspring)), key=lambda i: calculate_fitness(offspring[i], ciphertext),
                                    reverse=True)[:10]

    # Replace worst fitness scores with best fitness scores from the existing population
    for i in range(10):
        population[worst_indices[i]] = population[best_population_indices[0]]
       # print(worst_indices[i])

    # Replace remaining worst fitness scores with best fitness scores from the offspring
    for i in range(10):
        population[worst_indices[i + 20]] = offspring[best_offspring_indices[i]]
        #print(worst_indices[i+20])

    # for i in range(10):
    #     population[worst_indices[i+10]] = population[best_population_indices[i]]
    #     #print(worst_indices[i+10])

    return population

def genetic_algorithm(ciphertext):
    # Initialize population
    population = [generate_random_key() for _ in range(POPULATION_SIZE)]

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
        best_index = fitness_scores.index(max(fitness_scores))
        best_key = population[best_index]
        best_fitness = fitness_scores[best_index]
        print(f"Generation: {generation+1} | Best Fitness: {best_fitness} | Best Decryption Key: {best_key}")

        # Termination condition
        # if best_fitness == len(word_list):
        #     break

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
