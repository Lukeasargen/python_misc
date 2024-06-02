import itertools

def generate_combinations():
    prefixes = ['m', 'p']
    endings = ['ace', 'ain', 'pin', 'pie', 'air', 'acn']
    length = 6

    combinations = []

    for prefix in prefixes:
        for ending in endings:
            for comb in itertools.product('abcdefghijklmnopqrstuvwxyz', repeat=length - len(prefix) - len(ending)):
                word = prefix + ''.join(comb) + ending
                if 'h' in word:
                    combinations.append(word)

    return combinations

if __name__ == "__main__":
    result = generate_combinations()
    print("\n".join(result))
    print(len(result))
