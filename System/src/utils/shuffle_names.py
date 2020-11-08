import random, sys

with open(sys.argv[1], 'r') as f:
    names = f.readlines()
    names = list(filter(lambda x: x.strip(), names))

random.shuffle(names)

try:
    with open(sys.argv[2], 'w') as f:
        f.write(''.join(names))
    print(f"[shuffle_names.py] Names in '{sys.argv[1]}' are shuffled to '{sys.argv[2]}'!")
except:
    with open(sys.argv[1], 'w') as f:
        f.write(''.join(names))
    print(f"[shuffle_names.py] Names in '{sys.argv[1]}' are shuffled to '{sys.argv[1]}'!")

