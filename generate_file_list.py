from pathlib import Path
import random

paths = Path('.').glob("musdb18hq/**/mixture.wav")

paths = [p for p in Path('.').glob('musdb18hq/**/mixture.wav')]
random.shuffle(paths)
# Python's random (Mersenne twister) only contains enough random state to
# shuffle up to 2080 elements fairly
assert len(paths) <= 2080

train_count = int(len(paths) * 0.6)
test_count = (len(paths) - train_count) // 2
validation_count = len(paths) - train_count - test_count


with open('train_list.txt', 'w', encoding='utf-8', newline='\n') as f:
    for path in paths[:train_count]:
        f.write(str(path))
        f.write('\n')

with open('validation_list.txt', 'w', encoding='utf-8', newline='\n') as f:
    for path in paths[train_count:train_count+validation_count]:
        f.write(str(path))
        f.write('\n')

with open('test_list.txt', 'w', encoding='utf-8', newline='\n') as f:
    for path in paths[-test_count:]:
        f.write(str(path))
        f.write('\n')