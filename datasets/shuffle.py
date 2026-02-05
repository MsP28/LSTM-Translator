import random
from tqdm import tqdm

# Define input and output file names
input_file_it = 'mixed.it'
input_file_en = 'mixed.en'
output_file_it = 'shuffled.it'
output_file_en = 'shuffled.en'

# Load lines from both files into memory
with open(input_file_it, 'r', encoding='utf-8') as file_it, open(input_file_en, 'r', encoding='utf-8') as file_en:
    # Read Italian and English lines
    print("Reading Italian and English lines...")
    lines_it = file_it.readlines()
    lines_en = file_en.readlines()

# Create a list of paired lines (Italian and English) and shuffle them
print("Shuffling lines...")
paired_lines = list(zip(lines_it, lines_en))
random.shuffle(paired_lines)

# Unzip the shuffled pairs
print("Unzipping shuffled lines...")
shuffled_it, shuffled_en = zip(*paired_lines)
# Write the shuffled lines to new output files, with a progress bar
with open(output_file_it, 'w', encoding='utf-8') as out_it, open(output_file_en, 'w', encoding='utf-8') as out_en:
    for it_line, en_line in tqdm(zip(shuffled_it, shuffled_en), total=len(shuffled_it), desc="Writing to file"):
        out_it.write(it_line)
        out_en.write(en_line)

print("Shuffling complete! Files saved as 'shuffled.it' and 'shuffled.en'.")
