import re
from tqdm import tqdm

DATASET_NAME = 'WikiMatrix'  # Change this to the name of the dataset you are working with (folder in datasets/)

def clean_files(input_file_it, input_file_en, output_file_it, output_file_en):
    # Define the regex pattern to match non-common characters except Italian accented characters
    pattern = re.compile(r'[^\x20-\x7E\nÀ-ÿ]')

    print("Opening files...")
    with open(input_file_it, 'r', encoding='utf-8') as file_it, open(input_file_en, 'r', encoding='utf-8') as file_en:
        lines_it = file_it.readlines()
        lines_en = file_en.readlines()

    cleaned_lines_it = []
    cleaned_lines_en = []

    for line_it, line_en in tqdm(zip(lines_it, lines_en), total=len(lines_it), desc="Cleaning lines"):
        # Remove line breaks and other special characters
        line_it = line_it.replace('\u2028', '').replace('\u2029', '')
        line_en = line_en.replace('\u2028', '').replace('\u2029', '')

        # Remove other non-common characters
        cleaned_line_it = pattern.sub('', line_it)
        cleaned_line_en = pattern.sub('', line_en)

        # Check if both lines are not empty after cleaning
        if cleaned_line_it.strip() and cleaned_line_en.strip():
            cleaned_lines_it.append(cleaned_line_it)
            cleaned_lines_en.append(cleaned_line_en)

    print("Writing cleaned lines to files...")
    with open(output_file_it, 'w', encoding='utf-8') as file_it, open(output_file_en, 'w', encoding='utf-8') as file_en:
        file_it.writelines(cleaned_lines_it)
        file_en.writelines(cleaned_lines_en)
    print("Cleaning complete! Files saved.")

# Example usage
input_file_it = f'./{DATASET_NAME}/en-it_lenfiltered.it'  # Italian file
input_file_en = f'./{DATASET_NAME}/en-it_lenfiltered.en'  # English file
output_file_it = f'./{DATASET_NAME}/en-it_cleaned.it'  # Cleaned Italian file
output_file_en = f'./{DATASET_NAME}/en-it_cleaned.en'  # Cleaned English file
clean_files(input_file_it, input_file_en, output_file_it, output_file_en)