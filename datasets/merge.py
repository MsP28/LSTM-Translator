import itertools
import os

def line_reader(dataset, file1, file2):
    with open(os.path.join(dataset, file1), 'r', encoding='utf-8') as f1, open(os.path.join(dataset, file2), 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            yield line1, line2

def merge_files(datasets, dest_it, dest_en):
    readers = []
    for dataset in datasets:
        readers.append(line_reader(dataset, "en-it_cleaned.it", "en-it_cleaned.en"))

    with open(dest_it, 'w', encoding='utf-8') as out_it, open(dest_en, 'w', encoding='utf-8') as out_en:
        for i, (line_it, line_en) in enumerate(itertools.chain(*readers)):
            if i % 100000 == 0:  # stampa il progresso ogni 10000 righe
                print(f"\rProcessed {i} lines", end = '')
            out_it.write(line_it.lower())
            out_en.write(line_en.lower())

# usa la funzione
datasets = ['CCAligned', 'WikiMatrix']  # aggiungi qui i nomi dei tuoi dataset
merge_files(datasets, 'mixed.it', 'mixed.en')