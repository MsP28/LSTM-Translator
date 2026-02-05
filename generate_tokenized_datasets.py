import sentencepiece as spm

MAX_LINES_PER_FILE = 10000000
NUM_SAMPLES = 2
MAX_INDEX = MAX_LINES_PER_FILE / NUM_SAMPLES

# Carica i modelli di tokenizzazione
it_tokenizer = spm.SentencePieceProcessor(model_file='./vocabs/it_SHUFFLED_LOWER_CC_MATRIX.model')
en_tokenizer = spm.SentencePieceProcessor(model_file='./vocabs/en_SHUFFLED_LOWER_CC_MATRIX.model')

# Funzione per leggere le linee di testo dai file
def read_lines(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            yield line.strip()

# Funzione per tokenizzare le linee con SentencePiece e scrivere su file
def tokenize_and_write(it_file_path, en_file_path, it_lines, en_lines, it_tokenizer, en_tokenizer):
    with open(it_file_path, 'w', encoding='utf8') as it_f, open(en_file_path, 'w', encoding='utf8') as en_f:
        index = 0
        print('Tokenizzazione delle linee...', end='\n')
        for it_line, en_line in zip(it_lines, en_lines):
            # Genera NUM_SAMPLES campioni di tokenizzazione per ogni linea
            for _ in range(NUM_SAMPLES):
                it_tokens = it_tokenizer.encode(it_line, out_type=int, enable_sampling=True, alpha=0.1)
                en_tokens = en_tokenizer.encode(en_line, out_type=int, enable_sampling=True, alpha=0.1)
                it_f.write(' '.join(map(str, it_tokens)) + '\n')
                en_f.write(' '.join(map(str, en_tokens)) + '\n')
            index += 1

            # Stampa il numero di linee processate ogni 10.000 linee e stima la percentuale di completamento sul totale massimo di linee
            if index % 10000 == 0:
                print(f'\r{index} linee processate ({index / MAX_INDEX:.2%} completato)', end='')

            # Se abbiamo raggiunto il numero massimo di linee per file, esci dal ciclo
            if index >= MAX_INDEX:
                break

# Leggi le linee dai file
it_lines = read_lines('datasets/shuffled.it')
en_lines = read_lines('datasets/shuffled.en')

# Tokenizza le linee e scrivi su file
tokenize_and_write('datasets/tokenized/CC_WIKI_LOWER_MEDIUM.it', 'datasets/tokenized/CC_WIKI_LOWER_MEDIUM.en', it_lines, en_lines, it_tokenizer, en_tokenizer)
print('Tokenizzazione completata!')