import sentencepiece as spm

DICTIONARY_SIZE = 12000
DICTIONARY_NAME = 'SHUFFLED_LOWER_CC_MATRIX'

spm.SentencePieceTrainer.train(input='datasets/shuffled.en', model_prefix=f'./vocabs/en_{DICTIONARY_NAME}', vocab_size=DICTIONARY_SIZE, input_sentence_size=4000000, shuffle_input_sentence=True, pad_id=0, bos_id=1, eos_id=2, unk_id=3)
spm.SentencePieceTrainer.train(input='datasets/shuffled.it', model_prefix=f'./vocabs/it_{DICTIONARY_NAME}', vocab_size=DICTIONARY_SIZE, input_sentence_size=4000000, shuffle_input_sentence=True, pad_id=0, bos_id=1, eos_id=2, unk_id=3)