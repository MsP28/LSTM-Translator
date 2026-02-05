# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import gc
import  copy
import math
import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import keract
from keract import get_activations
import random

# tf.compat.v1.disable_eager_execution()

# Define hyperparameters
num_lines = 800000 # Maximum number of lines to use for training and validation
embed_dim = 128 # Dimension of token embeddings, incapsula la semantica della frase. Più alto, più informazioni, ma più tempo di calcolo
num_lstm = 3
latent_dim = 512 # Dimension of hidden state of LSTM

batch_size = 128 # Batch size for training
val_batch_size = 16 # Validation batch size
epochs = 30 # Training epochs
validation_split = 0.2
training_steps = int((1-validation_split)*num_lines/batch_size)
validation_steps = int(validation_split*num_lines/val_batch_size)

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# # Load data from files
# with open('datasets/merged_tato_ntrex_euro_ted_opus.it', 'r', encoding='utf-8') as f:
#     input_lines = f.read().split('\n') # np.array(f.read().split('\n'))
# with open('datasets/merged_tato_ntrex_euro_ted_opus.en', 'r', encoding='utf-8') as f:
#     target_lines = f.read().split('\n') # np.array(f.read().split('\n'))

# couples = list(zip(input_lines, target_lines))
# couples.sort(key=lambda x: len(x[0]))
# input_lines, target_lines = zip(*couples)
# print(len(input_lines))

# # Creo le liste di frasi
# for line in target_lines[: min(num_lines, len(target_lines) - 1)]:
#     target_texts.append(line)

# for line in input_lines[: min(num_lines, len(input_lines) - 1)]:
#     input_texts.append(line)
                        
# del input_lines, target_lines #, couples
# gc.collect()


# Tokenizzo
tokenizer_input = spm.SentencePieceProcessor(model_file='it_mixed_LOWER_CC_MATRIX.model')
tokenizer_output = spm.SentencePieceProcessor(model_file='en_mixed_LOWER_CC_MATRIX.model')
num_encoder_tokens = tokenizer_input.vocab_size()
num_decoder_tokens = tokenizer_output.vocab_size()

max_decoder_seq_length = 40
max_encoder_seq_length = 32

# Custom layers
class EncoderOutputReshaper(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x = tf.expand_dims(inputs[0], 2)
        x = tf.tile(x, [1, 1 , tf.keras.backend.shape(inputs[1])[1], 1])
        return x
    
class AttentionLSTMOutputReshaper(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x = tf.expand_dims(inputs[0], 1)
        x = tf.tile(x, [1, tf.keras.backend.shape(inputs[1])[1] , 1, 1])
        return x
    
class AttentionReshape(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x =  tf.reshape(inputs, (tf.keras.backend.shape(inputs)[0], tf.keras.backend.shape(inputs)[1], tf.keras.backend.shape(inputs)[2]) )
        return x

model = keras.models.load_model('models\\RE-NEW_ATTENTION_3LSTM_512LATDIM_256EMBEDDIM_.keras', custom_objects={"EncoderOutputReshaper": EncoderOutputReshaper, "AttentionLSTMOutputReshaper": AttentionLSTMOutputReshaper, "AttentionReshape":AttentionReshape}, compile=True)

# Testo il modello
output_tokenized = tokenizer_output.EncodeAsIds("where did you go yesterday?", add_bos = True)
sentence = "dove sei andato ieri?"
input_tokenized = tokenizer_input.EncodeAsIds(sentence, add_eos = False)
print(tokenizer_input.EncodeAsPieces(sentence, add_eos = False))
input_data = np.zeros(
  (1, len(input_tokenized)), dtype="int32"
)
for t, token in enumerate(input_tokenized):
        input_data[0, t] = token
gr_data = np.zeros(
  (1, len(output_tokenized)), dtype="int32"
)
for t, token in enumerate(output_tokenized):
        gr_data[0, t] = token
prediction = model.predict([input_data, gr_data])
prediction = np.argmax(prediction[0,:,:], axis = 1)
prediction_data = np.zeros(
  (1, len(prediction)), dtype="int32"
)
for t, token in enumerate(prediction):
        prediction_data[0, t] = token
print(prediction)
print('-')
print('Input (model):', sentence)
translation = tokenizer_output.decode(prediction.tolist())
print('Translation:', translation, '(', tokenizer_output.IdToPiece(prediction.tolist()), ')')
accuracy = keras.metrics.Accuracy()
accuracy.update_state(gr_data, prediction_data)
print(accuracy.result().numpy())
# print('GR: ', target_texts[index])

# Visualizzo le attivazioni
activations = get_activations(model, [input_data, gr_data], auto_compile=True, layer_names='context_softmax')
keract.display_activations(activations, save = True, directory='./activations/', fig_size=(10,7), cmap='magma')


# Parte di traduzione della frase
# Creo l'encoder che usa i layer del modello addestrato
# Costruisco l'encoder
encoder_inputs = model.input[0]

encoder_embedding = model.get_layer('enc_emb')
encoder_lstms = [
                    model.get_layer("enc_bidirectional_" + str(i+1) )
                    for i in range(num_lstm)
                ]

encoder_skip_concatenate = model.get_layer(name = "enc_skip_concatenate")

encoder_lstm_states_h = []
encoder_lstm_states_c = []

encoder_embedding_out = encoder_embedding(encoder_inputs)
encoder_lstm_output = encoder_embedding_out

for i in range(len(encoder_lstms)):
    # Se bidirectional
    if i < len(encoder_lstms) - 1:
        encoder_lstm_output, state_h, state_h_back, state_c, state_c_back = encoder_lstms[i](encoder_lstm_output)
    else:
        encoder_lstm_output = encoder_skip_concatenate( [encoder_lstm_output, encoder_embedding_out] )
        encoder_lstm_output, state_h, state_h_back, state_c, state_c_back = encoder_lstms[i](encoder_lstm_output)
    state_h = model.get_layer(name = 'tf.concat' + ( ('_' + str(i) ) if i > 0 else '' ) )([state_h, state_h_back], axis = -1)
    state_c =  model.get_layer(name = 'tf.concat' + ( ('_' + str(i) ) if i > 0 else '' ) )([state_c, state_c_back], axis = -1)
    encoder_lstm_states_h.append(state_h)
    encoder_lstm_states_c.append(state_c)
encoder_lstm_states = list(zip(encoder_lstm_states_h, encoder_lstm_states_c))
print(len(encoder_lstm_states))


encoder_model = tf.keras.Model(encoder_inputs, [encoder_lstm_output, encoder_lstm_states])
# encoder_model.summary()
# tf.keras.utils.plot_model(encoder_model, './imgs/encoder_model_' + str(num_lstm) + 'lstm' + '.png', show_shapes = True, show_layer_names = True)


# Costruisco il decoder
decoder_inputs = model.input[1]
decoder_state_h_inputs = [ tf.keras.Input(shape=(latent_dim,), name="dec_state_h_input" + str(i+1)) for i in range(num_lstm)]
decoder_state_c_inputs = [ tf.keras.Input(shape=(latent_dim,), name="dec_state_c_input" + str(i+1)) for i in range(num_lstm)]
# decoder_context_input = tf.keras.Input(shape=(embed_dim,), name = 'dec_context_input')
decoder_states_sequences_input = tf.keras.Input(shape=(None,latent_dim), name = 'dec_states_seq_input')
decoder_attention_state_input = [ tf.keras.Input(shape = (latent_dim,), name = 'attention_state_h_input'), tf.keras.Input(shape = (latent_dim,), name = 'attention_state_c_input') ]
# decoder_context_h_inputs = keras.Input(shape=(latent_dim,), name="dec_context_h_input")
# decoder_context_c_inputs = keras.Input(shape=(latent_dim,), name="dec_context_c_input")
decoder_state_inputs = list(zip(decoder_state_h_inputs, decoder_state_c_inputs))

decoder_embedding = model.get_layer('dec_emb')

attention_lstm = model.get_layer(name = 'attention_lstm')

encoder_reshape = model.get_layer(name = 'encoder_reshape')
attention_lstm_reshape = model.get_layer(name = 'attention_lstm_reshape')
encoder_attention_adder = model.get_layer(name =  'encoder_attention_adder' ) 
attention_nework_1 = model.get_layer(name = 'attention_nework_1')
attention_nework_1_dropout = model.get_layer(name = 'attention_nework_1_dropout')
attention_nework_2 = model.get_layer(name = 'attention_nework_2')
attention_nework_2_dropout = model.get_layer(name = 'attention_nework_2_dropout')
attention_network_final = model.get_layer(name = 'attention_network_final')
attention_reshape = model.get_layer(name = 'attention_reshape')
weight_creator = model.get_layer(name = 'context_softmax')
context_creator = model.get_layer(name = 'context_creator')

context_reshaper_embed = model.get_layer(name = 'context_reshape_embed')
context_reshaper_latent = model.get_layer(name = 'context_reshape_latent')
decoder_context_concat_embed = model.get_layer(name = 'context_concatenate_embed')
decoder_context_concat_latent = model.get_layer(name = 'context_concatenate_latent')

decoder_lstms = [
                    model.get_layer("dec_lstm_" + str(i + 1))
                    for i in range(num_lstm)
                ]

decoder_lstm_states_h = []
decoder_lstm_states_c = []

decoder_embedding_out = decoder_embedding(decoder_inputs)

#Tentativo di attention
processed_decoder_inputs,attention_h, attention_c = attention_lstm(decoder_embedding_out, initial_state = decoder_attention_state_input)
encoder_output_repeated = encoder_reshape([decoder_states_sequences_input, processed_decoder_inputs])
attention_lstm_repeated = attention_lstm_reshape([processed_decoder_inputs, decoder_states_sequences_input])
encoder_attention_adder = encoder_attention_adder([attention_lstm_repeated, encoder_output_repeated])
attention_nework_1_out = attention_nework_1(encoder_attention_adder)
attention_nework_1_out = attention_nework_1_dropout(attention_nework_1_out)
attention_nework_2_out = attention_nework_2(attention_nework_1_out)
attention_nework_2_out = attention_nework_2_dropout(attention_nework_2_out)
attention_network_final_out = attention_network_final(attention_nework_2_out)
attention_reshaped = attention_reshape(attention_network_final_out)
weights = weight_creator(attention_reshaped)
permute_context = tf.keras.layers.Permute((2,1))(weights)
context = context_creator([ weights, decoder_states_sequences_input])
context_reshaped_embed = context_reshaper_embed(context)
decoder_lstm_input = decoder_context_concat_embed([decoder_embedding_out, context_reshaped_embed])

lstm_output = decoder_lstm_input #decoder_embedding_out

for i in range(len(decoder_lstms)):
    lstm_output, state_h, state_c = decoder_lstms[i](lstm_output, initial_state=decoder_state_inputs[i])
    decoder_lstm_states_h.append(state_h)
    decoder_lstm_states_c.append(state_c)
    # Di nuovo: attention
    if i < len(decoder_lstms) - 1:
        context_reshaped_lat = context_reshaper_latent(context)
        lstm_output = decoder_context_concat_latent([lstm_output, context_reshaped_lat])

decoder_lstm_states = list(zip(decoder_lstm_states_h, decoder_lstm_states_c))
decoder_outputs = model.get_layer('time_distributed')(lstm_output)

decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_state_inputs + decoder_attention_state_input + [decoder_states_sequences_input], [decoder_outputs] + [attention_h, attention_c] + decoder_lstm_states
)
# decoder_model.summary()
# plot_model(decoder_model, './imgs/decoder_model_' + str(num_lstm) + 'lstm' + '.png', show_shapes = True, show_layer_names = True)

class Token():
    def __init__(self, token, prob):
        self.token = token
        self.prob = prob
    def get_token(self):
        return self.token
    def get_prob(self):
        return self.prob

class Sequence():     
    def __init__(self):
        self.sequence = []
        self.state = None
        self.attention_state = None
        self.prob = 1
    def AddToken(self, token: Token, state, attention_state):
        self.sequence.append(token)
        self.state = state
        self.attention_state = attention_state
    def GetLastToken(self):
        return self.sequence[-1].get_token()
    def GetSequence(self):
        return [int(token.get_token()) for token in self.sequence]
    def GetState(self):
        return self.state
    def GetAttentionState(self):
        return self.attention_state
    def CalculateProb(self):
        result = 0
        for token in self.sequence:
            result += - math.log(token.get_prob())
        self.prob = result/len(self.sequence)
        return self.prob
    def isCompleted(self):
        return self.sequence[-1].get_token() == tokenizer_output.eos_id() or len(self.sequence) > max_decoder_seq_length

def BeamDecode(input_seq, K):
    # Genero il contesto
    # RICORDA: encoder_model = keras.Model(encoder_inputs, [lstm_output, encoder_lstm_states])
    enc_out = encoder_model.predict(input_seq, verbose = 0)
    enc_sequences = enc_out[0]
    states = enc_out[1:]

    # Beam Search
    all_ended = False

    completed_seq = []
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_output.bos_id()
    attention_state = [tf.zeros(shape = (1,latent_dim))]*2

    # Genero la distribuzione di probabilità dei token successivi a quello di start e estraggo i k più probabili
    decoder_outputs = decoder_model.predict([target_seq] + states + attention_state + [enc_sequences], verbose = 0)   # Uso decoder per ottenere l'output
    out_tokens_distribution = decoder_outputs[0]
    attention_state = decoder_outputs[1:3]
    new_state = decoder_outputs[3:]
    token_hypotesis = np.argpartition(out_tokens_distribution, -K, axis = 2)[:, :, -K:]
    
    # print('   token proposti: ', [tokenizer_output.decode(int(token_hypotesis[0,0,i])) for i in range(len(token_hypotesis[0,0,:]))], ' - prob: ', [out_tokens_distribution[0,0,index] for index in token_hypotesis[0,0,:]])

    # Generazione dei primi K token (probabili future sequenze)
    for last_seq_token in token_hypotesis[0,-1,:]:
        seq = Sequence()
        seq.AddToken(Token(last_seq_token, out_tokens_distribution[0,-1,last_seq_token]), new_state, attention_state)
        completed_seq.append(seq)

    # print('attuali completed: ', [tokenizer_output.decode(seq.GetSequence()) for seq in completed_seq])

    if all(sequence.isCompleted() for sequence in completed_seq):
        all_ended = True

    counter = 0
    while not all_ended:
        print(' Ciclo numero ', counter, end = '\n')
        frontier = []
        # print('sequenze complete: ', len(completed_seq))
        for i in range(len(completed_seq)):
            # print(' entro nel for sulle seq complete ', i, ' di ', len(completed_seq))
            if not completed_seq[i].isCompleted():
                # print(' entro nella condizione non è completa')
                target_seq[0,0] = completed_seq[i].GetLastToken()
                # Genero la distribuzione di probabilità dei token successivi a quello precedente e estraggo i k più probabili
                decoder_outputs = decoder_model.predict([target_seq] + completed_seq[i].GetState() + [completed_seq[i].GetAttentionState()] + [enc_sequences], verbose = 0)   # Uso decoder per ottenere l'output
                out_tokens_distribution = decoder_outputs[0]
                attention_state = decoder_outputs[1:3]
                new_state = decoder_outputs[3:]
                token_hypotesis = np.argpartition(out_tokens_distribution, -K, axis = 2)[:,:,-K:]

                # print('   token proposti: ', [tokenizer_output.decode(int(token_hypotesis[0,0,i])) for i in range(len(token_hypotesis[0,0,:]))], ' - prob: ', [out_tokens_distribution[0,0,index] for index in token_hypotesis[0,0,:]])

                # Generazione dei primi K token e aggiunta alla frontiera
                for last_seq_token in token_hypotesis[0,-1,:]:
                    new_seq = copy.deepcopy(completed_seq[i])
                    new_seq.AddToken(Token(last_seq_token, out_tokens_distribution[0,-1,last_seq_token]), new_state, attention_state)
                    frontier.append(new_seq)

            else:
                # print(' entro nella condizione è completa')
                frontier.append(completed_seq[i])   

        # print('attuali completed: ', [tokenizer_output.decode(seq.GetSequence()) for seq in completed_seq])
        # print('frontiera attuale: ', [tokenizer_output.decode(seq.GetSequence()) for seq in frontier])

        # Ora abbiamo le nuove K*K (o meno, se alcune erano finite) proposte di sequenze
        # Devo mantenere nella frontiera leK più probabili
        probabilities = np.array([sequence.CalculateProb() for sequence in frontier])
        index_of_seq_to_mantain = np.argpartition(probabilities, -K)[:K]
        best_in_frontier = [frontier[i] for i in index_of_seq_to_mantain]
        completed_seq = best_in_frontier

        # print('  completed = ', len(completed_seq), ' - frontier = ', len(frontier))
        # print('migliori della frontiera: ', [tokenizer_output.decode(seq.GetSequence()) for seq in completed_seq])

        if all(sequence.isCompleted() for sequence in completed_seq):
            all_ended = True

        counter += 1
    
    probabilities = np.array([sequence.CalculateProb() for sequence in completed_seq])
    result_index = np.argmin(probabilities)
    result = completed_seq[result_index].GetSequence()
    # print()
    # print(result)

    return tokenizer_output.decode(result)

nodes = ['dec_input', 'dec_emb', 'attention_state_c_input', 'attention_state_h_input', 'attention_lstm', 'dec_states_seq_input', 'attention_lstm_reshape', 'encoder_reshape', 'encoder_attention_adder', 'attention_nework_1', 'attention_nework_1_dropout', 'attention_nework_2', 'attention_nework_2_dropout', 'attention_network_final', 'attention_reshape', 'context_softmax']

# Funzione di decodifica
def decode_sequence(input_seq):
    enc_out = encoder_model.predict(input_seq, verbose = 0)
    enc_sequences = enc_out[0]
    states_value = enc_out[1:]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_output.bos_id()

    attention_state = [np.zeros(shape = (1,latent_dim))]*2

    stop_condition = False
    decoded_sentence = []
    sampled_token_index = 0
    while not stop_condition:
        # activations = get_activations(decoder_model, [target_seq] + states_value + attention_state + [enc_sequences], auto_compile=True, layer_names='context_softmax' )
        # keract.custom_display_activations(activations, directory='./activations' + str(sampled_token_index) + '/', fig_size=(10,7), cmap='magma')

        decoder_outputs = decoder_model.predict([target_seq] + states_value + attention_state + [enc_sequences], verbose = 0)
        output_tokens = decoder_outputs[0]
        attention_state = decoder_outputs[1:3]
        states = decoder_outputs[3:]
        # print(states_value[0][0][0][0:5])

        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        decoded_sentence += [sampled_token_index]

        if sampled_token_index == tokenizer_output.eos_id() or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = states
    return tokenizer_output.decode(decoded_sentence)

while True:
    sentence = input('Inserisci la frase in italiano: ').lower()
    input_tokenized = tokenizer_input.EncodeAsIds(sentence, reverse=False, add_eos = False)
    while len(input_tokenized) >= max_encoder_seq_length:
        print('Hai inserito una frase troppo lunga!')
        sentence = input("Inserisci un'altra frase in italiano: ").lower()
        input_tokenized = tokenizer_input.EncodeAsIds(sentence, reverse=False, add_eos = False)
    input_data = np.zeros(
        (1, len(input_tokenized)), dtype="int32"
        )
    for t, token in enumerate(input_tokenized):
                input_data[0, t] = token
    print('-')
    print('Input:', sentence)
    translation = decode_sequence(input_data)
    # translation_beam = BeamDecode(input_data, 3)
    print('Translation:', translation)
    # print('Beam Translation: ', translation_beam)
    print('-')
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# couples = list(zip(input_texts, target_texts))
# random.shuffle(couples)
# input_texts, target_texts = zip(*couples)
# validation_lines = 500
# input_tokenized = [tokenizer_input.EncodeAsIds(sentence, reverse=False) for sentence in input_texts[:validation_lines]]
# input_data = [np.zeros(
#         (1, len(tokenized)), dtype="int32"
#         ) for tokenized in input_tokenized]
# for i in range(validation_lines):
#     for t, token in enumerate(input_tokenized[i]):
#             input_data[i][0, t] = token
# translations = []
# for i in range(validation_lines):
#     translations.append(decode_sequence(input_data[i]).split())
#     print(' ', int(i/validation_lines*1000)/10, '%', end = '\r')
# ground_truths = [sentence.split() for sentence in target_texts[:validation_lines]]
# references_list = [[ref] for ref in ground_truths]
# bleu_score_corpus = corpus_bleu(references_list, translations)
# print("Corpus BLEU Score: ", bleu_score_corpus)

# index = 145969
# sentence = input_texts[index] # 'Sono molto contento!'
# sentence = 'Ho letto tre libri.'
# input_tokenized = tokenizer_input.EncodeAsIds(sentence, reverse=False)
# input_data = np.zeros(
#   (1, len(input_tokenized)), dtype="int32"
# )
# for t, token in enumerate(input_tokenized):
#         input_data[0, t] = token
# print('-')
# print('Input:', sentence)
# translation = decode_sequence(input_data)
# print('Translation:', translation)
# print('GR: ', target_texts[index])

# # CHECK DELL'ACCURACY
# trans_tokenized = tokenizer_output.Encode(translation)
# output_tokenized = tokenizer_output.Encode(target_texts[index])

# translation_data = np.zeros(
#   (1, max(len(trans_tokenized), len(output_tokenized))), dtype="int32"
# )
# for t, token in enumerate(trans_tokenized):
#         translation_data[0, t] = token
# gr_data = np.zeros(
#   (1, max(len(trans_tokenized), len(output_tokenized))), dtype="int32"
# )
# for t, token in enumerate(output_tokenized):
#         gr_data[0, t] = token


# accuracy = keras.metrics.Accuracy()
# accuracy.update_state(gr_data, translation_data)
# print(accuracy.result().numpy())

