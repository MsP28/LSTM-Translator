# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
from datetime import datetime
import numpy as np
import sentencepiece as spm
import tensorflow as tf
# from keras.utils import plot_model


# Define hyperparameters
embed_dim = 256 # Dimension of token embeddings, incapsula la semantica della frase. Più alto, più informazioni, ma più tempo di calcolo e memoria
num_lstm = 4
latent_dim = 512 + 128 # Dimension of hidden state of LSTM

batch_size = 128 - 64 # Batch size for training
val_batch_size = 128 - 64 # Validation batch size
validation_split = 0.05
epochs = 1000 # Maximum number of epochs

# Carica i modelli di tokenizzazione
tokenizer_input = spm.SentencePieceProcessor(model_file='./vocabs/it_SHUFFLED_LOWER_CC_MATRIX.model')
tokenizer_output = spm.SentencePieceProcessor(model_file='./vocabs/en_SHUFFLED_LOWER_CC_MATRIX.model')
num_encoder_tokens = tokenizer_input.vocab_size()
num_decoder_tokens = tokenizer_output.vocab_size()
max_encoder_seq_length = 50
max_decoder_seq_length = 50

# Funzione per leggere le linee tokenizzate dai file
def read_tokenized_lines(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield list(map(int, line.strip().split()))

# Funzione per creare il dataset
def create_dataset(it_file_path, en_file_path, size, validation_size):
    # Leggi le linee tokenizzate dai file
    it_dataset = tf.data.Dataset.from_generator(lambda: read_tokenized_lines(it_file_path), output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))
    en_dataset_in = tf.data.Dataset.from_generator(lambda: read_tokenized_lines(en_file_path), output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))
    en_dataset_target = tf.data.Dataset.from_generator(lambda: read_tokenized_lines(en_file_path), output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))

    # Aggiungi i token di inizio e fine alle frasi inglesi
    start_token = [tokenizer_output.piece_to_id('<s>')]
    end_token = [tokenizer_output.piece_to_id('</s>')]
    en_dataset_in = en_dataset_in.map(lambda x: tf.concat([start_token, x], axis=0))
    en_dataset_target = en_dataset_target.map(lambda x: tf.concat([x, end_token], axis=0))

    # Crea il dataset finale
    dataset = tf.data.Dataset.zip(((it_dataset, en_dataset_in), en_dataset_target))

    # Dividi il dataset in training e validation
    shuffle_chunck = size//100000 + 1
    dataset = dataset.shuffle(shuffle_chunck)
    
    val_dataset = dataset.take(validation_size)
    train_dataset = dataset.skip(validation_size)

    return train_dataset, val_dataset

def get_len(x, _):
    return tf.shape(x[0])[0]

input_file_name = 'datasets/tokenized/CC_WIKI_LOWER_MEDIUM.it'
target_file_name = 'datasets/tokenized/CC_WIKI_LOWER_MEDIUM.en'

# Crea il dataset
# Valuto dimensione del dataset
bucket_bound = [4,5,6,7,8,9,10,11,12,14,16,18,20,24,30,40]
size = 0
file_size = os.path.getsize(input_file_name)
max_num_MB = 1500
max_num_bytes = max_num_MB * 1024 * 1024
print(f'File size: {file_size} bytes')
print(f'Max num bytes: {max_num_bytes} bytes')
num_bytes = min(max_num_bytes, file_size)
print('Stimo dimensione del dataset...', end = '')
with open(input_file_name, 'r', encoding='utf8') as file:
    # read the file in chunks of num_bytes
    chunk = file.read(num_bytes)
    num_linee = chunk.count('\n') # Count the number of lines
    dimensione_file = file_size # Get the file size in bytes
    size = num_linee * (dimensione_file // num_bytes) # Estimate the total number of lines    
print(f'\rDimensione del dataset: {size} (circa)')

validation_size = int( validation_split * size) #min( int( validation_split * size), val_batch_size * 3000)
print(f'Dimensione del dataset di training: {size - validation_size}')
print(f'Dimensione del dataset di validation: {validation_size}')

training_steps = int( (size - validation_size) /batch_size)
validation_steps = int(validation_size/val_batch_size)
print(f'Training steps: {training_steps}')
print(f'Validation steps: {validation_steps}')

# set a up constraint for the number of steps
max_training_steps = 2000
max_validation_steps = 500
training_steps = min(training_steps, max_training_steps)
validation_steps = min(validation_steps, max_validation_steps)
# Print new steps
print(f'New training steps: {training_steps}')
print(f'New validation steps: {validation_steps}')

print('Costruisco i dataset...', end = '')
dataset_tr, dataset_val = create_dataset(input_file_name, target_file_name, size, validation_size)

# shuffle_chunck = size // 100000
# print(f'\rRipeto e mescolo i dataset (chunck = {shuffle_chunck})')
# dataset_tr = dataset_tr.shuffle(shuffle_chunck , reshuffle_each_iteration=True)
dataset_tr = dataset_tr.bucket_by_sequence_length(
        element_length_func=get_len,
        bucket_boundaries=bucket_bound,
        bucket_batch_sizes=[batch_size]*(len(bucket_bound)+1)
        )
dataset_tr=dataset_tr.prefetch(tf.data.AUTOTUNE)

dataset_val = dataset_val.bucket_by_sequence_length(
        element_length_func=get_len,
        bucket_boundaries=bucket_bound,
        bucket_batch_sizes=[val_batch_size]*(len(bucket_bound)+1)
        )
dataset_val=dataset_val.prefetch(tf.data.AUTOTUNE)

# Costruisco l'encoder
encoder_inputs = tf.keras.Input(shape=(None,), name =  "enc_inputs")   # shape = (batch, seq_len)

encoder_embedding = tf.keras.layers.Embedding(input_dim=num_encoder_tokens,
                                      output_dim=embed_dim,
                                      name = 'enc_emb') # shape = (batch, seq_len, embed)

encoder_lstms = [
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(latent_dim//2,
                                      return_state=True,
                                      return_sequences=True,
                                      # kernel_initializer='glorot_normal',
                                      dropout = 0.2,
                                      name = "enc_lstm_" + str(i + 1)), name = 'enc_bidirectional_' + str(i+1) )  # shape = [(batch, seq_len, latent), (batch, latent), (batch, latent)]
                    for i in range(num_lstm)
                ]

encoder_skip_concatenate = tf.keras.layers.Concatenate(name = "enc_skip_concatenate")

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
    state_h = tf.concat([state_h, state_h_back], -1, name = 'encoder_states_concatenate' + str(i+1))
    state_c = tf.concat([state_c, state_c_back], -1, name = 'encoder_states_concatenate' + str(i+1))
    # encoder_lstm_output, state_h, state_c = encoder_lstms[i](encoder_lstm_output)
    encoder_lstm_states_h.append(state_h)
    encoder_lstm_states_c.append(state_c)

encoder_lstm_states = list(zip(encoder_lstm_states_h, encoder_lstm_states_c))   # shape = [ (batch, latent), (batch, latent) ]*num_lstm

# Costruisco il decoder
decoder_inputs = tf.keras.Input(shape=(None,), name = "dec_inputs")  # Ex. shape = (batch, seq_len)

decoder_embedding = tf.keras.layers.Embedding(input_dim=num_decoder_tokens,
                                              output_dim=embed_dim,
                                              name = 'dec_emb'
                                             )  #Ex. shape = (batch, seq_len, embed) (acts on inputs)


###############################################################################################################
###############################################################################################################
# Tentativo di attention
attention_dim = 64
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
    
def WeightCreator(x): # att_lstm_sequence, enc_out
    encoder_seq_length = tf.keras.backend.shape(x[1])[1]
    att_dim = tf.keras.backend.shape(x[0])[2]
    batch = tf.keras.backend.shape(x[0])[0]
    # Ridimensiono la sequenza del decoder
    reshape_matrix = tf.eye(att_dim, encoder_seq_length, batch_shape=[batch])
    weights = tf.matmul(x[0], reshape_matrix)
    # weights = tf.tile(tf.expand_dims(weights, 0), [batch, 1, 1])
    weights = tf.nn.softmax(weights)
    return weights

# Processo l'input con una prima LSTM, i cui stati interni verranno usati per pesare l'output del decoder
attention_lstm = tf.keras.layers.LSTM(latent_dim,
                                      return_state=True,
                                      return_sequences=True,
                                      dropout=0.2,
                                      name = 'attention_lstm'
                                      )

encoder_reshape = EncoderOutputReshaper(name = 'encoder_reshape')
attention_lstm_reshape = AttentionLSTMOutputReshaper(name = 'attention_lstm_reshape')
encoder_attention_adder = tf.keras.layers.Concatenate(-1, name =  'encoder_attention_adder' ) 
attention_nework_1 = tf.keras.layers.Dense(attention_dim, activation='relu', name = 'attention_nework_1')
attention_nework_1_dropout = tf.keras.layers.Dropout(0.1, name = 'attention_nework_1_dropout')
attention_nework_2 = tf.keras.layers.Dense(attention_dim, activation='relu', name = 'attention_nework_2')
attention_nework_2_dropout = tf.keras.layers.Dropout(0.1, name = 'attention_nework_2_dropout')
attention_network_final = tf.keras.layers.Dense(1, name = 'attention_network_final')
attention_reshape = AttentionReshape(name = 'attention_reshape')
weight_creator = tf.keras.layers.Softmax(1, name = 'context_softmax')
context_creator = tf.keras.layers.Dot(1, name = 'context_creator')

context_reshaper_embed = tf.keras.layers.Dense(embed_dim, name = 'context_reshape_embed')
context_reshaper_latent = tf.keras.layers.Dense(latent_dim, name = 'context_reshape_latent')
decoder_context_concat_embed = tf.keras.layers.Concatenate(name = 'context_concatenate_embed')
decoder_context_concat_latent = tf.keras.layers.Concatenate(name = 'context_concatenate_latent')
###############################################################################################################
###############################################################################################################


decoder_lstms = [
                    tf.keras.layers.LSTM(latent_dim,
                                      return_state=True,
                                      return_sequences=True,
                                      # kernel_initializer='glorot_normal',
                                      dropout = 0.1,
                                      name = "dec_lstm_" + str(i + 1))
                    for i in range(num_lstm)
                ]
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens,
                                   activation='softmax',
                                   name = "dec_dense")

decoder_lstm_states_h = []
decoder_lstm_states_c = []

decoder_embedding_out = decoder_embedding(decoder_inputs)

# Tentativo numero 2
processed_decoder_inputs,_, _ = attention_lstm(decoder_embedding_out)
encoder_output_repeated = encoder_reshape([encoder_lstm_output, processed_decoder_inputs])
attention_lstm_repeated = attention_lstm_reshape([processed_decoder_inputs, encoder_lstm_output])
encoder_attention_adder = encoder_attention_adder([attention_lstm_repeated, encoder_output_repeated])
attention_nework_1_out = attention_nework_1(encoder_attention_adder)
attention_nework_1_out = attention_nework_1_dropout(attention_nework_1_out)
attention_nework_2_out = attention_nework_2(attention_nework_1_out)
attention_nework_2_out = attention_nework_2_dropout(attention_nework_2_out)
attention_network_final_out = attention_network_final(attention_nework_2_out)
attention_reshaped = attention_reshape(attention_network_final_out)
weights = weight_creator(attention_reshaped)
permute_context = tf.keras.layers.Permute((2,1))(weights)
context = context_creator([ weights, encoder_lstm_output])
context_reshaped_embed = context_reshaper_embed(context)
decoder_lstm_input = decoder_context_concat_embed([decoder_embedding_out, context_reshaped_embed])

lstm_output = decoder_lstm_input #decoder_embedding_out

for i in range(len(decoder_lstms)):
    lstm_output, state_h, state_c = decoder_lstms[i](lstm_output, initial_state=encoder_lstm_states[i])
    if i < len(decoder_lstms) - 1:
        context_reshaped_lat = context_reshaper_latent(context)
        lstm_output = decoder_context_concat_latent([lstm_output, context_reshaped_lat])

decoder_outputs = tf.keras.layers.TimeDistributed(decoder_dense, name = 'time_distributed')(lstm_output)

# Compilo e salvo il modello
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Plot model - requires pydot and graphviz
# tf.keras.utils.plot_model(model, './imgs/model_' + str(num_lstm) + 'lstm' + '.png', show_shapes = True, show_layer_names = True, show_layer_activations = True)


# Callbacks
callbacks = []
model_name = 'RE-NEW_ATTENTION_' + str(num_lstm) + 'LSTM_' + str(latent_dim) + 'LATDIM_' + str(embed_dim) + 'EMBEDDIM_'
# Creo la cartella per i risultati
now = datetime.now().strftime('%b%d_%H-%M-%S')
exps_dir = 'T:\Translator\\' # os.path.join('result_flder')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)
now = datetime.now().strftime('%b%d_%H-%M-%S')
exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Callback di EarlyStopping
# ---------------------------------
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)
callbacks.append(early)

# Callback per Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, 
                            profile_batch=0,
                            write_graph = True,
                            write_images = True,
                            histogram_freq=1, # if > 0 (epochs) shows weights histograms
                            )
callbacks.append(tb_callback)

# Callback per model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'checkpoint'), 
                                                    save_weights_only=False, # True to save only weights
                                                    save_best_only=True) # True to save only the best epoch 
callbacks.append(ckpt_callback)

# Callback per reduce on plateau
# ----------------
plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=1e-5, min_delta=0.001)
callbacks.append(plateau)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
# model.summary()

try:
    model.fit(
        x = dataset_tr,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=dataset_val,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps
    )
except KeyboardInterrupt:
    model.save('./models/' + model_name + '_INTERRUPTED.keras')
    # Along with the interrupted model, save a file with the hyperparameters
    with open('./models/' + model_name + '_hyperparameters_INTERRUPTED.txt', 'w') as f:
        # dataset used
        f.write(f'input_file_name = {input_file_name}\n')
        f.write(f'target_file_name = {target_file_name}\n')
        # vocabulary model
        f.write(f'tokenizer_input = {tokenizer_input}\n')
        # hyperparameters
        f.write(f'embed_dim = {embed_dim}\n')
        f.write(f'num_lstm = {num_lstm}\n')
        f.write(f'latent_dim = {latent_dim}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'val_batch_size = {val_batch_size}\n')
        f.write(f'validation_split = {validation_split}\n')
        f.write(f'epochs = {epochs}\n')
        f.write(f'num_encoder_tokens = {num_encoder_tokens}\n')
        f.write(f'num_decoder_tokens = {num_decoder_tokens}\n')
    print('\nTraining interrotto, modello salvato.')
    quit()

model.save('./models/' + model_name + '.keras')

# Along with the model, save a file with the hyperparameters
with open('./models/' + model_name + '_hyperparameters.txt', 'w') as f:
    # dataset used
    f.write(f'input_file_name = {input_file_name}\n')
    f.write(f'target_file_name = {target_file_name}\n')
    # vocabulary model
    f.write(f'tokenizer_input = {tokenizer_input}\n')
    # hyperparameters
    f.write(f'embed_dim = {embed_dim}\n')
    f.write(f'num_lstm = {num_lstm}\n')
    f.write(f'latent_dim = {latent_dim}\n')
    f.write(f'batch_size = {batch_size}\n')
    f.write(f'val_batch_size = {val_batch_size}\n')
    f.write(f'validation_split = {validation_split}\n')
    f.write(f'epochs = {epochs}\n')
    f.write(f'num_encoder_tokens = {num_encoder_tokens}\n')
    f.write(f'num_decoder_tokens = {num_decoder_tokens}\n')
          
print('Training finito, modello salvato.')