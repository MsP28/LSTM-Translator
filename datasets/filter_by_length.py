dataset_name = 'WikiMatrix'

def select_pairs(file1, file2, min_len, max_len):
  """
  Seleziona le coppie di linee da due file la cui lunghezza è compresa tra min_len e max_len.

  Args:
    file1: Il percorso del primo file.
    file2: Il percorso del secondo file.
    min_len: La lunghezza minima delle linee da selezionare.
    max_len: La lunghezza massima delle linee da selezionare.

  Returns:
    (file1_out, file2_out): I percorsi dei due nuovi file.
  """

  # Apriamo i due file in modalità lettura.
  with open(file1, "r", encoding='utf8') as f1, open(file2, "r", encoding='utf8') as f2:
    # Inizializziamo due iteratori sui due file.
    line1 = f1.readline()
    line2 = f2.readline()

    # Creiamo due nuovi file in modalità scrittura.
    file1_out = open('./' + dataset_name + '/en-it_lenfiltered.it', "w", encoding='utf8')
    file2_out = open('./' + dataset_name + '/en-it_lenfiltered.en', "w", encoding='utf8')

    # Iteriamo sui due file, selezionando le coppie di linee valide.
    total_lines = 0
    matched_lines = 0
    index = 0

    while line1 and line2:
      # Controlliamo che la lunghezza delle linee sia compresa tra min_len e max_len.
      if (min_len <= len(line1) <= max_len) and (min_len <= len(line2) <= max_len):
        # Scriviamo le linee sui due nuovi file.
        file1_out.write(line1)
        file2_out.write(line2)
        matched_lines += 1

      # Leggiamo la prossima linea dal primo file.
      line1 = f1.readline()

      # Se il primo file è terminato, terminiamo anche la selezione.
      if not line1:
        break

      # Leggiamo la prossima linea dal secondo file.
      line2 = f2.readline()

      # Aggiorniamo la percentuale di esecuzione.
      total_lines += 1
      index += 1
      if total_lines % 100000 == 0:
        print(f" Indice linea corrente: {index}", end='\r')

    # Chiudiamo i due nuovi file.
    file1_out.close()
    file2_out.close()

  # Restituiamo i percorsi dei due nuovi file.
  return "file1_out", "file2_out"

file1 = './' + dataset_name + '/en-it.it'
file2 = './' + dataset_name + '/en-it.en'

# La lunghezza minima e massima delle linee da selezionare.
min_len = 4
max_len = 130

# Selezioniamo le coppie di linee valide e le scriviamo su due nuovi file.
file1_out, file2_out = select_pairs(file1, file2, min_len, max_len)