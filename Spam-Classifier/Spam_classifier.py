import numpy as np
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent #Spam-Classifier Folder

# Load SMS data
sms_data = np.loadtxt(HERE/"SMSSpamCollection_cleaned.csv",delimiter="\t",skiprows=1, dtype=str)

# Check data size
num_message = sms_data.shape[0]

# Corpus D construction
def corpus_contruction(data):
    word_dictionary = {}
    index=0

    for i in range(len(data)):
        sms_words =data[i][1].split()
        for sms_word in sms_words:
            if sms_word not in word_dictionary:
                word_dictionary[sms_word]= index
                index +=1

    return word_dictionary

#Print the corpus D
D_corpus = corpus_contruction(sms_data)
print(f"Number of unique words in the dataset:",len(D_corpus))

# Count the 10 most common words
word_doc_count = Counter()
for i in range(len(sms_data)):
    word_in_msg = set(sms_data[i][1].split())
    word_doc_count.update(word_in_msg)
print("10 most common words:", word_doc_count.most_common(10))


# Recoding the message into binary matrix
def recode_message(data,corpus):
    binary_array = np.zeros((len(data),len(corpus)),dtype=float)

    for i in range(len(data)):
        sms_words = data[i][1].split()
        for sms_word in sms_words:
            if sms_word in corpus:
                j=corpus[sms_word]
                binary_array[i][j] =1

    return binary_array

b_array = recode_message(sms_data,D_corpus)

print(b_array)



     