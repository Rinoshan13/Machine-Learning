import numpy as np
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

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

binary_array = recode_message(sms_data,D_corpus)

print(binary_array)


# Construct a trainig and testing dataset
def train_test_split(X,Y, train_percentage =0.8):
    assert X.shape[0] == Y.shape[0]

    number_of_data = X.shape[0]
    num_train = int(train_percentage * number_of_data)

    indices = np.random.permutation(number_of_data)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

Y = sms_data[:,0]

np.random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(binary_array, Y)
print(f"Train size:{X_train.shape[0]}, Test size:{X_test.shape[0]}")
print(f"Spam in train:{np.sum(Y_train == 'spam')}, Ham in train:{np.sum(Y_train == 'ham')}")


# Class priors
n_spam_train = np.sum(Y_train == 'spam')
n_ham_train = np.sum(Y_train == 'ham')
N_train = len(Y_train)
p_spam = n_spam_train / N_train
p_ham = n_ham_train / N_train
print(f"P(spam) = {p_spam:.4f}, P(ham) = {p_ham:.4f}")

def estimate_proportions(data_matrix):
    counts= np.sum(data_matrix,axis=0)
    N = data_matrix.shape[0]
    theta = (counts +1) / (N + 2)
    return theta

#Compute the theta matrix
theta_ham = estimate_proportions(X_train[Y_train=="ham"])
theta_spam = estimate_proportions(X_train[Y_train=="spam"])
Theta = np.column_stack([theta_ham,theta_spam])
print("Theta shape:",Theta.shape)

#Barplot of top 30 words: class-conditional proba
top_words_idx = np.argsort(theta_spam)[::-1][:30]
word_list = list(D_corpus.keys())
print(word_list)
top_words = [word_list[i] for i in top_words_idx]

fig,axes = plt.subplots(2,1,figsize = (14,8))
axes[0].bar(range(30), theta_ham[top_words_idx], color='steelblue')
axes[0].set_xticks(range(30))
axes[0].set_xticklabels(top_words, rotation=45, ha='right')
axes[0].set_title('P(word | ham) - top 30 spam words')
axes[0].set_ylabel('theta_ham')

axes[1].bar(range(30), theta_spam[top_words_idx], color='tomato')
axes[1].set_xticks(range(30))
axes[1].set_xticklabels(top_words, rotation=45, ha='right')
axes[1].set_title('P(word | spam) - top 30 spam words')
axes[1].set_ylabel('theta_spam')

plt.tight_layout()
plt.savefig(HERE/"spam_ham_top_words.png", dpi=300, bbox_inches="tight")
plt.show()


# Classfication using MAP
def compute_log_posterior(X,Theta,p_spam,p_ham):
    log_theta_ham = np.log(Theta[:,0])
    log_theta_spam = np.log(Theta[:,1])
    log_1_theta_ham = np.log(1-Theta[:,0])
    log_1_theta_spam = np.log(1-Theta[:,1])

    # log P(x | ham) = sum_i [ x_i*log(theta_h_i) + (1-x_i)*log(1-theta_h_i) ]
    log_p_x_ham = X @ log_theta_ham + (1-X) @log_1_theta_ham
    log_p_x_spam = X @ log_theta_spam + (1-X) @ log_1_theta_spam

    log_post_ham = log_p_x_ham +np.log(p_ham)
    log_post_spam = log_p_x_spam +np.log(p_spam)

    # P(spam|x) via log-diff trick
    log_diff = log_post_spam - log_post_ham
    p_spam_given_x = 1 / (1 + np.exp(-log_diff))
    return p_spam_given_x

probs_test = compute_log_posterior(X_test, Theta, p_spam, p_ham)
preds_test = probs_test >= 0.5

TP = np.sum((preds_test == True) & (Y_test == 'spam'))
FN = np.sum((preds_test == False) & (Y_test == 'spam'))
FP = np.sum((preds_test == True) & (Y_test == 'ham'))
TN = np.sum((preds_test == False) & (Y_test == 'ham'))

TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)   
accuracy = (TP + TN) / len(Y_test)

print(f"True Positive Rate (Sensitivity): {TPR:.4f}")
print(f"False Positive Rate (1-Specificity): {FPR:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TP={TP}, FN={FN}")
print(f"  FP={FP}, TN={TN}")


## ROC Curve 
thresholds = np.linspace(0, 1, 200)
tprs = []
fprs = []

for t in thresholds:
    preds = probs_test >= t
    tp = np.sum((preds == True) & (Y_test == 'spam'))
    fn = np.sum((preds == False) & (Y_test == 'spam'))
    fp = np.sum((preds == True) & (Y_test == 'ham'))
    tn = np.sum((preds == False) & (Y_test == 'ham'))
    tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

plt.figure(figsize=(7, 6))
plt.plot(fprs, tprs, color='darkorange', lw=2, label='Bernoulli NB ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random classifier')
plt.scatter([FPR], [TPR], color='red', zorder=5, label=f'Threshold=0.5 (TPR={TPR:.2f}, FPR={FPR:.2f})')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve - Bernoulli Naive Bayes Spam Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(HERE/"spam_ham_top_words.png", dpi=300, bbox_inches="tight")
plt.show()
