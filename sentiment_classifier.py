#!/usr/bin/env python
import re, random, math, collections, itertools
from colorama import Fore, Style

PRINT_ERRORS=1

#------------- Function Definitions ---------------------

# Function to read and process input files
def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):
    # Reading pre-labeled input and splitting into lines
    posSentences = open('data/rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('data/rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('data/nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('data/nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    posDictionary = open('data/positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

    negDictionary = open('data/negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datasets:
    # We want to test on sentences we haven't trained on, to see how well the model generalizes to previously unseen sentences

    # Create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # Create Nokia Dataset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 


#---------------------------End Training ----------------------------------

# Implement naive bayes algorithm
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos):
    """
    Test the Naive Bayes classifier on the test data.

    Args:
        sentencesTest (dict): Dictionary of test sentences and their sentiments
        dataName (str): Name of the dataset
        pWordPos (dict): Dictionary of word probabilities given positive sentiment
        pWordNeg (dict): Dictionary of word probabilities given negative sentiment
        pWord (dict): Dictionary of overall word probabilities
        pPos (float): Prior probability of positive sentiment
    """
    pNeg = 1 - pPos

    # These variables will store results
    total = 0
    correct = 0
    totalpos = 0
    totalpospred = 0
    totalneg = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0

    # For each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)  # Collect all words

        pPosW = pPos
        pNegW = pNeg

        for word in wordList:  # Calculate over unigrams
            if word in pWord:
                if pWord[word] > 0.00000001:
                    pPosW *= pWordPos[word]
                    pNegW *= pWordNeg[word]

        prob = 0
        if pPosW + pNegW > 0:
            prob = pPosW / float(pPosW + pNegW)

        total += 1
        if sentiment == "positive":
            totalpos += 1
            if prob > 0.5:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                correct += 0
                totalnegpred += 1
        else:
            totalneg += 1
            if prob <= 0.5:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                correct += 0
                totalpospred += 1
    sumstats(correctpos, totalpospred, correctneg, totalnegpred)

# Function to calculate and print summary statistics
def sumstats(correctpos, totalpospred, correctneg,totalnegpred):
    """
    Print classification report based on the test results.

    Args:
        correctpos (int): Number of true positive predictions
        totalpospred (int): Number of predicted positive sentences
        correctneg (int): Number of true negative predictions
        totalnegpred (int): Number of predicted negative sentences
    """
    truepos = correctpos
    trueneg = correctneg 
    falsepos = totalpospred - correctpos
    falseneg = totalnegpred - correctneg
    total = truepos + trueneg + falsepos + falseneg

    if total > 0:
        accuracy = (truepos + trueneg)*100 / total
        print(f"{Fore.GREEN}Accuracy{Style.RESET_ALL}")
        print(f"Accuracy = {accuracy:.2f}%")

        print(f"\n{Fore.BLUE}Positive Reviews{Style.RESET_ALL}")
        # True Positives: correctly predicted positive sentences
        # False Positives: incorrectly predicted positive sentences
        # True Negatives: correctly predicted negative sentences
        # False Negatives: incorrectly predicted negative sentences
        if (truepos+falsepos)>0 and (truepos+falseneg)>0:
            # Precision: proportion of true positives among all predicted positive sentences
            precpos = truepos/(truepos + falsepos)
            # Recall: proportion of true positives among all actual positive sentences
            recpos = truepos/(truepos+falseneg)
            print(f"Precision (Positive Reviews) = {precpos:.4f}")
            print(f"Recall (Positive Reviews) = {recpos:.4f}")
            try:
                # F-measure: harmonic mean of precision and recall
                f_measure_pos = 2*precpos*recpos/(precpos+recpos)
                print(f"F-measure Pos = {f_measure_pos:.3f}")
            except ZeroDivisionError:
                print("F-measure Pos value is invalid as both Precision and Recall are 0")
        else:
            print("Precision and Recall for classification of Positive reviews cannot be calculated as the denominator is invalid")

        print(f"\n{Fore.RED}Negative Reviews{Style.RESET_ALL}")
        if (falseneg+trueneg)>0  and (falsepos+trueneg)>0:
            # Precision: proportion of true negatives among all predicted negative sentences
            precneg = trueneg/(falseneg+trueneg)
            # Recall: proportion of true negatives among all actual negative sentences
            recneg = trueneg/(falsepos+trueneg)
            print(f"Precision (Negative Reviews) = {precneg:.4f}")
            print(f"Recall (Negative Reviews) = {recneg:.4f}")
            try:
                # F-measure: harmonic mean of precision and recall
                f_measure_neg = 2*precneg*recneg/(precneg+recneg)
                print(f"F-measure Neg = {f_measure_neg:.3f}")
            except ZeroDivisionError:
                print("F-measure Neg value is invalid as both Precision and Recall are 0")
        else:
            print("Precision and Recall for classification of Negative reviews cannot be calculated as the denominator is invalid")
    else:
        print("Accuracy for classification cannot be calculated as the denominator is invalid")


def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    # Initialize counters for overall statistics
    total = 0  # Total number of sentences
    correct = 0  # Total number of correct classifications
    totalpos = 0  # Total number of positive sentences
    totalneg = 0  # Total number of negative sentences
    totalpospred = 0  # Total number of predicted positive sentences
    totalnegpred = 0  # Total number of predicted negative sentences
    correctpos = 0  # Number of correctly classified positive sentences
    correctneg = 0  # Number of correctly classified negative sentences

    # Iterate through each sentence in the test data
    for sentence, sentiment in sentencesTest.items():
        # Extract words from the sentence
        Words = re.findall(r"[\w']+", sentence)
        
        # Initialize sentiment score for the sentence
        score = 0
        
        # Calculate sentiment score by summing word values from the sentiment dictionary
        for word in Words:
            if word in sentimentDictionary:
                score += sentimentDictionary[word]
 
        # Increment overall counters
        total += 1
        
        # Check if the sentence is positive
        if sentiment == "positive":
            totalpos += 1
            # Check if the sentiment score meets the threshold for positive classification
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
        # Check if the sentence is negative
        else:
            totalneg += 1
            # Check if the sentiment score meets the threshold for negative classification
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1
    
    # Calculate and display statistics
    sumstats(correctpos, totalpospred, correctneg, totalnegpred)
 

def newdictclassify(sentences_test, data_name, sentiment_dictionary, threshold):
    """
    Classify sentences using a sentiment dictionary with modifications to the
    sentiment score based on capitalization, degree adverbs, and negation words.
    
    Parameters
    ----------
    sentences_test : Dict[str, str]
        Test sentences and their corresponding sentiment.
    data_name : str
        Name of the data set.
    sentiment_dictionary : Dict[str, float]
        Sentiment dictionary containing words and their sentiment scores.
    threshold : float
        Threshold for determining positive or negative sentiment.
    
    Returns
    -------
    None
    """
    total = 0
    correct = 0
    total_pos = 0
    total_neg = 0
    total_pos_pred = 0
    total_neg_pred = 0
    correct_pos = 0
    correct_neg = 0
    n_value = -1
    punc_const = 0.15
    int_const = 0.75
    dim_const = -0.75
    neg_dim_const = 0.5
    
    negation_list = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
                     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
                     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
                     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
                     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
                     "oughtnt", "shant", "shouldnt", "wasnt", "werent",
                     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
                     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
    
    deg_adv = {
        "absolutely": int_const, "amazingly": int_const, "awfully": int_const,
        "completely": int_const, "considerable": int_const, "considerably": int_const,
        "decidedly": int_const, "deeply": int_const, "enormous": int_const, "enormously": int_const,
        "entirely": int_const, "especially": int_const, "exceptional": int_const, "exceptionally": int_const,
        "extreme": int_const, "extremely": int_const, "fabulously": int_const, "fully": int_const, "too": int_const,
        "greatly": int_const, "highly": int_const, "hugely": int_const, "incredible": int_const, "incredibly": int_const, 
        "intensely": int_const, "major": int_const, "majorly": int_const, "more": int_const, "most": int_const, 
        "particularly": int_const, "purely": int_const, "quite": int_const, "really": int_const, "remarkably": int_const,
        "so": int_const, "substantially": int_const, "thoroughly": int_const, "total": int_const, "totally": int_const, 
        "tremendous": int_const, "tremendously": int_const, "uber": int_const, "unbelievably": int_const, "unusually": int_const, 
        "utter": int_const, "utterly": int_const, "very": int_const, "almost": dim_const, "barely": dim_const, "hardly": dim_const, 
        "just enough": dim_const, "kind of": dim_const, "kindof": dim_const, "less": dim_const, "little": dim_const, 
        "marginal": dim_const, "marginally": dim_const, "occasional": dim_const, "occasionally": dim_const, "partly": dim_const,
        "scarce": dim_const, "scarcely": dim_const, "slight": dim_const, "slightly": dim_const, "somewhat": dim_const,
        "sort of": dim_const, "sortof": dim_const}
    
    def punc_check(input_sent):
        """
        Count the number of exclamation marks and question marks in a sentence.
        
        Parameters
        ----------
        input_sent : str
            Input sentence.
        
        Returns
        -------
        int
            Number of punctuation marks.
        """
        em_count = input_sent.count('!')
        qm_count = input_sent.count('?')
        if qm_count > 3:
            qm_count = 3
        if em_count > 3:
            em_count = 3
        return em_count + qm_count
    
    def caps_check(caps_words, sent, word):
        """
        Check if a word is capitalized and assign a sentiment score based on
        its position in the sentence.
        
        Parameters
        ----------
        caps_words : int
            Number of capitalized words in the sentence.
        sent : str
            Input sentence.
        word : str
            Word to check.
        
        Returns
        -------
        float
            Sentiment score.
        """
        caps_value = 1
        if 0 < caps_words < len(sent) and word.isupper():
            caps_value = 1.25
        return caps_value
    
    for sent, sentiment in sentences_test.items():
        words = re.findall(r"[\w']+", sent)
        score = 0
        caps_words = 0
        unedit_sent = sent.split() #Non-delimited sentence required to assign punctuation sentiment
        
        for word in words:
            if word.isupper():
                caps_words += 1
        
        for i in range(len(words)):
            if words[i].lower() in sentiment_dictionary:
                caps_value = caps_check(caps_words, sent, words[i])
                score_upd = sentiment_dictionary[words[i]].lower() * caps_value #Base score value required by all outputs
                if i >= 1:
                    if words[i-1] in deg_adv:
                        score_upd += deg_adv[words[i-1].lower()]
                        if i >= 2 and words[i-2] in negation_list:
                            score_upd *= neg_dim_const
                    elif words[i-1] in negation_list:
                        score_upd *= n_value
                    elif i >= 2 and words[i-2] in negation_list and words[i-1] not in deg_adv:
                        score_upd *= neg_dim_const             
                score += score_upd
        
        punc_val = punc_check(sent)
        
        listr = [unedit_sent.index(x) for x in unedit_sent if x == '!' or x =='?']
        for edits in listr:
            if unedit_sent[edits-1] in sentiment_dictionary:
                score += punc_val * punc_const * (sentiment_dictionary[unedit_sent[edits-1]].lower())
        
        total += 1
        if sentiment == "positive":
            total_pos += 1
            if score >= threshold:
                correct += 1
                correct_pos += 1
                total_pos_pred += 1
            else:
                correct += 0
                total_neg_pred += 1
        else:
            total_neg += 1
            if score < threshold:
                correct += 1
                correct_neg += 1
                total_neg_pred += 1
            else:
                correct += 0
                total_pos_pred += 1
    sumstats(correct_pos, total_pos_pred, correct_neg, total_neg_pred)


def mostUseful(pWordPos, pWordNeg, pWord, n):
    """Prints out the top n most useful predictors for positive and negative sentiment.
    
    Parameters
    ----------
    pWordPos : dict
        A dictionary with words as keys and the probability of each word given positive sentiment as values.
    pWordNeg : dict
        A dictionary with words as keys and the probability of each word given negative sentiment as values.
    pWord : dict
        A dictionary with words as keys and the probability of each word as values.
    n : int
        The number of top predictors to print.
    """
    predictPower = {}
    for word in pWord:
        # If the word has a very low probability of occurrence given negative sentiment, assign it a weight of 1.0
        if pWordNeg[word] < 0.0000001:
            predictPower[word] = 1.0
        else:
            # Calculate the weight of the word as the probability of the word given positive sentiment divided by the probability of the word given negative sentiment
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])
    
    sortedPower = sorted(predictPower.items(), key=lambda x: x[1])
    
    print(f"{Fore.RED}Top {n} Negative Predictors:{Style.RESET_ALL}")
    for word, power in sortedPower[:n]:
        # Print each word with its weight, with a width of 20 and a precision of 4
        print(f"{word:<20} {power:.4f}")
    
    print(f"\n{Fore.BLUE}Top {n} Positive Predictors:{Style.RESET_ALL}")
    for word, power in sortedPower[-n:]:
        # Print each word with its weight, with a width of 20 and a precision of 4
        print(f"{word:<20} {1 - power:.4f}")



#---------- Main Script --------------------------

sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

# Initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# Build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

from colorama import Fore, Style

# Run naive bayes classifier on datasets
print(f"\n{Fore.CYAN}**Naive Bayes Classifier on Rotten Tomatoes Test Data**:{Style.RESET_ALL}")
testBayes(sentencesTest, "Rotten Tomatoes (Test Data, Naive Bayes)", pWordPos, pWordNeg, pWord, 0.5)

print(f"\n{Fore.CYAN}**Naive Bayes Classifier on Nokia Reviews**:{Style.RESET_ALL}")
testBayes(sentencesNokia, "Nokia Reviews (All Data, Naive Bayes)", pWordPos, pWordNeg, pWord, 0.7)

# Run sentiment dictionary based classifier on datasets
print(f"\n{Fore.CYAN}**Rule-Based Classifier on Rotten Tomatoes Test Data**:{Style.RESET_ALL}")
testDictionary(sentencesTest, "Rotten Tomatoes (Test Data, Rule-Based)", sentimentDictionary, -4)

print(f"\n{Fore.CYAN}**Rule-Based Classifier on Nokia Reviews**:{Style.RESET_ALL}")
testDictionary(sentencesNokia, "Nokia Reviews (All Data, Rule-Based)", sentimentDictionary, -3)

# Print most useful words
print(f"\n{Fore.CYAN}**MOST USEFUL WORDS**:{Style.RESET_ALL}")
mostUseful(pWordPos, pWordNeg, pWord, 10)
print("\n")