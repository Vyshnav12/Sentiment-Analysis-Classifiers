# Sentiment Analysis of Movie and Product Reviews

## Overview

This project implements sentiment analysis on two datasets—Rotten Tomatoes movie reviews and Nokia phone reviews—using two different classifiers: **Naive Bayes** and **Rule-Based** approaches. The goal is to determine the effectiveness of these methods in classifying reviews as positive or negative.

## Datasets

All datasets required for the analysis are placed in the `data` folder.

- **Positive and Negative Words Lists**:
  - `positive-words.txt`: Contains a list of positive sentiment words.
  - `negative-words.txt`: Contains a list of negative sentiment words.

- **Rotten Tomatoes Reviews**:
  - `rt-polarity.pos`: A collection of positive movie reviews.
  - `rt-polarity.neg`: A collection of negative movie reviews.

- **Nokia Phone Reviews**:
  - `nokia-pos.txt`: A collection of positive Nokia phone reviews.
  - `nokia-neg.txt`: A collection of negative Nokia phone reviews.

## Methodology

### 1. Naive Bayes Classifier

The Naive Bayes Classifier is a probabilistic model that applies Bayes' theorem with the assumption of independence between every pair of features. In this context, the features are the presence of specific words in a review. The classifier calculates the probability of a review being positive or negative based on the frequency of positive and negative words.

### 2. Rule-Based Classifier

The Rule-Based Classifier uses a predefined list of positive and negative words to classify the sentiment of reviews. It counts the occurrence of positive and negative words in a review and classifies the review based on which count is higher. If a review contains more positive words, it's classified as positive, and vice versa.

## Results

### 1. Naive Bayes Classifier

| **Dataset**                  | **Accuracy** | **Precision (Positive)** | **Recall (Positive)** | **F-measure (Positive)** | **Precision (Negative)** | **Recall (Negative)** | **F-measure (Negative)** |
|------------------------------|--------------|--------------------------|-----------------------|--------------------------|--------------------------|-----------------------|--------------------------|
| Rotten Tomatoes Test Data     | 79.43%       | 0.7945                   | 0.7989                | 0.797                    | 0.7940                   | 0.7896                | 0.792                    |
| Nokia Reviews                 | 58.27%       | 0.7778                   | 0.5645                | 0.654                    | 0.3817                   | 0.6250                | 0.474                    |

### 2. Rule-Based Classifier

| **Dataset**                  | **Accuracy** | **Precision (Positive)** | **Recall (Positive)** | **F-measure (Positive)** | **Precision (Negative)** | **Recall (Negative)** | **F-measure (Negative)** |
|------------------------------|--------------|--------------------------|-----------------------|--------------------------|--------------------------|-----------------------|--------------------------|
| Rotten Tomatoes Test Data     | 56.73%       | 0.5646                   | 0.6234                | 0.593                    | 0.5708                   | 0.5102                | 0.539                    |
| Nokia Reviews                 | 67.67%       | 0.7941                   | 0.7258                | 0.758                    | 0.4688                   | 0.5625                | 0.511                    |

## Most Influential Words

<div style="display: flex; justify-content: space-between;">

### Top 10 Negative Predictors

<table>
  <tr>
    <th style="color: red;">Word</th>
    <th style="color: red;">Weight</th>
  </tr>
  <tr>
    <td>stupid</td>
    <td>0.0368</td>
  </tr>
  <tr>
    <td>routine</td>
    <td>0.0414</td>
  </tr>
  <tr>
    <td>generic</td>
    <td>0.0414</td>
  </tr>
  <tr>
    <td>mediocre</td>
    <td>0.0432</td>
  </tr>
  <tr>
    <td>unfunny</td>
    <td>0.0432</td>
  </tr>
  <tr>
    <td>badly</td>
    <td>0.0432</td>
  </tr>
  <tr>
    <td>dull</td>
    <td>0.0545</td>
  </tr>
  <tr>
    <td>mindless</td>
    <td>0.0552</td>
  </tr>
  <tr>
    <td>boring</td>
    <td>0.0585</td>
  </tr>
  <tr>
    <td>poorly</td>
    <td>0.0585</td>
  </tr>
</table>

### Top 10 Positive Predictors

<table>
  <tr>
    <th style="color: green;">Word</th>
    <th style="color: green;">Weight</th>
  </tr>
  <tr>
    <td>chilling</td>
    <td>0.0629</td>
  </tr>
  <tr>
    <td>wonderfully</td>
    <td>0.0629</td>
  </tr>
  <tr>
    <td>gem</td>
    <td>0.0592</td>
  </tr>
  <tr>
    <td>mesmerizing</td>
    <td>0.0559</td>
  </tr>
  <tr>
    <td>delicate</td>
    <td>0.0559</td>
  </tr>
  <tr>
    <td>inventive</td>
    <td>0.0559</td>
  </tr>
  <tr>
    <td>refreshingly</td>
    <td>0.0530</td>
  </tr>
  <tr>
    <td>realistic</td>
    <td>0.0530</td>
  </tr>
  <tr>
    <td>refreshing</td>
    <td>0.0503</td>
  </tr>
  <tr>
    <td>engrossing</td>
    <td>0.0314</td>
  </tr>
</table>


</div>

## Interpretation of Results

### Naive Bayes Classifier
The Naive Bayes Classifier generally performs well on the Rotten Tomatoes dataset, exhibiting balanced precision and recall for both positive and negative reviews. This suggests that the model is capable of accurately distinguishing between positive and negative sentiment in a relatively large and diverse dataset. However, its performance on the Nokia reviews dataset is less consistent. This inconsistency may indicate that the model struggles with domain-specific language or the smaller size of the dataset, which might lead to less reliable predictions.

### Rule-Based Classifier
The Rule-Based Classifier, while showing reasonable performance, is generally less accurate than the Naive Bayes Classifier. Interestingly, it performs better on the Nokia dataset compared to the Rotten Tomatoes dataset. This improved performance on Nokia reviews suggests that predefined word lists can be effective in contexts where the language is more straightforward and predictable. However, in more diverse datasets like movie reviews, where language usage can be more varied and nuanced, the Rule-Based approach may lack the sophistication needed to accurately classify sentiment.

## Conclusion

This project demonstrates the application of a probabilistic model and a rule-based approach to sentiment analysis. The Naive Bayes classifier generally outperformed the Rule-Based classifier on the Rotten Tomatoes dataset, but its performance was more variable on the Nokia reviews. The results highlight the importance of dataset characteristics in determining the effectiveness of different sentiment analysis methods.

## Potential Improvements

### Data Augmentation
Expanding the training dataset with more examples, particularly for the Nokia reviews, could improve the model's performance. A larger dataset would provide the model with more diverse examples, potentially leading to better generalization and more accurate predictions, especially in domains with limited data.

### Feature Engineering
Incorporating additional features, such as bigrams or part-of-speech tags, might enhance the model's ability to capture context. Bigrams could help the model understand common word pairs that convey sentiment, while part-of-speech tagging could improve its ability to differentiate between similar words used in different contexts.

### Advanced Models
Experimenting with more sophisticated models like Support Vector Machines (SVM) or deep learning approaches could provide further insights and potentially improve accuracy. These models can capture complex patterns in the data that simpler models might miss, making them well-suited for handling nuanced language in sentiment analysis tasks.

## License

This project is licensed under the MIT License.
