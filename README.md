# QuoraQuestionPairs
Developed a Machine Learning Ensemble Model (Boosting) to identify the question pairs that have the same intent in a Kaggle competition combining two weak learners semantic similarity and order similarity.
Design
Project Project Project Project Project DescriptionDescription Description DescriptionDescriptionDescriptionDescriptionDescriptionDescription
To identify question pairs that have the same intent. This is an active Kaggle Competition.
Dataset DescriptionDescriptionDescriptionDescription DescriptionDescriptionDescriptionDescriptionDescription
The dataset was downloaded from https://www.kaggle.com/c/quora-question-pairs/data.
The Training Data contains 404351 question pairs (instances) with 255045 negative samples (non-duplicates) and 149306 positive samples (duplicates), approximately 40% positive samples.
Data fields:
• id - the id of a training set question pair
• qid1, qid2 - unique ids of each question (only available in train.csv)
• question1, question2 - the full text of each question
• is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
A sample of the data is as follows:
Label distribution:
Class Label 0 indicates that the question pair is not duplicate, and 1 indicates that the question pair is duplicate.
Data Preprocessing Data PreprocessingData PreprocessingData PreprocessingData Preprocessing Data PreprocessingData PreprocessingData PreprocessingData PreprocessingData Preprocessing
The records that have Null values in the Questions columns are simply ignored.
Why are they ignored?
It’s impossible to calculate a similarity between a Null and a Question. Also, it is not a good choice to replace a Null with a random Question of our own interest. We have used Python Pandas for this.
We have removed unnecessary characters from the Question texts.
The stop words and function words are not removed because they are required to compute the overall Semantic Meaning of the sentence.
Algorithm Analysis Algorithm AnalysisAlgorithm AnalysisAlgorithm AnalysisAlgorithm Analysis Algorithm AnalysisAlgorithm AnalysisAlgorithm AnalysisAlgorithm Analysis
Already existing methods that are used for calculation of Document similarity are not suitable for this particular problem, which is calculating the similarity for short texts like sentences.
So, we have proposed a similarity metric that is a combination of Semantic Vector Similarity and Word Order Vector Similarity, which is closer to Human Intuition.
Although, the similarity of two sentences is based upon the number of common words, they may not exactly be similar in their meaning. This is the reason why we had to consider both Semantic Vector Similarity and Word Order Vector Similarity.
The different stages in the Algorithm are:
1. Interaction Diagram

As mentioned in the Dataset Description above, we get a pair of sentences (questions) to predict whether they are duplicate or not.
From Sentence1 and Sentence2, we have taken the common words in both the sentences as Joint Word Set without removing the functional words, as shown in the Figure. Using Sentence1 and Sentence2 along with the Joint Word Set, and WordNet Lexical Database, we have derived Raw Semantic Vector 1, Order Vector 1 for Sentence1, and Raw Semantic Vector 2, Order Vector 2 for Sentence2.
Raw Semantic Vector: The dimensions of Raw Semantic Vector is based on the Joint Word Set, so the number of dimensions is equal to the size (m) of the Joint Word Set.
Vector Creation Process: It is based on the similarity between two words, the common word from the Joint Word Set and the most similar word in the sentence. So, we get m different coefficients for each vector. And, these are just based on the similarity between two words using WordNet.
Similarity between words: This is a function of path length of the two words (Path Similarity), and the depth of the two words in the taxonomy and that of their least common ancestor (WUP Similarity).
We used WordNet for deriving Path Similarity and WUP Similarity.
Semantic Vector: The Raw Semantic Vector derived in the above step is used to create the Actual Semantic Vector. In Raw Semantic Vector, we are giving each and every dimension the same weight. But, the functional words and common words would not contribute much to the meaning of the sentence, and different words would contribute much more. So, we have assigned different weights for different dimensions using “Brown Corpus” data. We can approximate the frequency of occurrence of words in general using Brown Corpus and thus able to assign weight for each and every dimension. Therefore, the Raw Semantic Vector is transformed into an Actual Semantic Vector which gives the actual meaning.
Cosine Similarity: From the above two Semantic Vectors, we have computed the Cosine Similarity as
Sc = (S1.S2)/(||S1||.||S2||)
Normalized Euclidean Distance: From the above two Semantic Vectors, we have computed the Euclidean Distance as
Se = (Sqrt(Sum(S1 – S2)2)/ (||S1||.||S2||))
Manhattan Distance: From the above two Semantic Vectors, we have computed the Manhattan Distance as
Sm = Sum(|S1 – S2|)/ (||S1||.||S2||))
We have got almost similar accuracies for Cosine Similarity and Euclidean Distance with a little higher accuracy using Cosine Similarity.
Word Order Vector: The Word Order Vector is computed using Joint Word Set and the two sentences, and it is computed using the index of the words appeared in the sentences.
Word Order Vector Similarity: It is calculated using
Sr = 1 – (||r1 – r2||/||r1 + r2||)
Sentence Similarity: The Sentence Similarity is calculated using Word Order Vector Similarity and the Semantic Vector Similarity (calculated using Cosine Similarity) as
S(sentence1, sentence2) = KSc + (1 – K)Sr
Where K is a constant parameter > 0.5
By repeated experiments, we have got the best results for K = 0.85.
As this is a Supervised Learning, we used the Training Data to decide the Threshold for overall similarity of two sentences and by repeated experiments, we have got the best results for
Threshold = 0.75
Performance Evaluation: The various Performance Evaluation Metrics used are:
Accuracy = (Total number of correct predictions)/(Total number of records)
Precision = (True Positive)/(True Positive + False Positive), from the Confusion Matrix
Recall = (True Positive)/(True Positive + False Negative), from the Confusion Matrix
F-measure = (2*True Positive)/((2*True Positive) + False Positive + False Negative)
2. Analysis of Results Analysis of ResultsAnalysis of ResultsAnalysis of ResultsAnalysis of ResultsAnalysis of Results Analysis of ResultsAnalysis of Results Analysis of ResultsAnalysis of Results
After repeated experiments, we have got the best results for K = 0.85, where K is a constant parameter used in Sentence Similarity.
The Threshold for overall similarity of two sentences, by repeated experiments, is estimated to be
Threshold = 0.75
3. Conclusion
From this Project, we came to know that the Similarity Metrics used for calculating Document Similarity cannot be applied to compute Short Text Sentence Similarity.
For this, we have to predict the similarity something nearer to Human Intuition. We made efforts to achieve the above, in this project.
