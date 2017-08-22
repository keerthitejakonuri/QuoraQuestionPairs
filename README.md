# QuoraQuestionPairs

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. 

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question.

Currently, Quora uses a Random Forest model to identify duplicate questions.

This is a kaggle competition. Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not.

The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. 

And, We developed a Machine Learning Ensemble Model (Boosting) to identify the question pairs that have the same intent in a Kaggle competition combining two weak learners semantic similarity and order similarity.

Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

Data Description:

The Training Data contains 404351 question pairs (instances) with 255045 negative samples (non-duplicates) and 149306 positive samples (duplicates), approximately 40% positive samples.

Data fields:
• id - the id of a training set question pair
• qid1, qid2 - unique ids of each question (only available in train.csv)
• question1, question2 - the full text of each question
• is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning

A sample of the data is as follows:
![quoraquestionpair-datadescription](https://user-images.githubusercontent.com/16725903/29592663-1d46778a-876b-11e7-9720-b92c28ce1b81.png)



Label distribution:
Class Label 0 indicates that the question pair is not duplicate, and 1 indicates that the question pair is duplicate.

Interaction Diagram:

![quoraquestionpair-interactiondiagram](https://user-images.githubusercontent.com/16725903/29592501-2473a290-876a-11e7-9056-c841c691ae45.png)




Given two words, w1 and w2, we need to find the semantic similarity s(w1,w2). We can do this by analysis of the lexical knowledge base (in this paper, we have used WordNet) as follows: Words are organized into synonym sets (synsets) in the knowledge base [26], with semantics and relation pointers to other synsets. Therefore, we can find the first class in the hierarchical semantic network that
subsumes the compared words. One direct method for similarity calculation is to find the minimum length of path connecting the two words [30]. For example, the shortest path between boy and girl in Fig. 2 is boy-male-person-femalegirl, the minimum path length is 4, the synset of person is called the subsumer for words of boy and girl, while the minimum path length between boy and teacher is 6. Thus, we
could say girl is more similar to boy than teacher to boy. Rada et al. [30] demonstrated that this method works well on
their much constrained medical semantic nets (with 15,000 medical terms).
However, this method may be less accurate if it is applied to larger and more general semantic nets such as WordNet [26]. For example, the minimum length from boy to animal is 4, less than from boy to teacher, but, intuitively, boy is more similar to teacher than to animal (unless you are cursing the boy). To address this weakness, the direct path length method must be modified by utilizing more
information from the hierarchical semantic nets. It is apparent that words at upper layers of the hierarchy have more general semantics and less similarity between them, while words at lower layers have more concrete semantics
and more similarity. Therefore, the depth of word in the hierarchy should be taken into account. In summary, similarity between words is determined not only by path lengths but also by depth. We propose that the similarity s(w1,w2) between words w1 and w2 is a function of path length and depth as follows:
s(w1,w2) fðl; hÞ; ð1Þ
where l is the shortest path length between w1 and w2, h is the depth of subsumer in the hierarchical semantic nets. We
assume that (1) can be rewritten using two independent functions as:
sðw1; w2Þ ¼ f1ðlÞ  f2ðhÞ: ð2Þ f1 and f2 are transfer functions of path length and depth, respectively. We call these information sources, of path length and depth, attributes.
Properties of Transfer Functions:
The values of an attribute in (2) may cover a large range up to infinity, while the interval of similarity should be finite with extremes of exactly the same to no similarity at all. If we assign exactly the same with a value of 1 and no similarity
as 0, then the interval of similarity is [0, 1]. The direct use of information sources as a metric of similarity is inappropriate due to its infinite property. Therefore, it is intuitive that the transfer function from information sources to semantic similarity is a nonlinear function. Taking path length as an example, when the path length decreases to zero, the similarity would monotonically increase toward the limit 1, while path length increases infinitely (although this would not happen in an organized lexical database), the similarity should monotonically decrease to 0. Therefore, to meet these constraints the transfer function must be a nonlinear
function. The nonlinearity of the transfer function is taken into account in the derivation of the formula for semantic similarity between two words, as in the following sections.
Contribution of Path Length:
For a semantic net hierarchy, as in Fig. 2, the path length between two words, w1 and w2, can be determined from one of three cases:
1. w1 and w2 are in the same synset.
2. w1 and w2 are not in the same synset, but the synset for w1 and w2 contains one or more common words.
For example, in Fig. 2, the synset for boy and synset for girl contain one common word child.
3. w1 and w2 are neither in the same synset nor do their synsets contain any common words.
Case 1 implies that w1 and w2 have the same meaning;
we assign the semantic path length between w1 and w2 to 0.
Case 2 indicates that w1 and w2 partially share the same
features; we assign the semantic path length between w1 and w2 to 1. For case 3, we count the actual path length between w1 and w2. Taking the above considerations into account, we set f1ðlÞ in (2) to be a monotonically decreasing function of l:
fiðlÞ ¼ el; ð3Þ
where  is a constant. The selection of the function in
exponential form ensures that f1 satisfies the constraints

Scaling Depth Effect:

Words at upper layers of hierarchical semantic nets have more general concepts and less semantic similarity between words than words at lower layers. This behavior must be taken into account in calculating sðw1; w2Þ. We therefore need to scale down sðw1; w2Þ for subsuming words at upper
layers and to scale up sðw1; w2Þ for subsuming words at lower layers. As a result, f2ðhÞ should be a monotonically increasing function with respect to depth h. We set f2 as:
f2ðhÞ ¼
e h   e  h
e h þ e  h ; ð4Þ
where   > 0 is a smoothing factor. As  !1, then the depth of a word in the semantic nets is not considered. In summary, we propose a formula for a word similarity measure as:
sðw1; w2Þ ¼ e  l  
e h   e  h
e h þ e  h ; ð5Þ
where   2 ½0; 1 ;   2 ð0; 1  are parameters scaling the contribution of shortest path length and depth, respectively.
The optimal values of   and   are dependent on the knowledge base used and can be determined using a set of word pairs with human similarity ratings. For WordNet, the optimal parameters for the proposed measure are: ¼0:2 and   ¼ 0:45, as reported in [20].


Semantic Similarity between Sentences:
Given two sentences, T1 and T2, a joint word set is formed:
T ¼ T1 [ T2¼ fw1 q2 . . . wmg:
The joint word set T contains all the distinct words from T1 and T2. Since inflectional morphology may cause a word to appear in a sentence with different forms that convey a specific meaning for a specific context, we use word form as it appears in the sentence. For example, boy and boys, woman and women are considered as four distinct words and all included in the joint word set. Thus, the joint word set for
two sentences:
1. T1: RAM keeps things being worked with.
2. T2: The CPU uses RAM as a short-term memory store. is:
T ¼fRAM keeps things being worked with
The CPU uses as a short-term memory storage:
Since the joint word set is purely derived from the compared sentences, it is compact with no redundant information. The joint word set, T, can be viewed as the semantic information for the compared sentences. Each sentence is readily represented by the use of the joint word set as follows: The vector derived from the joint word set is called the lexical semantic vector, denoted by _s. Each entry of the semantic vector corresponds to a word in the joint word set, so the dimension equals the number of words in the joint word set. The value of an entry of the lexical semantic vector, _siði ¼ 1; 2; . . .;mÞ, is determined by the semantic similarity of the corresponding word to a word in the sentence. Take T1 as an example:
Case 1. If wi appears in the sentence, _si is set to 1.
Case 2. If wi is not contained in T1, a semantic similarity score is computed between wi and each word in the
sentence T1, using the method presented in Section 3.1. Thus, the most similar word in T1 to wi is that with the
highest similarity score &. If & exceeds a preset threshold, then _si ¼ &; otherwise, _si ¼ 0.


Word Order Similarity between Sentences:

Let us consider a pair of sentences, T1 and T2, that contain exactly the same words in the same order with the exception of two words from T1 which occur in the reverse order in T2. For example:
. T1: A quick brown dog jumps over the lazy fox.
. T2: A quick brown fox jumps over the lazy dog.
Since these two sentences contain the same words, any methods based on ”bag of words” will give a decision that T1 and T2 are exactly the same. However, it is clear for a human interpreter that T1 and T2 are only similar to some extent. The dissimilarity between T1 and T2 is the result of the different word order. Therefore, a computational method for sentence similarity should take into account
the impact of word order. For the example pair of sentences T1 and T2, the joint
word set is:
T ¼ fA quick brown dog jumps over the lazy foxg:
We assign a unique index number for each word in T1 and T2. The index number is simply the order number in which the word appears in the sentence. For example, the index number is 4 for dog and 6 for over in T1. In computing the word order similarity, a word order vector, r, is formed for T1 and T2, respectively, based on the joint word set T.
Taking T1 as an example, for each word wi in T, we try to find the same or the most similar word in T1 as follows:
1. If the same word is present in T1, we fill the entry for this word in r1 with the corresponding index number from T1. Otherwise, we try to find the most similar word ~wi in T1 (as described in Section 3.2).
2. If the similarity between wi and ~wi is greater than a preset threshold, the entry of wi in r1 is filled with the index number of ~wi in T1.
3. If the above two searches fail, the entry of wi in r1 is 0. Having applied the procedure on the previous page, the word order vectors for T1 and T2 are r1 and r2, respectively.
For the example sentence pair, we have:
r1 ¼ f1 2 3 4 5 6 7 8 9g
r2 ¼ f1 2 3 9 5 6 7 8 4g:
Thus, a word order vector is the basic structural information carried by a sentence. The task of dealing with word order is then to measure how similar the word order in two sentences is. We propose a measure for measuring the word order similarity of two sentences as:
Sr ¼ 1  
k r1   r2 k
k r1 þ r2 k
: ð8Þ
That is, word order similarity is determined by the normalized difference of word order. The following
analysis will demonstrate that Sr is an efficient metric for indicating word order similarity. To simplify the analysis, we will consider only a single word order difference, as in sentences T1 and T2.



Overall Sentence Similarity:
Semantic similarity represents the lexical similarity. On the other hand, word order similarity provides information
about the relationship between words: which words appear in the sentence and which words come before or after other words. Both semantic and syntactic information (in terms of word order) play a role in conveying the meaning of sentences. Thus, the overall sentence similarity is defined as a combination of semantic similarity and word order similarity:

SðT1; T2Þ ¼ _Ss þ ð1 _ _ÞSr
¼ _
s1 _ s2
ks1k _ ks2k
þ ð1 _ _Þ
kr1 _ r2k
kr1 þ r2k;
where _ _ 1 decides the relative contributions of semantic and word order information to the overall similarity computation. Since syntax plays a subordinate role for semantic processing of text [11], _ should be a value greater
than 0.5, i.e., _ 2 ð0:5; 1_.




Conclusion:
From this Project, we came to know that the Similarity Metrics used for calculating Document Similarity cannot be applied to compute Short Text Sentence Similarity.
For this, we have to predict the similarity something nearer to Human Intuition. We made efforts to achieve something nearer to human intuition, in this project.
