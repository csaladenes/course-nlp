# nlp 2022

Code-first, hand-on approach 
 - presenting a specific NLP topics using Notebook environment (e.g: Colab)
 - using most popular NLP libararies and tools

## Topics

### First Session (Nov 4 - 5)
- Introduction (1h)
  - Popular use cases
  - Philosophical debate about how to model and improve language learning
- Sentiment Analysis (scikit-learn + fast.ai) (2.5 h)
  - Dataset - IMDB reviews
  - Cloud Service example - AWS Comprehend
  - bag-of-word approach
  - Naive Bayes - using frequency counts
- Topic Modeling - LSA/LDA

### Second Session (Nov 18 - 19)

Block 1
 
	Introductions
	Recap 
		- use cases of NLP (Notion AI, Aircall voicemail transcription)
		- exercise: sentiment analysis of imdb movie reviews
		- solution tiers: 
			- bag-of-words models (Naicve Bayes, Logistic Regression)
			- sequential models (Language Models, RNN, Transformers)
	Evaluation - theory
		- accuracy, False Positives/False Negatives
		- F1 score, ROC - AUC
		- sklearn DummyClassifier
	Recap
		- Naive Bayes method for sentiment analysis

Block 2
	
	Vectorization and GPUs - theory
	Linear regression - theory
		- interactive website
	Logistic Regression - notebook
		- sigmoid, log-loss
		- term-dcoument matrix, sparse matrix
		- tokenization and vocabulary
		- N-Grams
	Talk to books - by google - NLP use case

Block 3

	Recap 
		- Logistic Regression method for sentiment analysis
	Sequential models - theory
		- RNN
		- Attention mechanism
		- Transformers
	Word Vector/Word Embeddings - theory and intution
	Word Embeddings - notebook 
	
Block 4

	Language Models - theory
	Pre-trained transformer fine-tuning - notebook
		- tokenization
		- hidden state extraction
		- feature matrix and Logistic Regression
		- HuggingFace pipelines - high level library

	Whisper - notebook
		- open source ASR
		- record voice - transcribe - translate
		- translation
		- Why is this model so good?
			- trained on multiple related tasks translation-transcription
			- trained on multiple languages

### Third Session (Dec  9 - 10)

- Zoltán és Orsolya 
  - [Magyar nyelmodellek létrehozása és felhasználása a tartalomelemzésben](https://www.nyest.hu/hirek/apanak-munkaja-van-anyanak-teste)
  - https://github.com/crow-intelligence/mccLMs
  - https://www.nyest.hu/hirek/a-gep-az-ember-tukre


### NLP libraries

- NLTK - released in 2001, very broud NLP tasks
- spacy - opinionated, parse trees, tokenizers
- gensim - topic modeling, similarity detection
- huggingface - transformer based models
- fastText - text classification and representation learning
- scikit-learn - general purpose ML library
- PyText - deep learning framework for NLP (based on pyTorch)
- fast.ai - deep learning library (based on pyTorch)


### Might be useful

https://towardsdatascience.com/how-to-compute-sentence-similarity-using-bert-and-word2vec-ab0663a5d64  
https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e
https://www.kdnuggets.com/2022/11/getting-started-spacy-nlp.html

## Drive folder - notebook & text files

https://drive.google.com/drive/folders/11FdU2aY3atYFES2gSnukyKBLEwPYeRAo?usp=sharing

## Links
- https://pair.withgoogle.com/explorables/measuring-fairness/
- https://www.geogebra.org/m/xC6zq7Zv
- https://books.google.com/talktobooks/

AntConc / MSNLP / Magyar Nemzeti Szovegtar
