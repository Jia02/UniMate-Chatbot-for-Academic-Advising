# Dialo-GPT model 
A Dialo-GPT model that is downloaded from [HuggingFace](https://huggingface.co/microsoft/DialoGPT-small?text=Hey+my+name+is+Mariama%21+How+are+you%3F) and trained based on the official document of Academic Regulation from the university. To complete the model training process, open the ipynb file on *Google Colab* or *Jupyter Notebook* and run the file accordingly. To evaluate the trained model based on the metrics, run the Python script on *Visual Studio Code*.

## Training
The training folder containing the files for training the model. The steps of completing model training involve: 

1. **Training data preparation:** A function is employed for parsing JSON files that store the post-processed data. This data structure is loaded into memory as a Python list of question-answer pairs. The ***modifiedQ&A.json*** is  formatted precisely in this manner.

2. **Dataset creation:** A class takes in a list of data and a tokenizer as input to prepare the trianing data in a suitable format for model training. Tokenization is accomplished whereby each sentence within each conversation is converted into tokenized input and output pairs. In this case, the input to the model constitutes one sentence, and the target output corresponds to the following sentence. 

3. **Model and Tokenizer Inialization:** A function is deployed to first initliaze a tokenizer and the model. ***GPT2Tokenizer*** and ***GPT2LMHeadModel*** classes are  responsible for loading a pre-trained DialoGPT model and its associated tokenizer. They are loaded based on the model's name, specified as ***microsoft/DialoGPT-small*** (which refers to the small variant of the DialoGPT model trained by Microsoft) in this case. If a GPU is accessible, the function further optimizes training speed by transferring the model to the GPU.

4.  **Preparation of Training :** The training configuration in the ***TrainingArguments*** class encompasses crucial parameters such as the model's output directory and checkpoints, number of training epochs (represents full passes through the data), batch size (indicates the number of examples processed concurrently), save_steps (controls how often model checkpoints are saved), save_total_limit (limits the total number of saved checkpoints), and logging_steps (dictates the frequency of progress logging during training). 

5. **Model Training :**  The ***Trainer*** class handles the process of training. This class is provided with the model, the training arguments, the data collator (which manages the batching of examples and prepares them for model input), and the dataset. Subsequently, the Trainer oversees the model's training and stores it in the designated output directory. The ***train_model*** function is called in the main function takes the model and the training data as input to complete the model traning process.  

## Testing
A python script is used for assessing the quality of the model's responses through various metrics and also measures the time it requires for the model to produce these responses. The steps of evaluating similarity scores, accuracy scores, and response time involve:

1. **Import and Setup :** The essential libraries and modules are first imported. The ***AutoTokenizer*** and ***AutoModelWithLMHead*** are employed to load the previously trained DialoGPT-small model and its corresponding tokenizer. Additionally, a ***Rouge*** object is initilized to calculate the ROUGE score. 

2. **Similarity Scores Calculation :** A function is deployed to evaluate the similarity between two texts by comparing the overlap of words orn-grams between the generated responses and the ground truth. It employs various metrics, including Jaccard similarity, BLEU score, and ROUGE score. 
    - **Jaccard similarity** quantifies the commonality between two sets of words by calculating the ratio of their shared elements to their total elements

    - **BLEU score** measures the correspondence between a candidate text and reference texts

    - **ROUGE score** assesses n-gram overlap between system and reference translations

3. **Chatbot Response Generation :** The ***query*** function processes input text (questions) and generates responses utilizing the model and tokenizer.

4. **Test Data Loading :** The script reads a JSON file ***rephraseQ&A.json*** containing a test data which consits of a collection of question-answer pairs structured in a format similar to ***modifiedQ&A.json***. 

5. **Evaluation Loop :** The primary section of the script runs an evaluation loop. In the run, it iterates through all the question-answer pairs in the test data. For each question, it queries the model, measures response time, and calculates similarity scores by comparing the generated response with the ground truth. This loop accumulates response texts, response times, and similarity scores.

6. **Accuracy calculation :** Under the same evalaution loop, the accuracy scores focussing on precision, recall, and F1-score are calculated which consider the presence of false positives or false negatives in the generated responses. These scores are printed out after the loop. 
    - **Precision** represents the proportion of total relevant generated responses, known as true positive, among the retrieved answers such as true positive and false positive 

    - **Recall** represents the proportion of total relevant generated responses or true positives among all the samples which include true positives and false negatives

    - **F1 score**  represents the harmonic mean of precision and recall

7. **Results Compilation :** After the loop,the results are compiled into a pandas DataFrame and saves to a CSV file. The results include the original question, the model's response, the expected response, the Jaccard similarity, BLEU score, ROUGE-1, ROUGE-2, ROUGE-L scores, and the response time for each question.




# GPT model 
A GPT-2 model that is downloaded from [HuggingFace](https://huggingface.co/gpt2) and trained based on the official document of Academic Regulation from the university. To complete the processes of model training and evaluation, open the ipynb file on* Google Colab* or *Jupyter Notebook* and run the file accordingly.  

## Training
The training folder containing the files for training the model. The steps of completing model training involve: 

1. **Training data preparation:** A ***read_documents_from_directory*** function is employed for combining a text file, which stores the post-processed data, that is parsed and returned from ***read_text*** function. This data structure is loaded into memory as a Python list of question-answer pairs. The ***modifiedQ&A.txt*** is formatted precisely in this manner.

3. **Model and Tokenizer Inialization:** A ***train_chatbotfunction*** is deployed to first initliaze a tokenizer and the model after loading the training data. ***GPT2Tokenizer*** and ***GPT2LMHeadModel*** classes are responsible for loading a pre-trained GPT-2 model and its associated tokenizer. The tokenizer performs tokenization on the data to prepare training dataset while a helper object is initialized with a tokenizer and set to handle masked language modeling (MLM) tasks.

4.  **Preparation of Training :** The training configuration in the ***TrainingArguments*** class encompasses crucial parameters such as the model's output directory and checkpoints, allow of overwriting the output directory if it already exists, batch size (indicates the number of examples processed concurrently), number of training epochs (represents full passes through the data), save_steps (controls how often model checkpoints are saved), save_total_limit (limits the total number of saved checkpoints), logging_dir (specifies directory where training logs will be stored), logging_steps (dictates the frequency of progress logging during training) and logging_first_step (ensures that logging begins with the first training step for comprehensive progress tracking).

5. **Model Training :**  The ***Trainer*** class handles the process of training. This class is provided with the model, the training arguments, the data collator (which manages the batching of examples and prepares them for model input), and the dataset. Subsequently, the Trainer oversees the model's training and stores it in the designated output directory. The ***train_chatbot*** function is called in the main function takes the model and the training data as input to complete the model traning process.  

## Testing
The same script is used for assessing the quality of the model's responses through various metrics and also measures the time it requires for the model to produce these responses. The testing folder containing the files for testing the model. The steps of evaluating similarity scores, accuracy scores, and response time involve:

1. **Chatbot Response Generation :** The ***generate_response*** function processes input text (questions) and returns the generated responses utilizing the previously trained model and tokenizer.
 
2. **Similarity Scores Calculation :** The ***calculate_similarity_score*** function is deployed to evaluate the similarity between two texts by comparing the overlap of words orn-grams between the generated responses and the ground truth. It employs various metrics, including Jaccard similarity, BLEU score, and ROUGE score. 

3. **Evaluation Loop :** The primary section of the script runs an evaluation loop. In the run, it iterates through all the question-answer pairs in the test data loaded from ***rephrasedQ&A.txt*** in the testing folder. For each question, it queries the model, measures response time, and calculates similarity scores by comparing the generated response with the ground truth. This loop accumulates response texts, response times, and similarity scores.

4. **Accuracy calculation :** Under the same evalaution loop, the accuracy scores focussing on precision, recall, and F1-score are calculated which consider the presence of false positives or false negatives in the generated responses. These scores are printed out after the loop. 

5. **Results Compilation :** The results are compiled into a pandas DataFrame and saves to a CSV file. The results include the original question, the model's response, the expected response, the Jaccard similarity, BLEU score, ROUGE-1, ROUGE-2, ROUGE-L scores, and the response time for each question.

