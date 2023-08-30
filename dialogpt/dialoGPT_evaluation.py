import timeit
import datetime
import pandas as pd
import json
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from collections import Counter

# Load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', padding_side='left')

# Fine-tuned model and tokenizer's directory: https://drive.google.com/drive/folders/10GTL7iIQQXHElqIhfuGVf1s9DKBQHOwB?usp=sharing
# Replace the model with the path of the actual one that has fine-tuned
model = AutoModelWithLMHead.from_pretrained("dialoGPT-checkpoint-19200")
rouge = Rouge()

# A function to calculate similarity scores (jaccard similarity, BLEU score, ROUGE scores)
def calculate_similarity_score(response, correct_answer):
    jaccard_similarity = len(set(response).intersection(correct_answer)) / len(set(response).union(correct_answer))
    bleu_score = sentence_bleu([correct_answer.split()], response.split())
    rouge_scores = rouge.get_scores(response, correct_answer)
    return jaccard_similarity, bleu_score, rouge_scores

# A function to load the query to the chatbot and returned the genrated response from the chatbot
def query(payload):
    
    bot_input_ids = tokenizer.encode(payload["inputs"]["text"] + tokenizer.eos_token, return_tensors='pt')

    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=4,       
        do_sample=False, 
        top_k=20,  # Adjust the value of top_k
        top_p=0.2,  # Adjust the value of top_p
        temperature=0.3  # Adjust the value of temperature
    )
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"generated_text": output}

# Load testing data set
with open("rephrasedQ&A.json", "r") as f:
    test_data = json.load(f)

questions = [item[0] for item in test_data]
ground_truth = [item[1] for item in test_data]

# Perform n time of runs
num_runs = 1

# Calculate the three similarity scores, precision, recall and f1-scores for each run
for run in range(num_runs):
    bot_responses = []
    response_times = []
    jaccard_similarities = []
    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Evaluate the similarities between the generated response from the chatbot and validate response
    for question, correct_answer in zip(questions, ground_truth):
        start_time = timeit.default_timer() # start time
        response = query({ # generate a response based on the query ask
            "inputs": {
                "past_user_inputs": [],
                "generated_responses": [],
                "text": question,
            }
        })
        end_time = timeit.default_timer() # start time
        # Measure the duration of generating a response
        response_times.append(end_time - start_time) 
        bot_responses.append(response["generated_text"]) # Append the response to a list

        # Call the function to calculate respective similarity scores
        jaccard_similarity, bleu_score, rouge_scores = calculate_similarity_score(response["generated_text"], correct_answer)
        jaccard_similarities.append(jaccard_similarity)
        bleu_scores.append(bleu_score)
        rouge_1_scores.append(rouge_scores[0]['rouge-1']['f'])
        rouge_2_scores.append(rouge_scores[0]['rouge-2']['f'])
        rouge_l_scores.append(rouge_scores[0]['rouge-l']['f'])

    # Define a list of:
    # num_c : to store the number of tokens that shared between gold and predicted answers, 
    # num_p : to store the number of predicted tokens, 
    # num_p : to store the number of ground truth tokens
    num_c = []
    num_p = []
    num_g = []

    # Calculate the number of common tokens that shared between ground truth and predicted answers,
    # the number of predicted tokens, and the number of ground truth tokens. 
    for a in range(len(bot_responses)):
        common = Counter(ground_truth[a].split()) & Counter(bot_responses[a].split())  # tokens shared between ground truth and predicted tokens
        num_common = sum(common.values())

        num_pred = len(str(bot_responses[a]).split())  # the number of predicted tokens
        num_gold = len(str(ground_truth[a]).split())  # the number of ground truth tokens

        num_c.append(num_common)
        num_c_sum = sum(num_c)

        num_p.append(num_pred)
        num_p_sum = sum(num_p)

        num_g.append(num_gold)
        num_g_sum = sum(num_g)

    # Calculate the precision, recall and f1-score
    precision = 1.0 * num_c_sum / (num_c_sum + (num_p_sum - num_c_sum))  
    recall = 1.0 * num_c_sum / (num_c_sum + (num_g_sum - num_c_sum)) 
    f1_score = (2 * precision * recall) / (precision + recall)

    # Create a DataFrame with the results of simililarity scores. 
    results = pd.DataFrame({
        "Question": questions,
        "Bot Response": bot_responses,
        "Expected Response": ground_truth,
        "Jaccard Similarity": jaccard_similarities,
        "BLEU Score": bleu_scores,
        "ROUGE-1": rouge_1_scores,
        "ROUGE-2": rouge_2_scores,
        "ROUGE-L": rouge_l_scores,
        "Response Time": response_times
    })

    # Save the results to a CSV file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results.to_csv(f'2dialogpt_accuracy_results_{timestamp}_run_{run + 1}.csv', index=False)

# print(results.iloc[0])   

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')
