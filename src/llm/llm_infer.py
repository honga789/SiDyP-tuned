import os,sys
import pandas as pd
from time import sleep, time
from datetime import date
import json
from openai import OpenAI
import argparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk import word_tokenize

from llm_utils import LLM_PROMPTS, LABEL_MAP


LLM_MAP = {
    "llama3-70b": "meta-llama/Llama-3-70b-chat-hf",
    "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama31-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "gpt4o": "gpt-4o",
    "mixtral822": "mistralai/Mixtral-8x22B-Instruct-v0.1"
}


def get_data(dataset, seed):
    data_list = []
    if dataset == "numclaim":
        for data_category in ["numclaim-train", "numclaim-test"]: 
            data_path = f"src/llm/datasets/{dataset}/{data_category}.xlsx"
            data_df = pd.read_excel(data_path)
            if data_category == "numclaim-train":
                train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=seed)
                data_list.append(train_df)
                data_list.append(valid_df)
            else:
                data_list.append(data_df)

    elif dataset in ["trec", "semeval"]:
        for data_category in ["train", "valid", "test"]:
            data_path = f"src/llm/datasets/{dataset}/{data_category}.json"
            data_json = json.load(open(data_path))
            data_list.append(data_json)
    
    elif dataset == "20news":
        for data_category in ["train", "test"]:
            if data_category == "train":
                # Fetch the train data from the dataset
                data = fetch_20newsgroups(subset='train', data_home=f"src/llm/datasets/{dataset}")

                # Split the training data into 80% training and 20% validation
                train_data, valid_data, train_labels, valid_labels = train_test_split(
                    data.data, data.target, test_size=0.2, random_state=seed)

                print(f"Train samples: {len(train_data)}")
                print(f"Validation samples: {len(valid_data)}")
                data_list.append([train_data, train_labels])
                data_list.append([valid_data, valid_labels])
            
            elif data_category == "test":
                data = fetch_20newsgroups(subset='test')
                print(f"Test samples: {len(data.data)}")
                data_list.append([data.data, data.target])
    else:
        raise NotImplemented
    
    return data_list 

def decode(x, dataset):
    try: 
        list_words = word_tokenize(x)
        label_word = list_words[0].lower()
        print(LABEL_MAP[dataset].keys())
        print(label_word)
        if label_word in LABEL_MAP[dataset].keys():
            return LABEL_MAP[dataset].get(label_word, -1)
        label_word = list_words[2].lower()
        print(label_word)
        return LABEL_MAP[dataset].get(label_word, -1)
    except Exception as e:
        print(e)
        return -1 
    

def parse_labels(dataset, data_path):
    df = pd.read_csv(data_path)
    df["noisy_label"] = df["text_output"].apply(lambda x: decode(x, dataset))
    df[["true_label", "original_sent", "noisy_label"]].to_csv(data_path, index=False)


def llm_inference(
        dataset,
        prompt_type,
        llm,
        seed
):
    # setup client
    if llm == "gpt-4o":
        client = OpenAI(api_key="your api key")
    else:
        client = OpenAI(api_key="your api key",
        base_url='https://api.together.xyz',
        )

    data_list = get_data(dataset, seed)

    category_dict = {0: "train", 1: "valid", 2: "test"}
    for idx, data in enumerate(data_list):
        if dataset == "numclaim":
            sentences = data['text'].to_list()
            labels = data['label'].to_numpy()

        elif dataset in ["trec", "semeval"]:
            sentences = [d["data"]["text"] for _, d in data.items()]
            labels = [d["label"] for _,d in data.items()]

        elif dataset == "20news":
            sentences, labels = data

        output_list = []
                
        for i in range(len(sentences)): 
            sen = sentences[i]
            message = LLM_PROMPTS[dataset][prompt_type] + sen
            prompt_json = [
                        {"role": "user", "content": message},
                ]
            try:
                chat_completion = client.chat.completions.create(
                        model=f"{LLM_MAP[llm]}",
                        messages=prompt_json,
                        temperature=0.0,
                        max_tokens=100
                )
            except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)

            answer = chat_completion.choices[0].message.content
            
            output_list.append([labels[i], sen, answer])

            print(i)

            if i == 3:
                break

        results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])
        save_path = f'datasets/llm/{prompt_type}/{llm}'
        os.makedirs(save_path, exist_ok=True)
        file_path = f'{save_path}/{dataset}_{category_dict[idx]}.csv'
        results.to_csv(file_path, index=False)

        parse_labels(dataset, file_path)

        

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="numclaim", type=str)
    parser.add_argument("--prompt_type", default="zeroshot", type=str)
    parser.add_argument("--llm", default="llama3-70b", type=str)
    args = parser.parse_args()

    llm_inference(args.dataset, args.prompt_type, args.llm, args.seed)

if __name__ == "__main__":
    main()