# LLM inference for llama3-70b by zeroshot and fewshot on numclaim, trec, semeval

for dataset in "numclaim" "trec" "semeval"
do
    for prompt_type in "zeroshot" "fewshot"
    do
        python src/llm/llm_infer.py --dataset $dataset --prompt_type $prompt_type --llm "llama3-70b" --seed 42
    done
done

# LLM inference for llama3-70b by zeroshot on 20news

python src/llm/llm_infer.py --dataset "20news" --prompt_type "zeroshot" --llm "llama3-70b" --seed 42


# LLM inference for llama31-70b, llama31-405b, gpt4o, mixtral822 by zeroshot and fewshot on semeval

for llm in "llama31-70b" "llama31-405b" "gpt4o" "mixtral822"
do
    for prompt_type in "zeroshot" "fewshot"
    do
        python src/llm/llm_infer.py --dataset "semeval" --prompt_type $prompt_type --llm $llm --seed 42
    done
done
