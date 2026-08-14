from datasets import load_dataset, load_from_disk
from collections import Counter
import tiktoken
from nltk import ngrams
from tqdm import tqdm
import datasets

def has_overlap(sample_1, sample_2):
    """Check if there is any N-gram overlap between the long string and a given string."""
    return not set(sample_1).isdisjoint(set(sample_2))

from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_overlap_percentage(samples_1, samples_2):
    def check_sample(sample):
        for long_sample in samples_2:
            if has_overlap(sample, long_sample["ngram"]):
                return 1
        return 0

    count = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_sample, sample) for sample in samples_1]
        for future in tqdm(as_completed(futures), total=len(futures)):
            count += future.result()

    return count / len(samples_1) * 100

def load_odex_data(n=10):
    def map_ngram(sample):
        return {"ngram": set([" ".join(ngram) for ngram in ngrams(sample["intent"].split(), n)])}
    dataset = load_dataset("neulab/odex", "en", split="test")
    dataset = dataset.map(map_ngram, num_proc=16, batch_size=16, remove_columns=dataset.column_names)
    return dataset

def load_stackoverflow(n=10):
    def map_ngram(sample):
        return {"ngram": set([" ".join(ngram) for ngram in ngrams(sample["question"].split(), n)])}
    dataset = load_dataset("bigcode/stack-exchange-preferences-20230914-clean-anonymization", split="train")
    dataset = dataset.map(map_ngram, num_proc=16, batch_size=16, remove_columns=dataset.column_names)
    dataset.push_to_hub(f"stackoverflow_ngram_{n}")
    return dataset


def load_starcoderdata(n=10):
    def map_ngram(sample):
        return {"ngram": set([" ".join(ngram) for ngram in ngrams(sample["content"].split(), n)])}
    dataset = load_dataset("bigcode/starcoderdata", data_dir="python", split="train")
    dataset = dataset.map(map_ngram, num_proc=16, batch_size=16, remove_columns=dataset.column_names)
    dataset.push_to_hub(f"starcoderdata_ngram_{n}")
    return dataset

def load_bigcodebench(n=10):
    def map_ngram(sample):
        return {"ngram": set([" ".join(ngram) for ngram in ngrams(sample["instruct_prompt"].split("```")[0].split(), n)])}
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.0_hf")
    dataset = dataset.map(map_ngram, num_proc=16, batch_size=16, remove_columns=dataset.column_names)
    dataset.push_to_hub(f"bigcodebench_ngram_{n}")
    return dataset


if __name__ == "__main__":
    n_gram_size = 10
    N_SHARDS = 50
    user_name = "terryyz"
    bigcodebench = load_dataset(f"{user_name}/bigcodebench_ngram_{n_gram_size}", split="train")

    dataset_name = "starcoderdata"
    print(dataset_name, n_gram_size)
    indices = []
    for i in tqdm(range(N_SHARDS)):
        ds = load_dataset(f"{user_name}/{dataset_name}_ngram_{n_gram_size}_overlap_{i}", split="train")
        overlap_indices = [idx for idx, example in enumerate(ds) if example["overlap"]]
        indices.extend(overlap_indices)
    with open(f"{dataset_name}_ngram_{n_gram_size}_overlap.txt", "w") as f:
        f.write(f"{len(set(indices))/1140*100:.2f}%")