if __name__ == "__main__":

    input_file = "D:/GitRepos/GitRepos/NERRE/scierc_aeco/data/topic_12_batch_0_inference_raw_short_prompt.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)