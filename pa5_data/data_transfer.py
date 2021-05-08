import csv
import json
import os

import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DataTransfer:
    def __init__(self, old_jsonfile, new_jsonfile):
        # delete file if exist
        if os.path.isfile(new_jsonfile):
            os.remove(new_jsonfile)
        self.old_jsonfile = old_jsonfile
        self.new_jsonfile = new_jsonfile
        print("initializing summarizer...")
        self.summarizer = pipeline('summarization')
        print("initializing tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
        print("initialized")

    # default summarizer
    def default_sum(self, min_length, max_length, text):
        return self.summarizer(text[:1024], max_length=max_length, min_length=min_length)[0]['summary_text']

    # pre-trained summarizer
    def pre_trained_sum(self, text):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'])
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    # transformer
    def transform_json(self):
        with open(self.old_jsonfile, "r", encoding="utf-8") as old_json:
            with open(self.new_jsonfile, "w") as new_json:
                for i, line in enumerate(old_json):
                    new_doc = {}
                    doc = json.loads(line)
                    new_doc['title'] = doc['title']
                    print("   subsection...")
                    new_doc['default_text'], new_doc['trained_text'] = self.sub_summarization(doc['content_str'], 1024)
                    print("   done")
                    new_doc['doc_id'] = doc['doc_id']
                    new_doc['author'] = doc['author']
                    new_doc['annotation'] = doc['annotation']
                    new_doc['published_date'] = doc['published_date']
                    json.dump(new_doc, new_json)
                    print(i, "done")

    # summarization
    def sub_summarization(self, para, k):
        default_text = []
        trained_text = []
        for i in range(len(para) // k + 1):
            if i < len(para) // k:
                batch = para[i * 1024:(i + 1) * 1024]
                default_text.append([self.default_sum(2, 150, batch)])
                trained_text.append(self.pre_trained_sum(batch))
            else:
                batch = para[i * 1024:len(para)]
                default_text.append([self.default_sum(2, 150, batch)])
                trained_text.append(self.pre_trained_sum(batch))
        return default_text, trained_text


if __name__ == "__main__":
    dt = DataTransfer("subset_wapo_50k_sbert_ft_filtered.jl", "filtered_data.json")
    dt.transform_json()
