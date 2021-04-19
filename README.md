# Elasticsearch
**Overview**
---
This assignment is intended to help you get familiar with Elasticsearch (ES) while doing a little information retrieval “research” to compare alternative approaches to document indexing and querying. We will be using the TREC 2018 core corpus subset and five TREC topics with relevance judgments for evaluation. 
For background, a high level overview of ES functionality with several case studies is found at https://apiumhub.com/tech-blog-barcelona/elastic-search-advantages-books/.

In this assignment, we will provide example code for:
Populating and querying a corpus using ES
implementing NDCG (normalized discounted cumulative gain) evaluation metric
experimenting with “semantic” indexing and searching using fastText and BERT 
You will:
Index the corpus into ES with default standard analyzer and your customized one for the text fields.
Integrate ES into your Flask service for interactive search. Beside the traditional lexical search as we did before, your system should also allow the user to select the text representation to use for search.
Create a command line interface that runs a TREC query against an index and search options and shows the evaluation result using NDCG.
Evaluate the performance of 5 provided TREC queries using NDCG. For each query, you should produce a result table along with a brief analysis.

