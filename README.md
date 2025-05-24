# BioLaySumm 2025
## Lay Summarization of Biomedical Research Articles and Radiology Reports @ BioNLP Workshop, ACL 2025

This is the repo for Team MIRAGE's submission for BiolaySumm 2025!

## Download the shared task dataset: [PLOS](https://drive.google.com/drive/u/0/folders/1__OvlFElNPmu5V58aqySSXwX-72_DhEo), [eLife](https://drive.google.com/drive/u/0/folders/1__OvlFElNPmu5V58aqySSXwX-72_DhEo)
## Repo Structure
Code in this repo will likely not be runnable on your own machine unless you've got a real BEEFY GPU (and even then it'll require some modification to work with your particular system). Code in `preprocessing_script` comes from CoLab notebooks and are meant to go through the datasets and extract the top 40 sentences based on different methods of evaluation. Everything else is the actual summarization code meant to be run on Hyak.

### Preprocessing
Our preprocessing mostly uses embeddings from BioBERT to make judgements about what is salient. Our preprocessing techniques are as follows:
1. Control, just take the first 4096 tokens from the article. This doesn't have a preprocessing script.
2. Comparing every sentence to the embedding for the title of the article.
3. Comparing every sentence to the embedding for the title and keywords of the article.
4. SVD Topic Modeling
5. Turn the entire article into an embedding and compare every sentence to that mean embedding.
6. Split the article by sections and compare sentences from sections to the mean embedding for the article. The article will have its title and keywords appended.
7. Same as 6 except title and keywords are appended to the final set of extracted sentences.

Each of the scripts are found in the `preprocessing_script` and can be reimported into CoLab for use directly. You will need a GPU for this, and likely one stronger than the free tier GPUs.

### Summarization
The actual summarization code can be found in `summarize.py`. It uses Llama-3-8B-Instruct to do inference, but you'll need a real good GPU to do this or it'll take forever. The code was designed to run on Hyak and the SLURM file to submit the code is also in the repo. Submit using the SLURM file and you should be able to generate summaries.
