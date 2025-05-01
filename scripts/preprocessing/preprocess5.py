
# Step 1: Mount Google Drive
from google.colab import drive
import os
import pandas as pd
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import torch

# Mount drive
drive.mount('/content/drive')

# Step 2: Prepare paths
json_path = '/content/drive/MyDrive/biolaysumm/eLife_train.json'

# Step 3: Load Dataset
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found at {json_path}")

df = pd.read_json(json_path)
print("✅ Dataset loaded!")

# Step 4: Flatten if needed
if isinstance(df.iloc[0], dict):
    df = pd.DataFrame(df.tolist())

# Step 5: Helper functions
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text).strip()
    return ""

def remove_parentheses_and_correct_spacing(text):
    if isinstance(text, str):
        # Remove nested parentheses content completely
        while re.search(r"\([^()]*\)", text):
            text = re.sub(r"\([^()]*\)", "", text)
        text = re.sub(r"\s+", " ", text)  # fix spacing
        return text.strip()
    return ""

if "article" not in df.columns:
    raise KeyError("'article' column not found in dataset.")

# Apply helper function first
df["article"] = df["article"].apply(remove_parentheses_and_correct_spacing)
# Then apply basic cleaning
df["article"] = df["article"].apply(clean_text)

summary_field = "summary"
if summary_field not in df.columns:
    raise KeyError(f"'{summary_field}' column not found in dataset.")
df[summary_field] = df[summary_field].apply(clean_text)

# Step 6: Sample 200 instances
sampled_df = df.sample(n=200, random_state=42).reset_index(drop=True)

# Step 7: Download and Load NLP Models
!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Step 8: Select top 40 sentences by cosine similarity to article average
selected_passages = []

for article in sampled_df["article"]:
    doc = nlp(article)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    if len(sentences) == 0:
        selected_passages.append("")
        continue

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    article_embedding = sentence_embeddings.mean(dim=0, keepdim=True)
    similarities = util.pytorch_cos_sim(article_embedding, sentence_embeddings)[0]

    top_k = min(40, len(sentences))
    top_indices = similarities.argsort(descending=True)[:top_k]
    selected = " ".join([sentences[i] for i in sorted(top_indices.tolist())])
    selected_passages.append(selected)

# Step 9: Create Final Output
output_df = pd.DataFrame({
    "preprocessed_article": selected_passages,
    "gold_summary": sampled_df[summary_field]
})

# Step 10: Save Output
save_path = "/content/drive/MyDrive/BioLaySum/preprocessed_article/eLife_train_sampled_top40.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
output_df.to_json(save_path, orient="records", indent=2, force_ascii=False)

print(f"Preprocessed sampled dataset saved to: {save_path}")

