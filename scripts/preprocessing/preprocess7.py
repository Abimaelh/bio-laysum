
# Step 1: Mount Google Drive
from google.colab import drive
import os
import pandas as pd
import re
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Mount drive
drive.mount('/content/drive')

# Step 2: Prepare paths
json_path = '/content/drive/MyDrive/biolaysumm/eLife_train.json'

# Step 3: Load Dataset
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found at {json_path}")

df = pd.read_json(json_path)
print("Dataset loaded!")

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
        while re.search(r"\([^()]*\)", text):
            text = re.sub(r"\([^()]*\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return ""

if "article" not in df.columns:
    raise KeyError("'article' column not found in dataset.")

# Apply helper function first
df["article"] = df["article"].apply(remove_parentheses_and_correct_spacing)
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

# Step 8: Section-aware extractor WITHOUT prepending title/keywords yet
def extract_target_sections(article_text, section_headings, target_sections=["abstract", "introduction", "results", "discussion"]):
    paragraphs = [p.strip() for p in article_text.split('
') if p.strip()]
    
    n_paragraphs = len(paragraphs)
    n_sections = len(section_headings)

    if n_sections == 0 or n_paragraphs == 0:
        return "
".join(paragraphs)

    chunk_size = max(1, n_paragraphs // n_sections)
    section_chunks = []

    for i, heading in enumerate(section_headings):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_sections - 1 else n_paragraphs
        chunk_text = "
".join(paragraphs[start:end])
        section_chunks.append((heading.strip().lower(), chunk_text))

    filtered_chunks = [text for name, text in section_chunks if name in target_sections]
    return "

".join(filtered_chunks)

# Step 9: Top-K sentence selector
def get_top_k_sentences(text, k=40):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return ""

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    avg_embedding = sentence_embeddings.mean(dim=0, keepdim=True)
    similarities = util.pytorch_cos_sim(avg_embedding, sentence_embeddings)[0]

    top_k = min(k, len(sentences))
    top_indices = similarities.argsort(descending=True)[:top_k]
    return " ".join([sentences[i] for i in sorted(top_indices.tolist())])

# Step 10: Preprocess each article
processed_articles = []
for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    structured_text = extract_target_sections(
        article_text=row["article"],
        section_headings=row["section_headings"]
    )
    top_sentences = get_top_k_sentences(structured_text)

    # After picking top sentences, now prepend Title and Keywords
    header = f"Title: {row['title'].strip()}
Keywords: {', '.join(row['keywords']).strip() if isinstance(row['keywords'], list) else row['keywords'].strip()}"
    final_text = header + "

" + top_sentences

    processed_articles.append(final_text)

# Step 11: Create final output
output_df = pd.DataFrame({
    "preprocessed_article": processed_articles,
    "gold_summary": sampled_df["summary"]
})

# Step 12: Save Output
save_path = "/content/drive/MyDrive/BioLaySum/preprocessed_article/eLife_train_sampled_top40in4sections_titlekeywords.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
output_df.to_json(save_path, orient="records", indent=2, force_ascii=False)

print(f"Preprocessed sampled dataset saved to: {save_path}")
