
# Step 1: Mount Google Drive
from google.colab import drive
import os
import re
import spacy
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

# Step 1: Authenticate with Hugging Face
login(token= "Your token here")

# Step 3: Load NLP tools
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Step 4: Helper to remove parentheses and fix spacing
def remove_parentheses_and_correct_spacing(text):
    if isinstance(text, str):
        while re.search(r"\([^()]*\)", text):
            text = re.sub(r"\([^()]*\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return ""

# Step 5: Main preprocessing function
def preprocess_article(article_text, section_headings, title, keywords,
                       target_sections=["abstract", "introduction", "results", "discussion"],
                       top_k=40):
    # Prepend Title and Keywords
    header = f"Title: {title.strip()}
Keywords: {', '.join(keywords).strip() if isinstance(keywords, list) else keywords.strip()}"
    full_text = header + "

" + article_text

    # Chunk text into sections
    paragraphs = [p.strip() for p in full_text.split('
') if p.strip()]
    n_paragraphs = len(paragraphs)
    n_sections = len(section_headings)

    if n_sections == 0 or n_paragraphs == 0:
        selected_body = full_text
    else:
        chunk_size = max(1, n_paragraphs // n_sections)
        section_chunks = []
        for i, heading in enumerate(section_headings):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_sections - 1 else n_paragraphs
            chunk_text = "
".join(paragraphs[start:end])
            section_chunks.append((heading.strip().lower(), chunk_text))
        selected_body = "

".join(text for name, text in section_chunks if name in target_sections)

    # Top-K representative sentences
    doc = nlp(selected_body)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        selected_summary = selected_body
    else:
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        avg_embedding = sentence_embeddings.mean(dim=0, keepdim=True)
        similarities = util.pytorch_cos_sim(avg_embedding, sentence_embeddings)[0]
        top_k = min(top_k, len(sentences))
        top_indices = similarities.argsort(descending=True)[:top_k]
        selected_summary = " ".join([sentences[i] for i in sorted(top_indices.tolist())])

    return selected_summary

# Step 6: Load dataset
df = pd.read_json("/content/drive/MyDrive/biolaysumm/eLife_train.json")

# Step 7: Flatten if needed
if isinstance(df.iloc[0], dict):
    df = pd.DataFrame(df.tolist())

# Step 8: Clean parentheses and spacing
df["article"] = df["article"].apply(remove_parentheses_and_correct_spacing)

# Step 9: Sample 200 articles
df = df.sample(n=200, random_state=42).reset_index(drop=True)

# Step 10: Check required fields
required = ["article", "summary", "section_headings", "title", "keywords"]
assert all(col in df.columns for col in required), f"Missing required columns: {required}"

# Step 11: Process articles
processed_articles = []
for _, row in df.iterrows():
    final_summary = preprocess_article(
        article_text=row["article"],
        section_headings=row["section_headings"],
        title=row["title"],
        keywords=row["keywords"]
    )
    processed_articles.append(final_summary)

# Step 12: Save output
output_df = pd.DataFrame({
    "preprocessed_article": processed_articles,
    "gold_summary": df["summary"]
})

output_path = "/content/drive/MyDrive/BioLaySum/preprocessed_article/eLife_train_sampled_titlekeywords_top40in4sections.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_json(output_path, orient="records", indent=2, force_ascii=False)

print(f"Preprocessed and saved to: {output_path}")

