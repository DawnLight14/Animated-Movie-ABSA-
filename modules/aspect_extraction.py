# modules/aspect_extraction.py
from transformers import pipeline as hf_pipeline

def extract_aspects(sentences, tokenizer, model, batch_size=32, device=-1):
    extractor = hf_pipeline("token-classification", model=model, tokenizer=tokenizer, device=device)

    all_aspects = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_aspects = []
        for sentence in batch:
            results = extractor(sentence)
            aspects = []
            for r in results:
                label = r.get("entity_group") or r.get("entity")
                if label in ["ASP", "B-ASP", "I-ASP", "B-ASPECT", "I-ASPECT"]:
                    aspects.append(r["word"])
            batch_aspects.append(aspects)
        all_aspects.extend(batch_aspects)

    return all_aspects
