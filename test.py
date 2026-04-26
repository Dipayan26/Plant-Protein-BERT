from transformers import EsmForMaskedLM, EsmTokenizer
import torch

model = EsmForMaskedLM.from_pretrained("dipayan26/PlantPLM-8M")
tokenizer = EsmTokenizer.from_pretrained("dipayan26/PlantPLM-8M")

# --- Masked token prediction ---
sequence = "MSPQTETKASVGFKAGVKDYKLTYYTPEYETK"
inputs = tokenizer(sequence, return_tensors="pt")

# mask one position
inputs["input_ids"][0, 5] = tokenizer.mask_token_id

with torch.no_grad():
    logits = model(**inputs).logits

masked_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[0, 1]
top5 = logits[0, masked_pos].topk(5)
print(tokenizer.convert_ids_to_tokens(top5.indices.tolist()))

# --- Sequence embedding ([CLS] token) ---
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    hidden = model.esm(**inputs).last_hidden_state
cls_embedding = hidden[0, 0, :]   # shape: [320]
print("Embedding shape:", cls_embedding.shape)
