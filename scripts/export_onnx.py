# scripts/export_onnx.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_DIR = Path("../frontend/public/models")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = EXPORT_DIR / "model.onnx"

print("⏳ Loading model...")
st_model = SentenceTransformer(MODEL_NAME)
transformer = st_model[0].auto_model  # the underlying BERT model
tokenizer = st_model[0].tokenizer
transformer.eval()

print("⏳ Creating dummy input...")
dummy = tokenizer(
    "How to add guarantor in loan",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
)

input_ids = dummy["input_ids"]
attention_mask = dummy["attention_mask"]
token_type_ids = dummy.get("token_type_ids", torch.zeros_like(input_ids))

print("⏳ Exporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        transformer,
        (input_ids, attention_mask, token_type_ids),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
print(f"✅ ONNX model saved → {ONNX_PATH}")
print(f"📦 Size: {size_mb:.1f} MB")

# Save tokenizer vocab files for browser use
tokenizer.save_pretrained(str(EXPORT_DIR))
print(f"✅ Tokenizer saved → {EXPORT_DIR}")

# Quick verification
import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX_PATH))
out = sess.run(None, {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy(),
    "token_type_ids": token_type_ids.numpy(),
})
print(f"✅ Verification passed! Output shape: {out[0].shape}")
print("🎉 Done! Model ready for browser deployment.")