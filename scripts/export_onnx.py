"""
ONNX EXPORT SCRIPT: Convert MiniLM Embedding Model for Browser Inference
File: scripts/export_onnx.py

Converts sentence-transformers/all-MiniLM-L6-v2 to ONNX format
so it can run client-side in the browser using onnxruntime-web.

STEPS:
  1. Export to ONNX (full float32)
  2. Validate the exported model
  3. Optionally convert to FP16 to halve file size (~45MB -> ~23MB)
  4. Test inference against original PyTorch model

USAGE:
  cd backend
  python ../scripts/export_onnx.py

OUTPUT:
  models/minilm_embedding.onnx       (float32, ~45MB)
  models/minilm_embedding_fp16.onnx  (float16, ~23MB)  <- deploy this to browser

REQUIREMENTS:
  pip install torch transformers onnx onnxruntime onnxconverter-common
"""

import torch
import numpy as np
import os
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("models")
ONNX_PATH = OUTPUT_DIR / "minilm_embedding.onnx"
ONNX_FP16_PATH = OUTPUT_DIR / "minilm_embedding_fp16.onnx"
MAX_SEQ_LEN = 128


def export_to_onnx():
    """Step 1 & 2: Export HuggingFace model to ONNX and validate."""
    from transformers import AutoTokenizer, AutoModel

    print(f"\n[1/4] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("[2/4] Creating dummy inputs...")
    dummy_text = "What is the minimum balance for a savings account?"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Exporting to ONNX -> {ONNX_PATH}")
    torch.onnx.export(
        model,
        args=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        f=str(ONNX_PATH),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids":        {0: "batch", 1: "seq_len"},
            "attention_mask":   {0: "batch", 1: "seq_len"},
            "token_type_ids":   {0: "batch", 1: "seq_len"},
            "last_hidden_state":{0: "batch", 1: "seq_len"},
            "pooler_output":    {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"    Saved: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")

    print("[4/4] Validating ONNX model...")
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("    ONNX model is valid!")

    return tokenizer, model, inputs


def convert_to_fp16():
    """Step 3: Convert float32 ONNX model to float16 (halves file size)."""
    print(f"\nConverting to FP16 -> {ONNX_FP16_PATH}")
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(ONNX_PATH))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(ONNX_FP16_PATH))
    print(f"    Saved: {ONNX_FP16_PATH} ({ONNX_FP16_PATH.stat().st_size / 1e6:.1f} MB)")


def mean_pooling(model_output, attention_mask):
    """Same pooling strategy as sentence-transformers."""
    token_embeddings = model_output[0]  # last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).float().expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def validate_onnx_vs_pytorch(tokenizer, pt_model, inputs):
    """Step 4: Compare ONNX output vs original PyTorch model output."""
    import onnxruntime as ort
    import torch.nn.functional as F

    print("\nValidating ONNX output against PyTorch output...")

    # PyTorch output
    with torch.no_grad():
        pt_out = pt_model(**inputs)
    pt_embedding = mean_pooling(pt_out, inputs["attention_mask"])
    pt_embedding = F.normalize(pt_embedding, p=2, dim=1).numpy()

    # ONNX output
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids":       inputs["input_ids"].numpy(),
        "attention_mask":  inputs["attention_mask"].numpy(),
        "token_type_ids":  inputs["token_type_ids"].numpy(),
    }
    ort_out = sess.run(None, ort_inputs)
    ort_hidden = torch.tensor(ort_out[0])
    ort_embedding = mean_pooling(
        [ort_hidden],
        inputs["attention_mask"]
    )
    ort_embedding = F.normalize(ort_embedding, p=2, dim=1).numpy()

    max_diff = np.max(np.abs(pt_embedding - ort_embedding))
    print(f"    Max difference between PT and ONNX embeddings: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("    PASS: Embeddings match within tolerance.")
    else:
        print("    WARNING: Embeddings differ more than expected. Check export settings.")


def print_browser_integration():
    """Print the JavaScript snippet HR's frontend dev can use."""
    print("""
=============================================================
  BROWSER INTEGRATION SNIPPET (for HR's frontend dev)
=============================================================

// 1. Install: npm install onnxruntime-web @xenova/transformers

// 2. Import and load the model once on page load:
import * as ort from 'onnxruntime-web';

let session;
async function loadEmbeddingModel() {
  session = await ort.InferenceSession.create(
    '/models/minilm_embedding_fp16.onnx',
    { executionProviders: ['wasm'] }
  );
  console.log('Embedding model loaded');
}

// 3. For tokenization in the browser, use @xenova/transformers:
import { AutoTokenizer } from '@xenova/transformers';
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');

// 4. Run inference:
async function getEmbedding(text) {
  const encoded = await tokenizer(text, { padding: true, truncation: true, max_length: 128 });
  const inputIds    = new ort.Tensor('int64', BigInt64Array.from(encoded.input_ids.data.map(BigInt)), encoded.input_ids.dims);
  const attMask     = new ort.Tensor('int64', BigInt64Array.from(encoded.attention_mask.data.map(BigInt)), encoded.attention_mask.dims);
  const tokenTypeIds= new ort.Tensor('int64', BigInt64Array.from(encoded.token_type_ids.data.map(BigInt)), encoded.token_type_ids.dims);

  const output = await session.run({ input_ids: inputIds, attention_mask: attMask, token_type_ids: tokenTypeIds });
  return output.last_hidden_state.data;  // raw embeddings -> apply mean pooling
}

// NOTE: For the full chatbot flow, it is easier to call the deployed API
// (POST /api/v1/chat with X-API-Key header) instead of running ONNX in-browser.
// Use in-browser ONNX only for offline/privacy-first mode.
=============================================================
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  ONNX EXPORT: all-MiniLM-L6-v2")
    print("=" * 60)

    tokenizer, model, inputs = export_to_onnx()
    convert_to_fp16()
    validate_onnx_vs_pytorch(tokenizer, model, inputs)
    print_browser_integration()

    print("\nDone! Upload minilm_embedding_fp16.onnx to your frontend /public/models/ folder.")
