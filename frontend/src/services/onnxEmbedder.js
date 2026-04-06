import * as ort from "onnxruntime-web";

let session = null;
let vocab = null;

//  Hosted on Hugging Face — no local file needed
const MODEL_URL =
  "https://huggingface.co/abhayjain09/banking-chatbot-onnx/resolve/main/model_quantized.onnx";

//  vocab still served from /public/models/vocab.txt (small file, fine to keep local)
const VOCAB_URL = "/models/vocab.txt";

export async function loadModel() {
   if (session) return session;
  console.log("⏳ Loading ONNX model from Hugging Face...");

  //  Point to node_modules dist — Vite will serve these correctly
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = 1;

  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  console.log(" ONNX model loaded!");
  return session;
}

async function loadVocab() {
  if (vocab) return vocab;
  const res = await fetch(VOCAB_URL);
  const text = await res.text();
  vocab = {};
  text.split("\n").forEach((word, idx) => {
    vocab[word.trim()] = idx;
  });
  return vocab;
}

function tokenize(text, vocabMap, maxLen = 128) {
  const cls = vocabMap["[CLS]"] ?? 101;
  const sep = vocabMap["[SEP]"] ?? 102;
  const unk = vocabMap["[UNK]"] ?? 100;
  const pad = vocabMap["[PAD]"] ?? 0;

  const words = text.toLowerCase().trim().split(/\s+/);
  const tokenIds = [cls];

  for (const word of words) {
    if (vocabMap[word] !== undefined) {
      tokenIds.push(vocabMap[word]);
    } else {
      for (const ch of word) {
        tokenIds.push(vocabMap[ch] ?? unk);
      }
    }
    if (tokenIds.length >= maxLen - 1) break;
  }
  tokenIds.push(sep);

  const inputIds      = new Array(maxLen).fill(pad);
  const attentionMask = new Array(maxLen).fill(0);
  const tokenTypeIds  = new Array(maxLen).fill(0);

  tokenIds.slice(0, maxLen).forEach((id, i) => {
    inputIds[i] = id;
    attentionMask[i] = 1;
  });

  return { inputIds, attentionMask, tokenTypeIds, seqLen: maxLen };
}

function meanPool(hiddenState, attentionMask, seqLen, hiddenDim) {
  const result = new Float32Array(hiddenDim).fill(0);
  let count = 0;
  for (let i = 0; i < seqLen; i++) {
    if (attentionMask[i] === 1) {
      for (let j = 0; j < hiddenDim; j++) {
        result[j] += hiddenState[i * hiddenDim + j];
      }
      count++;
    }
  }
  let norm = 0;
  for (let j = 0; j < hiddenDim; j++) {
    result[j] /= count;
    norm += result[j] * result[j];
  }
  norm = Math.sqrt(norm);
  for (let j = 0; j < hiddenDim; j++) result[j] /= norm;
  return Array.from(result);
}

export async function getEmbedding(text) {
  const [sess, vocabMap] = await Promise.all([loadModel(), loadVocab()]);
  const { inputIds, attentionMask, tokenTypeIds, seqLen } = tokenize(text, vocabMap);

  const feeds = {
    input_ids:      new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)),      [1, seqLen]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, seqLen]),
    token_type_ids: new ort.Tensor("int64", BigInt64Array.from(tokenTypeIds.map(BigInt)),  [1, seqLen]),
  };

  const output = await sess.run(feeds);
  return meanPool(output["last_hidden_state"].data, attentionMask, seqLen, 384);
}