from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
from typing import List

# MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1e-12
    return v / norm[:, np.newaxis]

class DefaultEmbeddingModel:
    def __init__(self,tokenizer_path: str, model_path: str):

        self.tokenizer_path = tokenizer_path
        self.model_path = model_path

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        self.model = ort.InferenceSession(self.model_path)

    def __call__(self, documents: List[str], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            encoded = [self.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array([np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64),
            }
            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
            embeddings = normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings)
         

# tokenizer_path="models/all-MiniLM-L6-v2/tokenizer.json"
# model_path="models/all-MiniLM-L6-v2/onnx/model_quantized.onnx"

# model = DefaultEmbeddingModel(tokenizer_path,model_path)
# texts = ["This is the first text", "This is the second text"]
# embeddings = model(texts)
# print(len(embeddings.tolist()[0]))



