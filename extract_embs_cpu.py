"""
CPU-compatible version of extract_embs.py for small-scale testing.
Extracts static word embeddings from contextualized models.
"""
import torch
import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def identify_word_indices(batch_ids, token, token_ids, tokenizer, init_idx=0, subword=False):
    """Find the token positions where a word/token appears in each sentence."""
    token_len = len(token_ids)
    if token_len == 0:
        return np.array([]), []

    token_col_idx = []
    valid_sent_id = []

    for sid, sent in enumerate(batch_ids):
        if len(sent) >= 500:
            continue

        sent_tmp = sent.copy()
        flag = False

        for i in range(init_idx, len(sent_tmp)):
            if i + len(token_ids) <= len(sent_tmp):
                if subword:
                    token_in_sent = [sent_tmp[i]]
                    decoded = "".join(tokenizer.convert_ids_to_tokens(token_in_sent)).replace("▁", "").strip()
                    if decoded == token:
                        flag = True
                else:
                    if all([sent_tmp[i + j] == token_ids[j] for j in range(len(token_ids))]):
                        flag = True

                if flag:
                    token_idx = [i + j for j in range(len(token_ids))]
                    token_col_idx.append(token_idx)
                    valid_sent_id.append(sid)
                    break

    token_col_idx = np.array(token_col_idx) if token_col_idx else np.array([]).reshape(0, 0)
    return token_col_idx, valid_sent_id


def encode_batch_cpu(tokenizer, model, sent_list, col_idx, device, batch_size=8):
    """
    Encode sentences in batches on CPU with smaller batch sizes.
    """
    all_token_states = []

    for i in range(0, len(sent_list), batch_size):
        batch_sents = sent_list[i:i+batch_size]
        batch_col_idx = col_idx[i:i+batch_size]

        batch = tokenizer(batch_sents, padding=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            hidden_states = outputs["last_hidden_state"]

        # Extract embeddings at specific positions
        for j, (hs, cidx) in enumerate(zip(hidden_states, batch_col_idx)):
            if len(cidx) > 0:
                token_emb = hs[cidx].mean(dim=0)  # Average across subword tokens
                all_token_states.append(token_emb.cpu().numpy())

    if all_token_states:
        return np.stack(all_token_states)
    return np.array([])


def main():
    parser = argparse.ArgumentParser(description='Extract static word embeddings (CPU version)')
    parser.add_argument('-model', required=True, help='HuggingFace model name')
    parser.add_argument('-output_folder', required=True, help='Output folder')
    parser.add_argument('-word2sent', required=True, help='Pickle file with word->sentences mapping')
    parser.add_argument('-nsent', default=15, type=int, help='Max sentences per word')
    parser.add_argument('-subword', action='store_true', help='Use subword matching')
    args = parser.parse_args()

    # Setup
    device = torch.device("cpu")
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)
    model.eval()

    print(f"Loading word2sent from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent = pickle.load(f)

    vocab = list(word2sent.keys())
    print(f"Vocabulary size: {len(vocab)}")

    vec_file = os.path.join(args.output_folder, "vec.txt")
    count_file = os.path.join(args.output_folder, "count.txt")

    with open(vec_file, "w") as f_vec, open(count_file, "w") as f_count:
        for token in tqdm(vocab, desc="Extracting embeddings"):
            sentences = word2sent[token][:args.nsent]

            if len(sentences) < 1:
                continue

            # Tokenize the target word
            token_ids = tokenizer(" " + token, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 0:
                continue

            # Remove leading ▁ if present
            if tokenizer.convert_ids_to_tokens(token_ids)[0] == '▁':
                token_ids = token_ids[1:]

            # Tokenize sentences
            sentences_tokens = tokenizer(sentences, padding=True, return_tensors='pt')
            sentences_ids = sentences_tokens["input_ids"].tolist()

            # Find word positions
            col_idx, valid_sent_id = identify_word_indices(
                sentences_ids, token, token_ids, tokenizer, subword=args.subword
            )

            if len(valid_sent_id) == 0:
                print(f"Warning: '{token}' not found in sentences")
                continue

            # Filter to valid sentences
            valid_sentences = [sentences[i] for i in valid_sent_id]

            # Extract embeddings
            token_states = encode_batch_cpu(tokenizer, model, valid_sentences, col_idx, device)

            if len(token_states) == 0:
                continue

            # Average across sentences
            static_embedding = token_states.mean(axis=0)

            # Write output
            veckey = "▁".join(token.split(" "))
            f_count.write(f"{veckey} {len(token_states)}\n")
            f_vec.write(f"{veckey} " + " ".join(map(str, static_embedding)) + "\n")

    print(f"\n✓ Embeddings saved to {vec_file}")
    print(f"✓ Counts saved to {count_file}")


if __name__ == "__main__":
    main()
