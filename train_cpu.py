"""
CPU-compatible version of train.py for small-scale testing.
Knowledge distillation training for static word embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer
import copy
from tqdm import tqdm
import string
import pickle
import argparse
from scipy import stats
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from util import load_w2v

PUNCTLIST = set(list(string.punctuation) + ["。", "、", "？", "！", "「", "」", "（", "）", "：", "・", "，"])


def parse_args():
    parser = argparse.ArgumentParser(description='Train sentence embeddings (CPU version)')
    parser.add_argument('-bs', default=32, type=int, help='Batch size (smaller for CPU)')
    parser.add_argument('-prompt', default="", type=str, help='Prompt for sentence transformer')
    parser.add_argument('-vec_path', required=True, type=str, help='Path to word vectors')
    parser.add_argument('-model', required=True, type=str, help='Sentence transformer model')
    parser.add_argument('-output_folder', required=True, type=str, help='Output folder')
    parser.add_argument('-epoch', default=1, type=int, help='Number of epochs')
    parser.add_argument('-temp', default=0.05, type=float, help='Temperature for distillation')
    parser.add_argument('-word2sent', required=True, type=str, help='Phrase to sentence mapping file')
    parser.add_argument('-sts_eval', action='store_true', help='Evaluate on STS-B')
    parser.add_argument('-max_steps', default=500, type=int, help='Max training steps (for quick testing)')
    return parser.parse_args()


class Net(nn.Module):
    """Neural network for sentence embedding - CPU version."""

    def __init__(self, word2vec, model_path, device):
        super().__init__()
        self.device = device

        # Get embedding dimension
        self.embd = len(next(iter(word2vec.values())))

        # Build vocabulary
        self.vocab2id = {w: idx + 1 for idx, w in enumerate(word2vec.keys())}

        # Embedding layer
        self.emb = nn.Embedding(len(word2vec) + 1, self.embd, padding_idx=0)

        # Initialize embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.punct_list = PUNCTLIST

        with torch.no_grad():
            print("Initializing embeddings...")
            for w, vec in word2vec.items():
                wid = self.vocab2id[w]
                self.emb.weight.data[wid] = torch.FloatTensor(vec)

    def encode(self, sents):
        """Encode sentences to embeddings."""
        embeddings = []

        for sent in sents:
            words = [x[0] for x in
                    self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]

            sent_emb = []
            idx = 0
            n_words = len(words)

            while idx < n_words:
                w = words[idx]

                if w not in self.punct_list:
                    # Try exact match or lowercase
                    if w in self.vocab2id or w.lower() in self.vocab2id:
                        w = w if w in self.vocab2id else w.lower()
                        emb = self.emb.weight[self.vocab2id[w]]
                        sent_emb.append(emb)
                        idx += 1
                        continue
                    else:
                        # Try subword matching
                        subwords = self.model_tokenizer.tokenize(w)
                        if len(subwords):
                            subword_flag = False
                            while len(subwords) > 1:
                                subwords = subwords[:-1]
                                subwords_str = "".join(subwords).replace("##", "").replace("▁", "")

                                if subwords_str in self.vocab2id or subwords_str.lower() in self.vocab2id:
                                    subwords_str = subwords_str if subwords_str in self.vocab2id else subwords_str.lower()
                                    emb = self.emb.weight[self.vocab2id[subwords_str]]
                                    sent_emb.append(emb)
                                    subword_flag = True
                                    break

                            if subword_flag:
                                idx += 1
                                continue

                idx += 1

            if len(sent_emb) > 0:
                sent_emb = torch.stack(sent_emb).sum(dim=0)
            else:
                sent_emb = torch.zeros(self.embd).to(self.device)

            embeddings.append(sent_emb)

        return torch.stack(embeddings)

    def forward(self, sents):
        return self.encode(sents)


def distill_loss(sim_s, sim_t):
    """Compute distillation loss."""
    p = F.log_softmax(sim_s, dim=-1)
    q = F.softmax(sim_t, dim=-1)
    loss = (-(q * p).nansum() / q.nansum()).mean()
    return loss


def compute_batch_loss(batch, st_model, model, temp, device):
    """Compute loss for a batch."""
    # Get teacher embeddings
    with torch.no_grad():
        if hasattr(st_model, 'prompt') and st_model.prompt:
            batch_st = [st_model.prompt + s for s in batch]
        else:
            batch_st = batch

        embeddings = st_model.encode(batch_st, convert_to_numpy=False)
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        embeddings = embeddings.to(device)

    embeddings = F.normalize(embeddings, dim=-1)

    # Get student embeddings
    static_embeddings = model(batch)
    static_embeddings = F.normalize(static_embeddings, dim=-1)

    # Compute similarity matrices
    cossim = torch.matmul(embeddings, embeddings.T)
    cossim_static = torch.matmul(static_embeddings, static_embeddings.T)

    # Mask diagonal
    for k in range(len(cossim_static)):
        cossim[k][k] = float("-inf")
        cossim_static[k][k] = float("-inf")

    # Compute distillation loss
    loss = distill_loss(cossim_static / temp, cossim / temp)
    return loss


def evaluate_sts(model, sentence1, sentence2, gold_score, device):
    """Evaluate on STS-B dataset."""
    cossim_list = []

    for j in tqdm(range(0, len(sentence1), 64), desc="Evaluating"):
        sent1_tmp = sentence1[j:j + 64]
        sent2_tmp = sentence2[j:j + 64]

        static_embeddings1 = model(sent1_tmp)
        static_embeddings2 = model(sent2_tmp)

        static_embeddings1 = F.normalize(static_embeddings1, dim=-1)
        static_embeddings2 = F.normalize(static_embeddings2, dim=-1)

        cossim = torch.sum(static_embeddings1 * static_embeddings2, dim=-1).cpu().tolist()
        cossim_list.extend(cossim)

    spearmanr_score = stats.spearmanr(cossim_list, gold_score)[0]
    return spearmanr_score


def main():
    args = parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load word vectors
    print(f"Loading word vectors from {args.vec_path}")
    word2vec, edim = load_w2v(args.vec_path)
    print(f"Loaded {len(word2vec)} words with dimension {edim}")

    # Load training data
    print(f"Loading training data from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent_dict = pickle.load(f)

    sents_train = []
    sents_dev = []
    n_sent_per_word = 3

    for w in word2sent_dict.keys():
        if len(word2sent_dict[w]) >= 5:
            sents = np.random.permutation(word2sent_dict[w])
            for k in range(n_sent_per_word):
                if k < 2:
                    sents_train.append(sents[k])
                else:
                    sents_dev.append(sents[k])

    sents_train = list(set(sents_train))
    sents_dev = list(set(sents_dev) - set(sents_train))
    sents_dev = np.random.permutation(sents_dev)[:min(500, len(sents_dev))].tolist()

    print(f"Training samples: {len(sents_train)}")
    print(f"Dev samples: {len(sents_dev)}")

    # Initialize model
    print(f"Initializing model with embedding dim: {edim}")
    model = Net(word2vec, args.model, device)
    model.to(device)

    # Load sentence transformer (teacher)
    print(f"Loading sentence transformer: {args.model}")
    st_model = SentenceTransformer(args.model, trust_remote_code=True, device='cpu')
    st_model.prompt = args.prompt if args.prompt else ""
    st_model.eval()

    for param in st_model.parameters():
        param.requires_grad = False

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load STS-B for evaluation (optional)
    if args.sts_eval:
        print("Loading STS-B dataset for evaluation")
        stsb_train = load_dataset("sentence-transformers/stsb", split="train")
        sentence1 = stsb_train['sentence1'][:200]  # Use subset for speed
        sentence2 = stsb_train['sentence2'][:200]
        gold_score = stsb_train['score'][:200]

    # Training loop
    best_dev_loss = float('inf')
    best_model = None
    best_step = 0

    total_steps = min(args.max_steps, len(sents_train) // args.bs)

    print(f"\n=== Starting Training ({total_steps} steps) ===")

    model.train()
    running_loss = 0.0

    for step in tqdm(range(total_steps), desc="Training"):
        # Sample batch
        batch_idx = np.random.choice(len(sents_train), args.bs, replace=False)
        batch = [sents_train[i] for i in batch_idx]

        optimizer.zero_grad()
        loss = compute_batch_loss(batch, st_model, model, args.temp, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()

        # Validation every 50 steps
        if (step + 1) % 50 == 0 or step == total_steps - 1:
            model.eval()
            dev_loss_total = 0.0

            with torch.no_grad():
                for j in range(0, len(sents_dev), args.bs):
                    batch = sents_dev[j:j + args.bs]
                    if len(batch) > 1:
                        dev_loss = compute_batch_loss(batch, st_model, model, args.temp, device)
                        dev_loss_total += dev_loss.item()

            avg_train_loss = running_loss / 50
            print(f"\nStep {step+1}: train_loss={avg_train_loss:.4f}, dev_loss={dev_loss_total:.4f}")
            running_loss = 0.0

            # Save best model
            if dev_loss_total < best_dev_loss:
                best_dev_loss = dev_loss_total
                best_step = step + 1
                best_model = copy.deepcopy(model)
                print(f"  -> New best model!")

            # Optional STS evaluation
            if args.sts_eval:
                with torch.no_grad():
                    spearman = evaluate_sts(model, sentence1, sentence2, gold_score, device)
                    print(f"  -> STS-B Spearman: {spearman:.4f}")

            model.train()

    # Save final embeddings
    print(f"\nSaving embeddings to {args.output_folder}")
    import os
    os.makedirs(args.output_folder, exist_ok=True)

    output_file = f"{args.output_folder}/trained_embeddings.txt"

    save_model = best_model if best_model is not None else model

    with open(output_file, "w") as f:
        for w in word2vec:
            vec = save_model.emb.weight[save_model.vocab2id[w]].data.cpu().numpy()
            f.write(w + " " + " ".join([str(x) for x in vec]) + "\n")

    print(f"Best step: {best_step}")
    print(f"Saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
