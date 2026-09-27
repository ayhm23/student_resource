"""Dense (transformer) candidate retrieval on the GPU -- a second blocking channel.

Exact-token keys (K1-K7) miss matches whose name AND address are both
re-spelled (transliteration, abbreviation, typos, word order). A multilingual
sentence encoder fine-tuned on the train matched pairs maps both sides of such
pairs close together, and nearest-neighbour search over its embeddings finds
them regardless of which tokens changed.

Model: ``intfloat/multilingual-e5-small`` (118M params, MIT) -- fits a 4 GB
laptop GPU for fine-tuning in fp16; the ``BER_DENSE_MODEL`` env var swaps in a
larger encoder on a bigger GPU.

Subcommands (run with the GPU venv, ``.venv-gpu``):
  finetune  contrastive fine-tuning (MultipleNegativesRankingLoss) on train
            matched pairs, each with the hardest blocking non-match of its S1
            as an explicit negative (the planted look-alikes)
  encode    embed every record of a split -> data/dense/{split}_{source}.npy (fp16, L2-normalized)
  search    exact top-K cosine neighbours per S1 within its country, on the GPU
            -> data/cand/dense_{split}.parquet (s1_id, match_id, match_source, dense_cos, dense_rank)
  recall    on the 300k train sample: recall of dense top-K alone and unioned
            with the key candidates -- the go/no-go number for this channel

Text fed to the model: "<business_name> | <business_address>" (raw, any script).
"""

import argparse
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from config import CAND_DIR, DATA_DIR, FEATURES_DIR, TRAIN_GT_LONG_PARQUET, TRAIN_SAMPLE_S1_PARQUET, raw_parquet_path
from perf import fmt_hms, progress_line
from resources import default_scratch_dir

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE_MODEL = os.environ.get("BER_DENSE_MODEL", "intfloat/multilingual-e5-small")
DENSE_DIR = DATA_DIR / "dense"
MODEL_DIR = DATA_DIR / "models" / "dense_encoder"
SCRATCH_DIR = default_scratch_dir()
MAX_SEQ_LEN = 64
PREFIX = "query: "  # e5 convention for symmetric similarity
TOP_K = int(os.environ.get("BER_DENSE_TOPK", "30"))
MINI_BATCH = int(os.environ.get("BER_DENSE_MINI_BATCH", "32"))


def record_text_sql(alias="r"):
    return (f"'{PREFIX}' || coalesce({alias}.business_name, '') || ' | ' || "
            f"coalesce({alias}.business_address, '')")


def _con():
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=4")
    return con


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------

def build_training_triplets(n_pairs, seed=42):
    """(anchor, positive, negative) texts: sampled GT pairs + the S1's hardest blocking non-match."""
    con = _con()
    s1 = raw_parquet_path("train", "S1").as_posix()
    s2 = raw_parquet_path("train", "S2").as_posix()
    s3 = raw_parquet_path("train", "S3").as_posix()
    gt = TRAIN_GT_LONG_PARQUET.as_posix()
    con.execute(f"SELECT setseed({(seed % 100) / 100})")
    con.execute(f"""
        CREATE TEMP TABLE pairs AS
        SELECT s1_id, match_id, match_source FROM '{gt}'
        USING SAMPLE reservoir({n_pairs} ROWS) REPEATABLE ({seed})
    """)
    # Hard negatives = the S1's highest cheap-score blocking candidate that is not
    # a true match (the planted look-alikes). Read from the blocking warehouse, so
    # this step only needs blocking to have run (not featurization).
    con.execute(f"ATTACH '{(SCRATCH_DIR / 'warehouse.duckdb').as_posix()}' AS wh (READ_ONLY)")
    con.execute(f"""
        CREATE TEMP TABLE hardneg AS
        SELECT s1_id, match_id AS neg_id, match_source AS neg_source FROM (
            SELECT c.s1_id, c.match_id, c.match_source,
                   row_number() OVER (PARTITION BY c.s1_id ORDER BY c.cheap_score DESC, c.match_id) AS rn
            FROM wh.train_scored c
            ANTI JOIN '{gt}' g ON g.s1_id = c.s1_id AND g.match_id = c.match_id AND g.match_source = c.match_source
            WHERE c.s1_id IN (SELECT DISTINCT s1_id FROM pairs)
        ) WHERE rn = 1
    """)
    con.execute(f"""
        CREATE TEMP VIEW s23 AS
        SELECT 'S2' AS source, id, business_name, business_address FROM '{s2}'
        UNION ALL SELECT 'S3' AS source, id, business_name, business_address FROM '{s3}'
    """)
    df = con.execute(f"""
        SELECT {record_text_sql('a')} AS anchor, {record_text_sql('b')} AS positive,
               {record_text_sql('n')} AS negative
        FROM pairs p
        JOIN '{s1}' a ON a.id = p.s1_id
        JOIN s23 b ON b.id = p.match_id AND b.source = p.match_source
        JOIN hardneg h ON h.s1_id = p.s1_id
        JOIN s23 n ON n.id = h.neg_id AND n.source = h.neg_source
    """).df()
    con.close()
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def finetune(n_pairs, epochs, batch_size, lr):
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)

    t0 = time.perf_counter()
    df = build_training_triplets(n_pairs)
    print(f"[dense] {len(df)} training triplets built in {fmt_hms(time.perf_counter() - t0)}; "
          f"example: {df.iloc[0].to_dict()}")
    model = SentenceTransformer(BASE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = MAX_SEQ_LEN
    # The 250k-token multilingual embedding matrix is most of the parameters; its
    # AdamW state alone does not fit a 4 GB GPU, and the pretrained token
    # embeddings are what we want to keep anyway -- fine-tune the transformer layers.
    frozen = 0
    for name, p in model.named_parameters():
        if "word_embeddings" in name:
            p.requires_grad = False
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dense] frozen {frozen / 1e6:.0f}M embedding params, training {trainable / 1e6:.0f}M")
    # GradCache variant: the full batch's in-batch negatives, computed in small
    # mini-batches -- a plain MNRL batch filled the 4 GB GPU and the driver
    # started spilling to system RAM (100% util at half power, 1.5 s/step).
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(DATA_DIR / "models" / "dense_ckpt"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr, warmup_ratio=0.05,
        fp16=torch.cuda.is_available(), logging_steps=200, save_strategy="no", report_to=[],
        dataloader_num_workers=0, batch_sampler="no_duplicates", seed=42,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_pandas(df),
                                         loss=loss)
    trainer.train()
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    print(f"[dense] fine-tuned model saved -> {MODEL_DIR} ({fmt_hms(time.perf_counter() - t0)})")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    path = str(MODEL_DIR) if MODEL_DIR.exists() else BASE_MODEL
    model = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = MAX_SEQ_LEN
    if torch.cuda.is_available():
        model.half()
    print(f"[dense] encoder: {path}")
    return model


def encode(split, sources, batch_size, id_filter=None):
    """Embed every record of (split, source) -> fp16 .npy + ids/country parquet, in file order."""
    model = load_model()
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    con = _con()
    for source in sources:
        path = raw_parquet_path(split, source).as_posix()
        where = f"WHERE id IN (SELECT id FROM '{id_filter}')" if id_filter else ""
        meta = con.execute(f"SELECT id, country, {record_text_sql('r')} AS text FROM '{path}' r {where}").df()
        n = len(meta)
        out = np.lib.format.open_memmap(DENSE_DIR / f"{split}_{source}.npy", mode="w+", dtype=np.float16,
                                        shape=(n, model.get_sentence_embedding_dimension()))
        # Sort by length so each batch pads to similar lengths (much faster), write back in file order.
        order = np.argsort(meta["text"].str.len().to_numpy(), kind="stable")
        texts = meta["text"].to_numpy()
        t0 = time.perf_counter()
        step = batch_size * 50
        for lo in range(0, n, step):
            idx = order[lo:lo + step]
            emb = model.encode(list(texts[idx]), batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            out[idx] = emb.astype(np.float16)
            print(f"  {progress_line(f'encode {split}_{source}', min(lo + step, n), n, t0)}")
        out.flush()
        del out
        meta[["id", "country"]].to_parquet(DENSE_DIR / f"{split}_{source}_ids.parquet", index=False)
    con.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search(split, top_k, query_batch, pool_chunk, s1_filter=None):
    """Exact top-K cosine neighbours of each S1 among S2+S3 records of the same country, on the GPU."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s1_ids = pd.read_parquet(DENSE_DIR / f"{split}_S1_ids.parquet")
    s1_emb = np.load(DENSE_DIR / f"{split}_S1.npy", mmap_mode="r")
    pools = []
    for src in ("S2", "S3"):
        ids = pd.read_parquet(DENSE_DIR / f"{split}_{src}_ids.parquet")
        pools.append((src, ids, np.load(DENSE_DIR / f"{split}_{src}.npy", mmap_mode="r")))
    if s1_filter is not None:
        keep = s1_ids["id"].isin(s1_filter).to_numpy()
    else:
        keep = np.ones(len(s1_ids), dtype=bool)

    out_parts = []
    t0 = time.perf_counter()
    for country in sorted(s1_ids["country"].unique()):
        q_idx = np.flatnonzero((s1_ids["country"].to_numpy() == country) & keep)
        if not len(q_idx):
            continue
        Q = torch.from_numpy(np.ascontiguousarray(s1_emb[q_idx])).to(dev)
        best_s = torch.full((len(q_idx), top_k), -2.0, device=dev, dtype=torch.float16)
        best_i = torch.full((len(q_idx), top_k), -1, device=dev, dtype=torch.int64)
        offset = 0
        pool_keys = []
        for src, ids, emb in pools:
            p_idx = np.flatnonzero(ids["country"].to_numpy() == country)
            pool_keys.append((src, ids["id"].to_numpy()[p_idx]))
            for lo in range(0, len(p_idx), pool_chunk):
                chunk = p_idx[lo:lo + pool_chunk]
                P = torch.from_numpy(np.ascontiguousarray(emb[chunk])).to(dev)
                for qlo in range(0, len(q_idx), query_batch):
                    s = Q[qlo:qlo + query_batch] @ P.T
                    k = min(top_k, s.shape[1])
                    cs, ci = torch.topk(s, k, dim=1)
                    ci = ci + offset + lo
                    merged_s = torch.cat([best_s[qlo:qlo + query_batch], cs], dim=1)
                    merged_i = torch.cat([best_i[qlo:qlo + query_batch], ci], dim=1)
                    ts, ti = torch.topk(merged_s, top_k, dim=1)
                    best_s[qlo:qlo + query_batch] = ts
                    best_i[qlo:qlo + query_batch] = torch.gather(merged_i, 1, ti)
                del P
            offset += len(p_idx)
        bs, bi = best_s.float().cpu().numpy(), best_i.cpu().numpy()
        n2 = len(pool_keys[0][1])
        all_ids = np.concatenate([pool_keys[0][1], pool_keys[1][1]])
        valid = bi >= 0
        rows = np.repeat(np.arange(len(q_idx)), top_k).reshape(len(q_idx), top_k)
        out_parts.append(pd.DataFrame({
            "s1_id": s1_ids["id"].to_numpy()[q_idx][rows[valid]],
            "match_id": all_ids[bi[valid]],
            "match_source": np.where(bi[valid] < n2, "S2", "S3"),
            "dense_cos": bs[valid].astype(np.float32),
            "dense_rank": (np.argsort(np.argsort(-bs, axis=1), axis=1) + 1)[valid].astype(np.int16),
        }))
        print(f"  [{split}] {country}: {len(q_idx)} S1 x {offset} S2/S3 searched "
              f"(elapsed {fmt_hms(time.perf_counter() - t0)})")
        del Q, best_s, best_i
    out = pd.concat(out_parts, ignore_index=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    path = CAND_DIR / f"dense_{split}.parquet"
    out.to_parquet(path, index=False)
    print(f"[dense] {len(out)} dense candidates -> {path}")
    return out


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

def recall_report(dense_path):
    """Recall on the 300k train sample: dense top-K, key candidates, and their union, by K and country."""
    con = _con()
    sample = TRAIN_SAMPLE_S1_PARQUET.as_posix()
    feats = (FEATURES_DIR / "train_candidates_features.parquet").as_posix()
    con.execute(f"""CREATE TEMP TABLE gt AS SELECT g.*, s.country FROM '{TRAIN_GT_LONG_PARQUET.as_posix()}' g
                    JOIN '{sample}' s ON s.id = g.s1_id""")
    con.execute(f"""CREATE TEMP TABLE keyc AS SELECT DISTINCT s1_id, match_id, other_source AS match_source
                    FROM '{feats}' WHERE s1_id IN (SELECT id FROM '{sample}')""")
    con.execute(f"CREATE TEMP TABLE dn AS SELECT * FROM '{dense_path.as_posix()}'")
    n = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    print(f"[dense] recall on {n} true pairs of the 300k sample:")
    base = con.execute("""SELECT count(*) FROM gt JOIN keyc USING (s1_id, match_id, match_source)""").fetchone()[0]
    print(f"  key candidates (current pipeline): {100 * base / n:.2f}%")
    for k in sorted({5, 10, 20, TOP_K}):
        if k > TOP_K:
            continue
        d = con.execute(f"""SELECT count(*) FROM gt JOIN (SELECT * FROM dn WHERE dense_rank <= {k}) d
                            USING (s1_id, match_id, match_source)""").fetchone()[0]
        u = con.execute(f"""SELECT count(*) FROM gt WHERE EXISTS (SELECT 1 FROM keyc c WHERE c.s1_id = gt.s1_id
                              AND c.match_id = gt.match_id AND c.match_source = gt.match_source)
                            OR EXISTS (SELECT 1 FROM dn d WHERE d.dense_rank <= {k} AND d.s1_id = gt.s1_id
                              AND d.match_id = gt.match_id AND d.match_source = gt.match_source)""").fetchone()[0]
        print(f"  dense top-{k}: {100 * d / n:.2f}%   union(keys, dense top-{k}): {100 * u / n:.2f}%")
    for country, tot, hit in con.execute(f"""
            SELECT gt.country, count(*), sum(CASE WHEN c.s1_id IS NOT NULL OR d.s1_id IS NOT NULL THEN 1 ELSE 0 END)
            FROM gt LEFT JOIN keyc c USING (s1_id, match_id, match_source)
            LEFT JOIN (SELECT DISTINCT s1_id, match_id, match_source FROM dn) d USING (s1_id, match_id, match_source)
            GROUP BY 1""").fetchall():
        print(f"  union by country {country}: {100 * hit / tot:.2f}%")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finetune")
    f.add_argument("--pairs", type=int, default=600_000)
    f.add_argument("--epochs", type=float, default=1.0)
    f.add_argument("--batch", type=int, default=96)
    f.add_argument("--lr", type=float, default=5e-5)
    e = sub.add_parser("encode")
    e.add_argument("split", choices=["train", "test"])
    e.add_argument("--sources", nargs="+", default=["S1", "S2", "S3"])
    e.add_argument("--batch", type=int, default=512)
    e.add_argument("--sample-s1", action="store_true", help="encode only the 300k recall-sample S1 (train)")
    s = sub.add_parser("search")
    s.add_argument("split", choices=["train", "test"])
    s.add_argument("--top-k", type=int, default=TOP_K)
    s.add_argument("--query-batch", type=int, default=2048)
    s.add_argument("--pool-chunk", type=int, default=200_000)
    s.add_argument("--sample-s1", action="store_true")
    sub.add_parser("recall")
    a = ap.parse_args()

    if a.cmd == "finetune":
        finetune(a.pairs, a.epochs, a.batch, a.lr)
    elif a.cmd == "encode":
        for src in a.sources:
            filt = TRAIN_SAMPLE_S1_PARQUET.as_posix() if (a.sample_s1 and src == "S1") else None
            encode(a.split, [src], a.batch, filt)
    elif a.cmd == "search":
        filt = pd.read_parquet(TRAIN_SAMPLE_S1_PARQUET)["id"] if a.sample_s1 else None
        search(a.split, a.top_k, a.query_batch, a.pool_chunk, filt)
    elif a.cmd == "recall":
        recall_report(CAND_DIR / "dense_train.parquet")


if __name__ == "__main__":
    main()
