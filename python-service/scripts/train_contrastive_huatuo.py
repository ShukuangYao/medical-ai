"""
对比学习训练脚本：用 Huatuo 的 (questions, answers) 构造检索式正负样本对

训练目标（推荐）：
- 使用 SentenceTransformer 的 MultipleNegativesRankingLoss
- 同一个 batch 内其它样本的 positive 当作 negatives（in-batch negatives）

输出：
- 保存到 --output-dir
"""

import argparse
import os
import random
from typing import Dict, Any, List

import datasets
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


def _get_train_split(ds: datasets.DatasetDict):
    if "train" in ds:
        return ds["train"]
    # 兼容其它 split 命名
    return list(ds.values())[0]


def _truncate(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    if max_chars <= 0:
        return s
    return s[:max_chars]


def _extract_qa(item: Dict[str, Any]):
    """
    兼容数据集字段在不同版本可能不同：
    - questions/answers
    - question/answer
    """
    q = item.get("question", item.get("questions", item.get("input", "")))
    a = item.get("answer", item.get("answers", item.get("output", "")))
    return q, a


def main():
    parser = argparse.ArgumentParser(description="Train contrastive retrieval model (Huatuo)")
    parser.add_argument("--dataset-name", default="FreedomIntelligence/huatuo26M-testdatasets")
    parser.add_argument("--model-name", default="BAAI/bge-large-zh")
    parser.add_argument("--output-dir", default="./outputs/huatuo-contrastive")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)

    # 为了避免训练时超长文本导致显存爆炸，做一个简单的字符截断
    parser.add_argument("--max-query-chars", type=int, default=256)
    parser.add_argument("--max-doc-chars", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=512)

    # device 常见取值：cpu/cuda/mps
    parser.add_argument("--device", default=None, help="例如 mps 或 cpu；默认让 SentenceTransformer 自动选择")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] Load dataset...")
    ds = datasets.load_dataset(args.dataset_name)
    train_split = _get_train_split(ds)
    n_total = len(train_split)
    n_train = min(args.max_train_samples, n_total)
    # shuffle + select，避免顺序偏差
    train_split = train_split.shuffle(seed=args.seed).select(range(n_train))
    print(f"Train rows: {n_train}/{n_total}")

    print("[2/5] Build training examples...")
    model = SentenceTransformer(args.model_name, device=args.device)
    model.max_seq_length = args.max_seq_length

    examples: List[InputExample] = []
    for item in train_split:
        q, a = _extract_qa(item)
        if not q or not a:
            continue

        q = _truncate(str(q), args.max_query_chars)
        a = _truncate(str(a), args.max_doc_chars)

        # 对齐你现有推理代码的格式：
        # - embed_query 会给 query 加前缀
        # - embed_documents 没有加前缀（这里 doc 也不加）
        query_text = f"为这个句子生成表示以用于检索相关文章：{q}"
        doc_text = f"问题：{q}\n回答：{a}"

        examples.append(InputExample(texts=[query_text, doc_text]))

    if not examples:
        raise RuntimeError("没有构造到任何训练样本，请检查数据集字段或截断参数。")
    print(f"Built {len(examples)} examples.")

    print("[3/5] Prepare DataLoader & Loss...")
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    # MultipleNegativesRankingLoss：in-batch negatives（同 batch 的其它正样本当 negatives）
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("[4/5] Train...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=args.output_dir,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
        use_amp=False,  # MPS 上先保守一些
        show_progress_bar=True,
    )

    print("[5/5] Save model...")
    # model.fit 已经会保存到 output_path，这里再显式保存一次更稳
    model.save(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

