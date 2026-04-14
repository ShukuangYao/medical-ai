"""
基于医疗数据集构建 Neo4j 知识图谱（参考 RAGQnASystem/build_up_graph.py）

输入数据格式（JSONL，每行一个 JSON）兼容字段示例：
- name / desc / cause / prevent / cure_lasttime / cured_prob / easy_get
- common_drug / recommand_drug
- do_eat / recommand_eat / not_eat
- check / cure_department / symptom / cure_way / acompany / drug_detail

使用方式：
python scripts/build_neo4j_graph.py \
  --data-file ../data/medical_new_2.json \
  --clear
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any

# 关系批量写入：每事务条数（过小则慢，过大易触发内存/超时）
_REL_BATCH_SIZE = 800

from neo4j import GraphDatabase

# 让脚本在直接运行时也能找到 `app` 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings


@dataclass
class GraphPayload:
    diseases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entities: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "Drug": set(),
            "Food": set(),
            "Check": set(),
            "Department": set(),
            "Symptom": set(),
            "Treatment": set(),
            "Producer": set(),
        }
    )
    relationships: Set[Tuple[str, str, str, str, str]] = field(default_factory=set)


def _safe_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _clean_symptom(text: str) -> str:
    return text[:-3] if text.endswith("...") else text


def _normalize_cure_way(cure_way: List[Any]) -> List[str]:
    out: List[str] = []
    for item in cure_way:
        if isinstance(item, list):
            if item:
                item = item[0]
            else:
                continue
        s = str(item).strip()
        if len(s) >= 2:
            out.append(s)
    return out


def parse_dataset(data_file: str) -> GraphPayload:
    payload = GraphPayload()

    with open(data_file, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            # medical_new_2.json 每行形如 {...},（数组元素导出），尾部逗号会导致 json.loads 报 Extra data
            if raw.endswith(","):
                raw = raw[:-1].rstrip()

            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(item, dict):
                continue

            disease_name = str(item.get("name", "")).strip()
            if not disease_name:
                continue

            payload.diseases[disease_name] = {
                "name": disease_name,
                "desc": str(item.get("desc", "")),
                "cause": str(item.get("cause", "")),
                "prevent": str(item.get("prevent", "")),
                "cure_lasttime": str(item.get("cure_lasttime", "")),
                "cured_prob": str(item.get("cured_prob", "")),
                "easy_get": str(item.get("easy_get", "")),
            }

            # 疾病 -> 药品
            drugs = _safe_list(item.get("common_drug")) + _safe_list(item.get("recommand_drug"))
            for d in drugs:
                payload.entities["Drug"].add(d)
                payload.relationships.add(("Disease", disease_name, "USES_DRUG", "Drug", d))

            # 疾病 -> 食物（宜吃 / 忌吃）
            do_eat = _safe_list(item.get("do_eat")) + _safe_list(item.get("recommand_eat"))
            no_eat = _safe_list(item.get("not_eat"))
            for food in do_eat:
                payload.entities["Food"].add(food)
                payload.relationships.add(("Disease", disease_name, "RECOMMEND_FOOD", "Food", food))
            for food in no_eat:
                payload.entities["Food"].add(food)
                payload.relationships.add(("Disease", disease_name, "AVOID_FOOD", "Food", food))

            # 疾病 -> 检查
            checks = _safe_list(item.get("check"))
            for ch in checks:
                payload.entities["Check"].add(ch)
                payload.relationships.add(("Disease", disease_name, "RECOMMEND_CHECK", "Check", ch))

            # 疾病 -> 科室（取最后一级）
            departments = _safe_list(item.get("cure_department"))
            if departments:
                dept = departments[-1]
                payload.entities["Department"].add(dept)
                payload.relationships.add(("Disease", disease_name, "IN_DEPARTMENT", "Department", dept))

            # 疾病 -> 症状
            symptoms = [_clean_symptom(s) for s in _safe_list(item.get("symptom"))]
            for sy in symptoms:
                payload.entities["Symptom"].add(sy)
                payload.relationships.add(("Disease", disease_name, "HAS_SYMPTOM", "Symptom", sy))

            # 疾病 -> 治疗方式
            cure_way = _normalize_cure_way(item.get("cure_way", []))
            for cw in cure_way:
                payload.entities["Treatment"].add(cw)
                payload.relationships.add(("Disease", disease_name, "HAS_TREATMENT", "Treatment", cw))

            # 疾病并发症
            for co in _safe_list(item.get("acompany")):
                payload.relationships.add(("Disease", disease_name, "ACOMPANY_WITH", "Disease", co))

            # 药品商 -> 药品（drug_detail: "药品,厂商"）
            for detail in _safe_list(item.get("drug_detail")):
                pair = [x.strip() for x in detail.split(",")]
                if len(pair) != 2:
                    continue
                drug, producer = pair[0], pair[1]
                if not drug or not producer:
                    continue
                payload.entities["Drug"].add(drug)
                payload.entities["Producer"].add(producer)
                payload.relationships.add(("Producer", producer, "PRODUCES", "Drug", drug))

    return payload


def _merge_named_node(tx, label: str, name: str):
    tx.run(f"MERGE (n:{label} {{name: $name}})", name=name)


def _merge_disease_node(tx, props: Dict[str, Any]):
    tx.run(
        """
        MERGE (d:Disease {name: $name})
        SET d.desc = $desc,
            d.cause = $cause,
            d.prevent = $prevent,
            d.cure_lasttime = $cure_lasttime,
            d.cured_prob = $cured_prob,
            d.easy_get = $easy_get
        """,
        **props,
    )


_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _merge_relationship_batch(
    tx,
    src_label: str,
    rel_type: str,
    dst_label: str,
    pairs: List[Tuple[str, str]],
):
    """同一 (源标签, 关系类型, 目标标签) 下批量 MERGE，减少事务次数。"""
    for s in (src_label, rel_type, dst_label):
        if not _ID.match(s):
            raise ValueError(f"非法 Neo4j 标识符: {s!r}")
    tx.run(
        f"""
        UNWIND $pairs AS p
        MATCH (a:{src_label} {{name: p.src}})
        MATCH (b:{dst_label} {{name: p.dst}})
        MERGE (a)-[r:{rel_type}]->(b)
        """,
        pairs=[{"src": a, "dst": b} for a, b in pairs],
    )


def import_to_neo4j(payload: GraphPayload, clear: bool = False):
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    # Neo4j 5 多数据库：显式选择目标业务库（例如 medical-db）
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        if clear:
            print("清空 Neo4j 旧数据...")
            session.run("MATCH (n) DETACH DELETE n")

        print("导入疾病节点...")
        for disease in payload.diseases.values():
            session.execute_write(_merge_disease_node, disease)

        for label, names in payload.entities.items():
            print(f"导入 {label} 节点: {len(names)}")
            for name in names:
                session.execute_write(_merge_named_node, label, name)

        rel_total = len(payload.relationships)
        print(f"开始导入关系: {rel_total} 条（按类型分批写入，请稍候）…")
        by_pattern: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = defaultdict(list)
        for src_label, src_name, rel, dst_label, dst_name in payload.relationships:
            by_pattern[(src_label, rel, dst_label)].append((src_name, dst_name))

        done = 0
        for (src_label, rel, dst_label), pairs in by_pattern.items():
            for i in range(0, len(pairs), _REL_BATCH_SIZE):
                batch = pairs[i : i + _REL_BATCH_SIZE]
                session.execute_write(
                    _merge_relationship_batch,
                    src_label,
                    rel,
                    dst_label,
                    batch,
                )
                done += len(batch)
                if done % 50000 == 0 or done == rel_total:
                    print(f"  关系进度: {done}/{rel_total}")

        print("正在统计节点与关系数量…")
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"导入完成: nodes={node_count}, relationships={rel_count}")
        type_rows = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS c ORDER BY c DESC"
        ).data()
        actual_by_type = {row["rel_type"]: row["c"] for row in type_rows}
        expected_by_type: Dict[str, int] = defaultdict(int)
        for (_, rel, _), pairs in by_pattern.items():
            expected_by_type[rel] += len(pairs)

        print("各关系类型写入统计（成功/候选）:")
        for rel_type in sorted(set(expected_by_type) | set(actual_by_type)):
            ok = actual_by_type.get(rel_type, 0)
            expected = expected_by_type.get(rel_type, 0)
            print(f"  {rel_type}: {ok}/{expected}")

    driver.close()


def main():
    parser = argparse.ArgumentParser(description="从 medical JSONL 构建 Neo4j 知识图谱")
    parser.add_argument(
        "--data-file",
        default=os.path.join("..", "data", "medical_new_2.json"),
        help="输入数据文件（JSONL）",
    )
    parser.add_argument("--clear", action="store_true", help="导入前清空 Neo4j")
    args = parser.parse_args()

    data_file = args.data_file
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), data_file))

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    print(f"读取数据: {data_file}")
    payload = parse_dataset(data_file)
    print(
        f"解析完成: diseases={len(payload.diseases)}, relationships={len(payload.relationships)}"
    )

    import_to_neo4j(payload, clear=args.clear)


if __name__ == "__main__":
    main()

