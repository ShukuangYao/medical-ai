"""ElasticSearch客户端 - BM25全文检索"""
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from app.config import settings


class StoreElasticSearchClient:
    """ElasticSearch客户端，通过关键词匹配进行精确检索
    适合处理包含专业术语、药品名称、疾病编码等需要精确匹配的问题
    """

    def __init__(self):
        self.index_name = settings.ES_INDEX
        self.client: Optional[Elasticsearch] = None

    def connect(self):
        """连接ElasticSearch"""
        self.client = Elasticsearch(
            f"http://{settings.ES_HOST}:{settings.ES_PORT}",
            request_timeout=30
        )
        if self.client.ping():
            print(f"已连接ElasticSearch: {settings.ES_HOST}:{settings.ES_PORT}")
        else:
            raise ConnectionError("无法连接ElasticSearch")

    def create_index(self):
        """创建索引"""
        if not self.client:
            raise RuntimeError("ES未连接")

        if self.client.indices.exists(index=self.index_name):
            print(f"索引 {self.index_name} 已存在")
            return

        '''
        Elasticsearch 索引的映射（mappings）和索引设置（settings），也就是：每条文档有哪些字段、字段怎么分词/索引、以及分片策略等。
        mappings：定义每个字段的类型与分词方式
          id: keyword
            keyword：把字符串当作“不可分词的整体”存储。
            用途：你在插入时用 "_id": doc["id"]，这对精确匹配、去重都更合适。
          text: text
            text：会被分词后建立倒排索引，用于全文检索（BM25）。
            analyzer: "ik_max_word"：索引时用的分词器（更“细/更多词”那种思路）。
            search_analyzer: "ik_smart"：搜索时用的分词器（更“少/更精简”那种思路）。
            这是一种常见技巧：索引更细，查询更精，能提升召回与精度的平衡。
          title: text
            和 text 一样：也是全文检索字段。
            在你的 search() 里会用 fields: ["text^2", "title^3"]，所以 title 的权重更高（^3）。
          source: keyword
            仍然是不分词的字段（用于过滤/精确匹配更合适）。
          page: integer
            页码是数值字段，类型设为整数，便于数值过滤/统计（虽然你当前检索代码没用到数值过滤）。
        settings：设置分片数量 + 定义自定义 analyzer
          number_of_shards: 1
            只有 1 个分片（开发/小数据很合适，生产可按规模调整）。
          number_of_replicas: 0
            没有副本分片（同样适合本地演示）。
          analysis.analyzer
            这里定义了两个自定义 analyzer：ik_max_word 和 ik_smart。
            注意：它们都用的是 tokenizer: "standard"。
            也就是说：虽然名字叫 ik_*，但实际并没有用到 IK 分词器（IK tokenizer 应该需要额外的分析插件/配置）
            因此这两个 analyzer 目前在行为上可能非常接近（都是 standard 分词），差别主要就来自“索引 analyzer vs search analyzer”这个机制，而不是“max_word vs smart”的分词策略本身。
        '''
        mappings = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        # 让 analyzer 真正使用 IK tokenizer
                        "ik_max_word": {"type": "custom", "tokenizer": "ik_max_word"},
                        "ik_smart": {"type": "custom", "tokenizer": "ik_smart"}
                    }
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=mappings)
        print(f"索引 {self.index_name} 创建完成")

    def drop_index(self):
        """删除索引（用于重跑时清空旧数据）"""
        if not self.client:
            raise RuntimeError("ES未连接")
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"索引 {self.index_name} 已删除")
        else:
            print(f"索引 {self.index_name} 不存在，无需删除")

    def insert(self, docs: List[Dict]):
        """批量插入文档"""
        if not self.client:
            raise RuntimeError("ES未连接")

        actions = []
        for doc in docs:
            actions.append({"index": {"_index": self.index_name, "_id": doc["id"]}})
            actions.append({
                "id": doc["id"],
                "text": doc["text"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "page": doc.get("page", 0),
            })

        if actions:
            self.client.bulk(body=actions, refresh=True)
            print(f"已插入 {len(docs)} 条文档到ES")

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """BM25关键词检索"""
        k = top_k or settings.ES_TOP_K
        if not self.client:
            raise RuntimeError("ES未连接")

        # 防止 query 过长导致 ES 解析成过多子句（触发 maxClauseCount）。
        # multi_match 会把 query 分词后在多个字段构造 should 子句，长文本（如整段病历）很容易爆。
        query = (query or "").strip()
        if len(query) > 256:
            query = query[:256]

        '''
        fields: ["text^2", "title^3"]：title 影响更大
        type: "best_fields"：在多个字段里取最好匹配
        minimum_should_match: "30%"：要求匹配的词占比达到一定程度
        所以这段 mappings 的核心意义是：把 text/title 做成“可全文检索的 text 字段”，其分词方式决定了 BM25 的效果。
        '''
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "title^3"],
                    "type": "best_fields",
                    "minimum_should_match": "30%"
                }
            },
            "size": k
        }

        response = self.client.search(index=self.index_name, body=body)

        docs = []
        for hit in response["hits"]["hits"]:
            docs.append({
                "id": hit["_id"],
                "text": hit["_source"].get("text", ""),
                "title": hit["_source"].get("title", ""),
                "source": hit["_source"].get("source", ""),
                "page": hit["_source"].get("page", 0),
                "score": hit["_score"],
            })
        return docs

    def count(self) -> int:
        """返回索引中的文档数量"""
        if self.client and self.client.indices.exists(index=self.index_name):
            return self.client.count(index=self.index_name)["count"]
        return 0
