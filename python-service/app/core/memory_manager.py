"""记忆管理器 - 短期记忆（Redis）+ 长期记忆（PostgreSQL）"""
from typing import List, Dict, Optional
import json
import redis
from datetime import datetime

from app.config import settings


class ShortTermMemory:
    """短期记忆 - Redis存储会话上下文"""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        effective_url = redis_url or settings.REDIS_URL
        self.client = redis.from_url(effective_url, decode_responses=True)
        self.ttl = ttl  # 默认1小时过期
        self.max_turns = 10  # 最多保留10轮对话

    def add_message(self, session_id: str, role: str, content: str):
        """添加一条消息到会话历史"""
        key = f"session:{session_id}:history"
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}

        # 添加到列表
        self.client.rpush(key, json.dumps(message))

        # 限制长度
        self.client.ltrim(key, -self.max_turns * 2, -1)

        # 刷新过期时间
        self.client.expire(key, self.ttl)

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取会话历史"""
        key = f"session:{session_id}:history"
        messages = self.client.lrange(key, -limit * 2, -1)
        return [json.loads(msg) for msg in messages]

    def clear_session(self, session_id: str):
        """清空会话"""
        key = f"session:{session_id}:history"
        self.client.delete(key)

    def set_user_context(self, session_id: str, context: Dict):
        """设置用户上下文（如用户ID、健康档案摘要）"""
        key = f"session:{session_id}:context"
        self.client.setex(key, self.ttl, json.dumps(context))

    def get_user_context(self, session_id: str) -> Optional[Dict]:
        """获取用户上下文"""
        key = f"session:{session_id}:context"
        data = self.client.get(key)
        return json.loads(data) if data else None


class LongTermMemory:
    """长期记忆 - PostgreSQL存储用户档案（简化版，使用SQLite演示）"""

    def __init__(self, db_path: str = "data/user_profiles.db"):
        import sqlite3
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 用户档案表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 健康档案表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                disease_history TEXT,
                allergy_history TEXT,
                medication_history TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """)

        # 咨询历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                intent TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_user_profile(self, user_id: str, name: str, age: int, gender: str):
        """创建用户档案"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles
            (user_id, name, age, gender, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, name, age, gender, now, now))

        conn.commit()
        conn.close()

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """获取用户档案"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "user_id": row[0],
                "name": row[1],
                "age": row[2],
                "gender": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
        return None

    def add_health_record(self, user_id: str, disease_history: str,
                         allergy_history: str, medication_history: str):
        """添加健康档案"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO health_records
            (user_id, disease_history, allergy_history, medication_history, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, disease_history, allergy_history, medication_history, now))

        conn.commit()
        conn.close()

    def get_health_record(self, user_id: str) -> Optional[Dict]:
        """获取健康档案"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM health_records
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "user_id": row[1],
                "disease_history": row[2],
                "allergy_history": row[3],
                "medication_history": row[4],
                "created_at": row[5]
            }
        return None

    def add_consultation(self, user_id: str, question: str, answer: str, intent: str):
        """记录咨询历史"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO consultation_history
            (user_id, question, answer, intent, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, question, answer, intent, now))

        conn.commit()
        conn.close()

    def get_consultation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取咨询历史"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM consultation_history
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "user_id": row[1],
                "question": row[2],
                "answer": row[3],
                "intent": row[4],
                "created_at": row[5]
            }
            for row in rows
        ]


class MemoryManager:
    """统一记忆管理器"""

    def __init__(self, redis_url: Optional[str] = None, db_path: str = "data/user_profiles.db"):
        self.short_term = ShortTermMemory(redis_url)
        self.long_term = LongTermMemory(db_path)

    def build_context_prompt(self, session_id: str, user_id: Optional[str] = None) -> str:
        """构建包含记忆的上下文提示词"""
        context_parts = []

        # 1. 短期记忆：会话历史
        try:
            history = self.short_term.get_history(session_id, limit=5)
            if history:
                context_parts.append("## 对话历史")
                for msg in history[-5:]:
                    role = "用户" if msg["role"] == "user" else "助手"
                    context_parts.append(f"{role}: {msg['content']}")
        except Exception as e:
            print(f"获取短期记忆失败（Redis可能未启动）: {e}")

        # 2. 长期记忆：用户档案
        if user_id:
            profile = self.long_term.get_user_profile(user_id)
            if profile:
                context_parts.append(f"\n## 用户信息")
                context_parts.append(f"姓名: {profile['name']}, 年龄: {profile['age']}, 性别: {profile['gender']}")

            # 健康档案
            health = self.long_term.get_health_record(user_id)
            if health:
                context_parts.append(f"\n## 健康档案")
                if health['disease_history']:
                    context_parts.append(f"既往病史: {health['disease_history']}")
                if health['allergy_history']:
                    context_parts.append(f"过敏史: {health['allergy_history']}")
                if health['medication_history']:
                    context_parts.append(f"用药史: {health['medication_history']}")

        return "\n".join(context_parts) if context_parts else ""
