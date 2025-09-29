# agent/memory.py
import sqlite3
import json
from datetime import datetime

DB_PATH = "agent_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_query TEXT,
        agent_response TEXT,
        metadata TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_interaction(user_query, agent_response, metadata=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO interactions (timestamp, user_query, agent_response, metadata) VALUES (?,?,?,?)",
                (ts, user_query, agent_response, json.dumps(metadata or {})))
    conn.commit()
    conn.close()

def last_k_interactions(k=5):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT timestamp,user_query,agent_response,metadata FROM interactions ORDER BY id DESC LIMIT ?", (k,))
    rows = cur.fetchall()
    conn.close()
    return [{"timestamp":r[0],"user_query":r[1],"agent_response":r[2],"metadata":json.loads(r[3])} for r in rows]
