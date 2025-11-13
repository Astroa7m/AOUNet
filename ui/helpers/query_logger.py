import os
from typing import Optional

import supabase
from dotenv import load_dotenv


class QueryLogger:
    """Async-safe logger for user queries"""
    TABLE_NAME = "query_logs"

    def __init__(self):
        """
        Initialize logger with Supabase PostgreSQL client
        Args:
        """
        load_dotenv()
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("supabase url and key are required")

        self.client = supabase.create_client(self.url, self.key)

    def log_query(
            self,
            query_text: str = None,
            id: int = None,
            ip_address: Optional[str] = None,
            user_agent: Optional[str] = None,
            error_message: Optional[str] = None,
            response: Optional[str] = None,
    ):
        payload = {
            k: v for k, v in {
                "id": id,
                "query_text": query_text,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "error_message": error_message,
                "response": response
            }.items() if v is not None}

        return self.client.table(self.TABLE_NAME).upsert(
            payload
        ).execute()
