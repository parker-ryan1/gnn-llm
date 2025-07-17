import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Dict, Any
import logging

class DatabaseConnector:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self, query: str = None, table_name: str = None) -> pd.DataFrame:
        """Fetch data from database using query or table name"""
        try:
            if query:
                return pd.read_sql(query, self.engine)
            elif table_name:
                return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
            else:
                raise ValueError("Either query or table_name must be provided")
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"table_name": table_name})
            return [dict(row._mapping) for row in result]
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in database"""
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [row[0] for row in result]