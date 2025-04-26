"""
Data Loader Module

This module provides functions to load data from various sources:
- CSV files
- Excel files
- Database connections
"""

import pandas as pd
import os
import configparser
import sqlite3
from typing import Union, Optional


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(file_path, **kwargs)


def load_excel(file_path: str, sheet_name: Union[str, int, list, None] = 0, **kwargs) -> Union[pd.DataFrame, dict]:
    """
    Load data from an Excel file.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name or index of the sheet to load, or None to load all sheets
        **kwargs: Additional arguments to pass to pd.read_excel

    Returns:
        DataFrame containing the loaded data or dict of DataFrames if loading multiple sheets
    """
    return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)


def load_database(connection_string: Optional[str] = None, config_file: Optional[str] = None, 
                  query: str = "SELECT * FROM main_table") -> pd.DataFrame:
    """
    Load data from a database.

    Args:
        connection_string: Database connection string. If None, will use config_file.
        config_file: Path to database configuration file. Default is "config/database.ini".
        query: SQL query to execute. Default selects all from "main_table".

    Returns:
        DataFrame containing the query results
    """
    if connection_string is None:
        if config_file is None:
            config_file = os.path.join("config", "database.ini")
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        db_section = config["database"]
        db_type = db_section.get("type", "sqlite")
        
        if db_type.lower() == "sqlite":
            db_path = db_section.get("path", "data/database.db")
            connection_string = db_path
    
    # Currently only supporting SQLite, but can be expanded
    conn = sqlite3.connect(connection_string)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df 