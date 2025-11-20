"""
pc_script12.py (PA Call Analyzer version)
-----------------------------------------
Provides two CLI commands:
  python pa_script12.py init_db        # ensure pa_ref_table exists
  python pa_script12.py load_base_ref  # insert/update canonical PA scripts (inbound & outbound)

Environment variables:
  DATABASE_URL - SQLAlchemy DB URL (optional; fallback used)
"""

import os  # to access environment variables and filesystem operations
import time  # small sleeps if needed (kept for potential future use)
import argparse  # for simple CLI argument parsing
from typing import Optional, List, Dict  # type hints used to clarify signatures
from dotenv import load_dotenv  # loads .env file values into environment if present

# load environment variables from a .env file if present (safe no-op otherwise)
load_dotenv()  # this populates os.environ with values from .env when available

# SQLAlchemy imports to create an engine and execute SQL safely
from sqlalchemy import create_engine, text  # create DB engine and write parameterized SQL
from sqlalchemy.engine import Engine  # typed return for get_engine()

# Configuration: read DATABASE_URL from env with a sensible fallback for local/dev
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://qispineadmin:TrOpSnl1H1QdKAFsAWnY@qispine-db.cqjl02ffrczp.ap-south-1.rds.amazonaws.com:5432/qed_prod"
)  # database connection string; replace or override via env var in production

# ---------- Database utility functions ----------

def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine configured for DATABASE_URL."""  # docstring
    return create_engine(
        DATABASE_URL,  # connection string
        pool_pre_ping=True,  # check connection liveness before using a connection
        pool_size=5,  # number of persistent connections in the pool
        max_overflow=5  # maximum connections above pool_size when needed
    )  # returns SQLAlchemy Engine instance

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Execute a SQL statement that doesn't return rows (INSERT/UPDATE/DDL)."""  # docstring
    with engine.begin() as conn:  # begin a transaction and yield a connection
        conn.execute(text(sql), params or {})  # execute SQL with given params or empty dict

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> list:
    """Execute a SELECT query and return rows as list of dicts."""  # docstring
    with engine.begin() as conn:  # transactional connection for safe reads
        result = conn.execute(text(sql), params or {})  # execute parameterized SELECT
        return [dict(row._mapping) for row in result.fetchall()]  # convert rows to list of dicts

# ---------- DDL: create pa_ref_table ----------
# Note: table has 'id' primary key and unique(script_code, call_type_io) to allow two rows per script_code

DDL_PA_REF_TABLE = """
CREATE TABLE IF NOT EXISTS pa_ref_table (
    id BIGSERIAL PRIMARY KEY,                           -- surrogate primary key
    script_code TEXT NOT NULL,                          -- identifier for script (e.g., PA_Script001)
    og_pa_script TEXT NOT NULL,                         -- canonical script text
    call_type_io TEXT NOT NULL,                         -- 'inbound' or 'outbound'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),  -- timestamp of insertion
    CONSTRAINT pa_unique_script_calltype UNIQUE (script_code, call_type_io)  -- prevent accidental duplicates
);
"""  # DDL to create the table if it doesn't exist

def init_db():
    """Create / ensure pa_ref_table exists in the database."""  # docstring
    engine = get_engine()  # get DB engine instance
    run_sql(engine, DDL_PA_REF_TABLE)  # execute DDL to create table
    print(" pa_ref_table ensured/created successfully.")  # user feedback

# ---------- Canonical PA scripts: inbound & outbound texts ----------
# Texts provided by user; stored as multiline strings for insertion into DB

PA_INBOUND_SCRIPT = """Hi this is Albert from QI Spine Clinic. How can I help you today? Before we proceed, I would like to inform you that this call is being recorded for Training and Quality purposes. Sir or Madam, May I know your name and age please.
Which city you’re from?
What is your pin code?
Are you able to do your daily activities without any support?
Does your job require you to stand for a long time?
What would you rate the pain on scale of 0 to 10, where 10 being highest 0 being lowest.
How Long are you suffering from this pain?
Have you shown it to any doctor before?
Have you gone through any Tests like Xray or MRI?
I can understand what you are going through
Rest assured you have reached the right place; we are here to help you.
Let me tell you in brief about who we are.
We are an 18-year-old Ortho Specialty Clinic with team of Ortho Specialist, Spine Specialist and Neuro Specialist capable of taking multidisciplinary approach to cure your spine, neck, back and joint related issues. We help patients in healing their pain without needing surgeries or medication. With our advanced technology and accurate diagnosis through DSA and CRT treatment, we helped more than 2 Lakhs + patients and avoided 16 thousand + surgeries. We have an industry best success rate of 93% in patient recoveries.
We provide Free Consultation, usually for 40 to 45 mins where our Spine Specialist will check for your medical movements and would identify the root cause of your pain and would suggest accurate line of treatment based on the diagnosis.
Sir or Madam, we have limited slots available with Senior Spine Specialist for the day.
As I could see we have slots available at X and Y PM. (immediate next slots available in next 3 hours). 
May I Book the X or Y PM slot for you.
 
If yes
I have booked your appointment at X PM with Dr. Y. You will receive a message with Clinic location and a call of Confirmation before your session.
 
If No
Sir or Madam, I can manage one more slot which got cancelled at Z PM. But, please make sure you reach on time.
If Again No
Sir or Madam, we will push for next available slots for Next Day and Subsequent days.
I have booked your appointment at X PM with Dr. Y. You will receive a message with Clinic location and a call of Confirmation before your session.
May I help you with anything else?
Thank you so much for your valuable time. Have a great day.
"""  # inbound script string

PA_OUTBOUND_SCRIPT = """Hi this is Albert from QI Spine Clinic. We have received a request regarding your Back/Neck/Joints pain related issues. May I know who is the concerned patient and what’s the issue? Before we proceed, I would like to inform you that this call is being recorded for Training and Quality purposes.Sir / Madam, May I know your name and age please.
Which city you’re from?
What is your pin code?
Are you able to do your daily activities without any support?
Does your job require you to stand for a long time?
What would you rate the pain on scale of 0-10, 10 being highest 0 being lowest.
How Long are you suffering from this pain?
Have you shown it to any doctor before?
Have you gone through any Tests like Xray or MRI?
I can understand what you are going through
Rest assured you have reached the right place; we are here to help you.
Let me tell you in brief about who we are.
We are an 18-year-old Ortho Speciality Clinic with team of Ortho Specialist, Spine Specialist and Neuro Specialist capable of taking multidisciplinary approach to cure your spine, neck, back and joint related issues. We help patients in healing their pain without needing surgeries or medication. With our advanced technology and accurate diagnosis through DSA and CRT treatment, we helped more than 2 Lakhs + patients and avoided 16 thousand + surgeries. We have an industry best success rate of 93% in patient recoveries.
We provide Free Consultation, usually for 40 to 45 mins where our Spine Specialist will check for your medical movements and would identify the root cause of your pain and would suggest accurate line of treatment based on the diagnosis.
Sir / Madam, we have limited slots available with Senior Spine Specialist for the day.
As I could see we have slots available at X and Y PM. (~ immediate next slots available in next 3 hours). 
May I Book the X / Y PM slot for you.
 
If yes
I have booked your appointment at X PM with Dr. Y. You will receive a message with Clinic location and a call of Confirmation before your session.
 
If No
Sir / Madam, I can manage one more slot which got cancelled at Z PM. But, please make sure you reach on time.
If Again No
Sir / Madam, we will push for next available slots for Next Day and Subsequent days.
I have booked your appointment at X PM with Dr. Y. You will receive a message with Clinic location and a call of Confirmation before your session.
May I help you with anything else?
Thank you so much for your valuable time. Have a great day.
"""  # outbound script string

# ---------- insert/update canonical PA scripts into pa_ref_table ----------

def load_base_ref():
    """Insert or update canonical PA scripts under script_code 'PA_Script001'."""  # docstring
    init_db()  # ensure table/schema exists before inserting rows
    engine = get_engine()  # obtain DB engine for queries

    # Upsert for inbound row: uses unique(script_code, call_type_io) constraint to avoid duplicates
    upsert_inbound = """
    INSERT INTO pa_ref_table (script_code, og_pa_script, call_type_io, created_at)
    VALUES (:script_code, :og_pa_script, :call_type_io, now())
    ON CONFLICT (script_code, call_type_io) DO UPDATE
      SET og_pa_script = EXCLUDED.og_pa_script,
          created_at = EXCLUDED.created_at;
    """  # SQL: insert inbound row or update existing one

    # Upsert for outbound row: same pattern for outbound calltype
    upsert_outbound = """
    INSERT INTO pa_ref_table (script_code, og_pa_script, call_type_io, created_at)
    VALUES (:script_code, :og_pa_script, :call_type_io, now())
    ON CONFLICT (script_code, call_type_io) DO UPDATE
      SET og_pa_script = EXCLUDED.og_pa_script,
          created_at = EXCLUDED.created_at;
    """  # SQL: insert outbound row or update existing one

    # Execute inbound upsert with parameters
    run_sql(engine, upsert_inbound, {
        "script_code": "PA_Script001",  # canonical script code for PA project
        "og_pa_script": PA_INBOUND_SCRIPT,  # inbound script content
        "call_type_io": "inbound"  # inbound indicator
    })

    # small sleep for clarity (not strictly necessary)
    time.sleep(0.05)  # brief pause to separate DB operations

    # Execute outbound upsert with parameters
    run_sql(engine, upsert_outbound, {
        "script_code": "PA_Script001",  # same script code reused with different call_type_io
        "og_pa_script": PA_OUTBOUND_SCRIPT,  # outbound script content
        "call_type_io": "outbound"  # outbound indicator
    })

    print(" PA Script rows (inbound & outbound) inserted/updated into pa_ref_table under PA_Script001.")  # feedback

# ---------- simple CLI to expose init_db and load_base_ref ----------

def main():
    """Command-line interface entry point to dispatch commands."""  # docstring
    parser = argparse.ArgumentParser(description="PA Call Analyzer - DB initialization & base reference loader")  # create argparser
    subparsers = parser.add_subparsers(dest="command", help="Available commands")  # create subcommands

    # expose init_db and load_base_ref commands to CLI users
    subparsers.add_parser("init_db", help="Initialize pa_ref_table")  # init_db CLI entry
    subparsers.add_parser("load_base_ref", help="Load PA canonical scripts (inbound & outbound)")  # load_base_ref CLI entry

    args = parser.parse_args()  # parse arguments provided on command line

    # dispatch based on selected command
    if args.command == "init_db":  # if user asked to init DB
        init_db()  # call function to ensure table exists
    elif args.command == "load_base_ref":  # if user asked to load base refs
        load_base_ref()  # call function to insert/update canonical scripts
    else:  # if no or unknown command provided
        parser.print_help()  # print CLI help for user guidance

# standard Python module guard to run main() when script executed directly
if __name__ == "__main__":
    main()
