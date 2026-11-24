# import json
# import math
# import os
# import re
# from contextlib import asynccontextmanager
# from typing import Literal, Optional

# from urllib.parse import unquote, urlparse

# import pg8000.dbapi  # pure Python driver, no DLL needed
# from fastapi import FastAPI, HTTPException
# from openai import OpenAI
# from pydantic import BaseModel, Field

# # Plain postgres URL, e.g. postgresql://user:pass@host:6543/postgres
# DATABASE_URL = os.environ["DATABASE_URL"]
# LLM_MODEL = os.environ["LLM_MODEL"]
# # Must be the same model that produced content_chunks.embedding
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# LOW_GROUNDING = 0.30

# # Table names (change here if yours are class/book instead of classes/books)
# T_CLASS, T_BOOK = "classes", "books"

# state = {}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     state["llm"] = OpenAI()
#     yield


# app = FastAPI(title="Paper Generator", lifespan=lifespan)


# # ---------------- DB ----------------

# class PgConn:
#     """Thin wrapper so the rest of the code can use conn.execute(sql, params)
#     with either %s tuples or %(name)s dicts."""

#     def __init__(self, raw):
#         self.raw = raw

#     def execute(self, sql, params=None):
#         cur = self.raw.cursor()
#         if isinstance(params, dict):
#             names = re.findall(r"%\((\w+)\)s", sql)
#             sql = re.sub(r"%\(\w+\)s", "%s", sql)
#             params = [params[n] for n in names]
#         cur.execute(sql, params or ())
#         return cur

#     def __enter__(self):
#         return self

#     def __exit__(self, *exc):
#         self.raw.close()


# def get_conn():
#     u = urlparse(DATABASE_URL)
#     raw = pg8000.dbapi.connect(
#         user=unquote(u.username),
#         password=unquote(u.password or ""),
#         host=u.hostname,
#         port=u.port or 5432,
#         database=(u.path or "/postgres").lstrip("/") or "postgres",
#         ssl_context=True,  # Supabase requires SSL
#         timeout=15,
#     )
#     return PgConn(raw)


# # ---------------- Embeddings (no numpy) ----------------

# def normalize_vector(vector):
#     vector = list(vector)
#     magnitude = math.sqrt(sum(v * v for v in vector))
#     if magnitude == 0:
#         return vector
#     return [v / magnitude for v in vector]


# def embed(texts):
#     response = state["llm"].embeddings.create(model=EMBED_MODEL, input=texts)
#     return [normalize_vector(item.embedding) for item in response.data]


# def to_pgvector(vec):
#     """Python list -> '[0.1,0.2,...]' literal accepted by ::vector"""
#     return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# # ---------------- Request Schema ----------------

# class PaperRequest(BaseModel):
#     class_name: str = Field(examples=["Class 9"])
#     book_name: str = Field(examples=["Physics"])
#     topics: list[str] = Field(min_length=1, examples=[["1.1", "1.2", "1.14"]])
#     difficulty: Literal["easy", "medium", "complex"] = "medium"
#     mcq: int = Field(0, ge=0, le=40)
#     short: int = Field(0, ge=0, le=30)
#     long: int = Field(0, ge=0, le=10)
#     marks_mcq: int = Field(1, ge=1)
#     marks_short: int = Field(2, ge=1)
#     marks_long: int = Field(5, ge=1)
#     focus: Optional[str] = Field(None, description="Optional, e.g. numerical problems")
#     top_k: int = Field(5, ge=1, le=10)
#     min_similarity: float = Field(0.25, ge=0, le=1)


# # ---------------- RAG: DB lookup + similarity search ----------------

# def retrieve(conn, req: PaperRequest):
#     row = conn.execute(
#         f"""
#         SELECT b.id
#         FROM {T_BOOK} b
#         JOIN {T_CLASS} c ON c.id = b.class_id
#         WHERE lower(c.name) = lower(%s)
#           AND lower(b.name) = lower(%s)
#         """,
#         (req.class_name.strip(), req.book_name.strip()),
#     ).fetchone()

#     if not row:
#         raise HTTPException(404, f"No book '{req.book_name}' found for '{req.class_name}'")

#     book_id = row[0]
#     wanted = [t.strip() for t in req.topics]

#     topics = conn.execute(
#         """
#         SELECT t.id, t.topic_number::text, t.name
#         FROM topics t
#         JOIN chapters ch ON ch.id = t.chapter_id
#         WHERE ch.book_id = %s
#           AND t.topic_number::text = ANY(%s::text[])
#         ORDER BY ch.chapter_number, t.sort_order
#         """,
#         (book_id, wanted),
#     ).fetchall()

#     missing = set(wanted) - {t[1] for t in topics}
#     if missing:
#         raise HTTPException(400, f"Topics not found in this book: {sorted(missing)}")

#     # one batched embedding call for all topics
#     queries = [f"{name} {req.focus or 'definition law formula solved example'}" for _, _, name in topics]
#     query_vecs = embed(queries)

#     ctx = []
#     for (tid, number, name), qvec in zip(topics, query_vecs):
#         hits = conn.execute(
#             """
#             SELECT page_number,
#                    content,
#                    embedding::text,
#                    1 - (embedding <=> %(q)s::vector) AS sim
#             FROM content_chunks
#             WHERE topic_id = %(tid)s
#               AND embedding IS NOT NULL
#             ORDER BY embedding <=> %(q)s::vector
#             LIMIT %(k)s
#             """,
#             {"q": to_pgvector(qvec), "tid": tid, "k": req.top_k},
#         ).fetchall()

#         # always keep the 2 best chunks, others must pass the threshold
#         hits = [h for i, h in enumerate(hits) if i < 2 or h[3] >= req.min_similarity]

#         if not hits:
#             raise HTTPException(422, f"Topic {number} {name} has no stored content yet")

#         ctx.append({
#             "number": number,
#             "name": name,
#             "chunks": [
#                 {"page": p, "text": c, "similarity": round(float(s), 3)}
#                 for p, c, _, s in hits
#             ],
#             # pgvector text format "[a,b,c]" is valid JSON
#             "vecs": [json.loads(e) for _, _, e, _ in hits],
#         })

#     return ctx


# # ---------------- Question distribution ----------------

# def split_counts(total, keys):
#     base, extra = divmod(total, len(keys))
#     return {k: base + (1 if i < extra else 0) for i, k in enumerate(keys)}


# # ---------------- Prompt ----------------

# def build_prompt(ctx, req, plan, problems):
#     blocks = []
#     for t in ctx:
#         body = "\n---\n".join(f"[p.{c['page']}] {c['text']}" for c in t["chunks"])
#         blocks.append(f"### TOPIC {t['number']} {t['name']}\n{body}")

#     fix = (
#         "\nYour previous attempt had these problems. Fix all of them:\n- "
#         + "\n- ".join(problems)
#     ) if problems else ""

#     return f"""
# You are an examiner for {req.class_name} {req.book_name} (Punjab textbook).

# Write a question paper ONLY from the textbook excerpts below.
# The excerpts come from OCR and contain errors: use them for facts
# and repair obvious OCR damage only when you are certain.

# Do not use outside facts.
# Do not copy sentences word for word.
# Write original questions.

# Excerpts from an exercise section show the style of questions
# the board asks; use them as a style guide.

# Difficulty: {req.difficulty}

# Number of questions per topic: {json.dumps(plan)}

# Rules:

# - MCQ: exactly 4 distinct options, exactly one correct,
#   3 plausible distractors, no "all of the above".
# - Short: answerable in 2-3 lines.
# - Long: multi-part, derivation or numerical, with a full model answer.
# - Numerical questions: show every step with units and double-check arithmetic.
# - No repeated or near-duplicate questions.
# - "topic" must be the topic number exactly as given.

# Return ONLY a JSON object, no markdown:

# {{
#     "mcq": [
#         {{"topic": "1.3", "question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "..."}}
#     ],
#     "short": [
#         {{"topic": "1.3", "question": "...", "answer": "..."}}
#     ],
#     "long": [
#         {{"topic": "1.3", "question": "...", "answer": "..."}}
#     ]
# }}

# {fix}

# TEXTBOOK EXCERPTS:

# {chr(10).join(blocks)}
# """


# # ---------------- LLM ----------------

# def call_llm(prompt):
#     try:
#         resp = state["llm"].chat.completions.create(
#             model=LLM_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#         )
#     except Exception as e:
#         raise HTTPException(502, f"LLM error: {e}")
#     return json.loads(resp.choices[0].message.content)


# # ---------------- Validation ----------------

# def validate(paper, allowed, req):
#     if not isinstance(paper, dict):
#         return ["Reply was not a JSON object"]

#     problems = []
#     seen = set()

#     for key, n in (("mcq", req.mcq), ("short", req.short), ("long", req.long)):
#         items = paper.get(key, [])

#         if not isinstance(items, list):
#             problems.append(f"{key}: expected a list")
#             continue

#         if len(items) != n:
#             problems.append(f"{key}: expected {n} questions, got {len(items)}")

#         for q in items:
#             if not isinstance(q, dict):
#                 problems.append(f"{key}: invalid question object")
#                 continue

#             if str(q.get("topic")) not in allowed:
#                 problems.append(f"{key}: topic {q.get('topic')} was not selected")

#             norm = re.sub(r"\W+", " ", str(q.get("question", "")).lower()).strip()
#             if not norm or norm in seen:
#                 problems.append(f"{key}: empty or duplicate question")
#             seen.add(norm)

#             if key == "mcq":
#                 opts = q.get("options", [])
#                 if not isinstance(opts, list) or len(opts) != 4 or len(set(map(str, opts))) != 4:
#                     problems.append(f"mcq needs 4 distinct options: {str(q.get('question'))[:50]}")
#                 if q.get("answer_index") not in (0, 1, 2, 3):
#                     problems.append(f"mcq has no valid answer_index: {str(q.get('question'))[:50]}")

#     return problems


# # ---------------- Grounding check ----------------

# def cosine_similarity(a, b):
#     dot = sum(x * y for x, y in zip(a, b))
#     ma = math.sqrt(sum(x * x for x in a))
#     mb = math.sqrt(sum(y * y for y in b))
#     if ma == 0 or mb == 0:
#         return 0.0
#     return dot / (ma * mb)


# def add_grounding(paper, ctx):
#     by_topic = {t["number"]: t["vecs"] for t in ctx}

#     items = [
#         (k, q)
#         for k in ("mcq", "short", "long")
#         for q in paper.get(k, [])
#         if isinstance(q, dict) and str(q.get("topic")) in by_topic
#     ]
#     if not items:
#         return []

#     vecs = embed([q["question"] for _, q in items])
#     flagged = []

#     for (key, q), v in zip(items, vecs):
#         grounding = max(cosine_similarity(sv, v) for sv in by_topic[str(q["topic"])])
#         q["grounding"] = round(float(grounding), 3)

#         if grounding < LOW_GROUNDING:
#             flagged.append(
#                 f"{key} (topic {q['topic']}) may not follow the book "
#                 f"[grounding {q['grounding']}]: {q['question'][:70]}"
#             )

#     return flagged


# # ---------------- Endpoint ----------------

# @app.post("/generate-paper")
# def generate_paper(req: PaperRequest):
#     if req.mcq + req.short + req.long == 0:
#         raise HTTPException(422, "Ask for at least one question")

#     with get_conn() as conn:
#         ctx = retrieve(conn, req)

#     numbers = [t["number"] for t in ctx]

#     plan = {
#         "mcq": split_counts(req.mcq, numbers),
#         "short": split_counts(req.short, numbers),
#         "long": split_counts(req.long, numbers),
#     }

#     paper = None
#     problems = []

#     for _ in range(3):
#         try:
#             paper = call_llm(build_prompt(ctx, req, plan, problems))
#         except json.JSONDecodeError:
#             paper = None
#             problems = ["Your previous reply was not valid JSON. Return ONLY the JSON object."]
#             continue

#         problems = validate(paper, set(numbers), req)
#         if not problems:
#             break

#     if paper is None:
#         raise HTTPException(502, "The model did not return a valid paper")

#     warnings = problems + add_grounding(paper, ctx)

#     return {
#         "class": req.class_name,
#         "book": req.book_name,
#         "difficulty": req.difficulty,
#         "total_marks": (
#             req.mcq * req.marks_mcq + req.short * req.marks_short + req.long * req.marks_long
#         ),
#         "paper": paper,
#         "sources": [
#             {
#                 "topic": t["number"],
#                 "name": t["name"],
#                 "chunks": [{"page": c["page"], "similarity": c["similarity"]} for c in t["chunks"]],
#             }
#             for t in ctx
#         ],
#         "warnings": warnings,
#     }



#new code with grok
# import time
# import random
# import json
# import math
# import os
# import re

# from contextlib import asynccontextmanager
# from typing import Literal, Optional
# from urllib.parse import unquote, urlparse

# from io import BytesIO
# from xml.sax.saxutils import escape

# import pg8000.dbapi

# from fastapi import FastAPI, HTTPException, Response
# from pydantic import BaseModel, Field

# from reportlab.lib import colors
# from reportlab.lib.enums import TA_CENTER
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import ParagraphStyle
# from reportlab.lib.units import mm
# from reportlab.pdfbase import pdfmetrics
# from reportlab.pdfbase.ttfonts import TTFont
# from reportlab.platypus import (
#     HRFlowable,
#     KeepTogether,
#     PageBreak,
#     Paragraph,
#     SimpleDocTemplate,
#     Spacer,
#     Table,
#     TableStyle,
# )

# from openai import OpenAI
# from google import genai
# from google.genai import types


# # ---------------- CONFIGURATION ----------------

# DATABASE_URL = os.environ["DATABASE_URL"]

# # Groq is used for question generation (OpenAI-compatible API)
# GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# # Gemini is used ONLY for embeddings
# EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# # Must match the vector(N) column size in content_chunks
# EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))

# LOW_GROUNDING = 0.30

# T_CLASS = "classes"
# T_BOOK = "books"

# state = {}


# # ---------------- LIFESPAN ----------------

# @asynccontextmanager
# async def lifespan(app: FastAPI):

#     groq_api_key = os.getenv("GROQ_API_KEY")
#     gemini_api_key = os.getenv("GEMINI_API_KEY")

#     if not groq_api_key:
#         raise RuntimeError("GROQ_API_KEY is missing from environment variables")

#     if not gemini_api_key:
#         raise RuntimeError("GEMINI_API_KEY is required for embeddings")

#     # Groq client for question generation
#     state["llm"] = OpenAI(
#         api_key=groq_api_key,
#         base_url=GROQ_BASE_URL,
#         timeout=300.0,
#     )

#     # Gemini client for embeddings only
#     state["embed_llm"] = genai.Client(api_key=gemini_api_key)

#     print("Groq and Gemini embedding clients initialized")
#     print(f"Generation model: {GROQ_MODEL}")
#     print(f"Embedding model: {EMBED_MODEL}")
#     print(f"Embedding dimensions: {EMBED_DIMENSIONS}")

#     yield

#     try:
#         state["llm"].close()
#     except Exception:
#         pass

#     try:
#         state["embed_llm"].close()
#     except Exception:
#         pass

#     state.clear()


# app = FastAPI(
#     title="RAG Question Paper Generator",
#     description="Generate textbook-grounded question papers using Groq and PostgreSQL",
#     lifespan=lifespan,
# )


# # ---------------- DATABASE ----------------

# class PgConn:

#     def __init__(self, raw):
#         self.raw = raw

#     def execute(self, sql, params=None):

#         cur = self.raw.cursor()

#         if isinstance(params, dict):

#             names = re.findall(r"%\((\w+)\)s", sql)
#             sql = re.sub(r"%\(\w+\)s", "%s", sql)
#             params = [params[n] for n in names]

#         cur.execute(sql, params or ())

#         return cur

#     def __enter__(self):
#         return self

#     def __exit__(self, *exc):
#         self.raw.close()


# def get_conn():

#     u = urlparse(DATABASE_URL)

#     raw = pg8000.dbapi.connect(
#         user=unquote(u.username),
#         password=unquote(u.password or ""),
#         host=u.hostname,
#         port=u.port or 5432,
#         database=(u.path or "/postgres").lstrip("/") or "postgres",
#         ssl_context=True,
#         timeout=15,
#     )

#     return PgConn(raw)


# # ---------------- VECTOR UTILITIES ----------------

# def normalize_vector(vector):

#     vector = list(vector)

#     magnitude = math.sqrt(sum(v * v for v in vector))

#     if magnitude == 0:
#         return vector

#     return [v / magnitude for v in vector]


# def embed(texts, task_type="RETRIEVAL_QUERY"):

#     if not texts:
#         return []

#     try:

#         response = state["embed_llm"].models.embed_content(
#             model=EMBED_MODEL,
#             contents=texts,
#             config=types.EmbedContentConfig(
#                 task_type=task_type,
#                 output_dimensionality=EMBED_DIMENSIONS,
#             ),
#         )

#         if not response.embeddings:
#             raise ValueError("Embedding provider returned no embeddings")

#         vectors = [normalize_vector(item.values) for item in response.embeddings]

#         if len(vectors) != len(texts):
#             raise ValueError("Embedding count does not match input text count")

#         return vectors

#     except Exception as e:

#         raise HTTPException(
#             status_code=502,
#             detail=f"Embedding error: {str(e)}",
#         )


# def to_pgvector(vec):

#     return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# # ---------------- REQUEST SCHEMA ----------------

# class PaperRequest(BaseModel):

#     class_name: str = Field(examples=["Class 9"])

#     book_name: str = Field(examples=["Physics"])

#     topics: list[str] = Field(min_length=1, examples=[["1.1", "1.2", "1.14"]])

#     difficulty: Literal["easy", "medium", "complex"] = "medium"

#     mcq: int = Field(0, ge=0, le=40)

#     short: int = Field(0, ge=0, le=30)

#     long: int = Field(0, ge=0, le=10)

#     marks_mcq: int = Field(1, ge=1)

#     marks_short: int = Field(2, ge=1)

#     marks_long: int = Field(5, ge=1)

#     focus: Optional[str] = Field(
#         None,
#         description="Optional focus, e.g. numerical problems",
#     )

#     top_k: int = Field(5, ge=1, le=10)

#     min_similarity: float = Field(0.25, ge=0, le=1)

#     # PDF options
#     institute_name: Optional[str] = Field(
#         None,
#         description="Printed at the top of the paper",
#         examples=["ABC Public School"],
#     )

#     time_allowed: Optional[str] = Field(
#         None,
#         description="Printed in the paper header",
#         examples=["1 hour 30 minutes"],
#     )

#     include_answer_key: bool = Field(
#         True,
#         description="Add an answer key and marking scheme on separate pages at the end",
#     )


# # ---------------- RAG RETRIEVAL ----------------

# def retrieve(conn, req: PaperRequest):

#     # STEP 1: Find the requested book

#     row = conn.execute(
#         f"""
#         SELECT b.id
#         FROM {T_BOOK} b
#         JOIN {T_CLASS} c
#             ON c.id = b.class_id
#         WHERE lower(c.name) = lower(%s)
#           AND lower(b.name) = lower(%s)
#         """,
#         (
#             req.class_name.strip(),
#             req.book_name.strip(),
#         ),
#     ).fetchone()

#     if not row:
#         raise HTTPException(
#             404,
#             f"No book '{req.book_name}' found for '{req.class_name}'",
#         )

#     book_id = row[0]

#     wanted = [t.strip() for t in req.topics]

#     # STEP 2: Find requested topics

#     topics = conn.execute(
#         """
#         SELECT
#             t.id,
#             t.topic_number::text,
#             t.name
#         FROM topics t
#         JOIN chapters ch
#             ON ch.id = t.chapter_id
#         WHERE ch.book_id = %s
#           AND t.topic_number::text = ANY(%s::text[])
#         ORDER BY
#             ch.chapter_number,
#             t.sort_order
#         """,
#         (book_id, wanted),
#     ).fetchall()

#     found = {t[1] for t in topics}
#     missing = set(wanted) - found

#     if missing:
#         raise HTTPException(
#             400,
#             f"Topics not found in this book: {sorted(missing)}",
#         )

#     # STEP 3: Create query embeddings

#     queries = [
#         f"{name} {req.focus or 'definition law formula solved example'}"
#         for _, _, name in topics
#     ]

#     query_vecs = embed(queries, task_type="RETRIEVAL_QUERY")

#     ctx = []

#     # STEP 4: Retrieve relevant textbook chunks

#     for (tid, number, name), qvec in zip(topics, query_vecs):

#         hits = conn.execute(
#             """
#             SELECT
#                 page_number,
#                 content,
#                 embedding::text,
#                 1 - (embedding <=> %(q)s::vector) AS sim
#             FROM content_chunks
#             WHERE topic_id = %(tid)s
#               AND embedding IS NOT NULL
#             ORDER BY embedding <=> %(q)s::vector
#             LIMIT %(k)s
#             """,
#             {
#                 "q": to_pgvector(qvec),
#                 "tid": tid,
#                 "k": req.top_k,
#             },
#         ).fetchall()

#         # Always retain the two best chunks.
#         # Remaining chunks must satisfy the threshold.

#         hits = [
#             h
#             for i, h in enumerate(hits)
#             if i < 2 or float(h[3]) >= req.min_similarity
#         ]

#         if not hits:
#             raise HTTPException(
#                 422,
#                 f"Topic {number} {name} has no stored content yet",
#             )

#         ctx.append({
#             "number": number,
#             "name": name,
#             "chunks": [
#                 {
#                     "page": p,
#                     "text": content,
#                     "similarity": round(float(sim), 3),
#                 }
#                 for p, content, _, sim in hits
#             ],
#             "vecs": [
#                 json.loads(embedding)
#                 for _, _, embedding, _ in hits
#             ],
#         })

#     return ctx


# # ---------------- QUESTION DISTRIBUTION ----------------

# def split_counts(total, keys):

#     if not keys:
#         return {}

#     base, extra = divmod(total, len(keys))

#     return {
#         key: base + (1 if i < extra else 0)
#         for i, key in enumerate(keys)
#     }


# # ---------------- PROMPT BUILDER ----------------

# def build_prompt(ctx, req, plan, problems):

#     blocks = []

#     for topic in ctx:

#         body = "\n---\n".join(
#             f"[p.{chunk['page']}] {chunk['text']}"
#             for chunk in topic["chunks"]
#         )

#         blocks.append(
#             f"""
# ### TOPIC {topic['number']} {topic['name']}

# {body}
# """
#         )

#     fix = (
#         "\nYour previous attempt had these problems. Fix all of them:\n- "
#         + "\n- ".join(problems)
#     ) if problems else ""

#     return f"""
# You are an experienced examination paper setter.

# Generate a question paper for:

# Class: {req.class_name}
# Book: {req.book_name}
# Difficulty: {req.difficulty}

# IMPORTANT:
# Use ONLY the provided textbook excerpts.

# Do not introduce outside facts.
# Do not invent formulas or definitions.
# Repair obvious OCR errors only when certain.
# Write original questions instead of copying sentences.

# Use exercise questions as a style guide if they appear
# in the provided textbook excerpts.

# QUESTION DISTRIBUTION:

# {json.dumps(plan, indent=2)}

# RULES:

# 1. MCQs:
# - Exactly 4 distinct options.
# - Exactly one correct answer.
# - Three plausible distractors.
# - No "all of the above".

# 2. Short questions:
# - Answers should be 2-3 lines.
# - Test understanding of the selected topic.

# 3. Long questions:
# - Include multi-part questions where appropriate.
# - Include derivations or numerical problems when supported.
# - Provide complete model answers.

# 4. Numerical questions:
# - Show every calculation step.
# - Include correct units.
# - Double-check arithmetic.

# 5. General:
# - No repeated or near-duplicate questions.
# - The topic field must exactly match the supplied topic number.
# - Follow the requested question distribution.
# - Do not generate extra questions.

# RETURN ONLY A VALID JSON OBJECT:

# {{
#     "mcq": [
#         {{
#             "topic": "1.3",
#             "question": "Question text",
#             "options": [
#                 "Option A",
#                 "Option B",
#                 "Option C",
#                 "Option D"
#             ],
#             "answer_index": 0,
#             "explanation": "Explanation"
#         }}
#     ],

#     "short": [
#         {{
#             "topic": "1.3",
#             "question": "Question text",
#             "answer": "Answer"
#         }}
#     ],

#     "long": [
#         {{
#             "topic": "1.3",
#             "question": "Question text",
#             "answer": "Detailed answer"
#         }}
#     ]
# }}

# {fix}

# TEXTBOOK EXCERPTS:

# {chr(10).join(blocks)}
# """


# # ---------------- GROQ GENERATION ----------------

# def error_details(e):
#     """Extract the real provider error message (not just 'Error code: 400')."""

#     body = getattr(e, "body", None)

#     if body:
#         return f"{str(e)} | {body}"

#     response = getattr(e, "response", None)

#     if response is not None:
#         try:
#             return f"{str(e)} | {response.text}"
#         except Exception:
#             pass

#     return str(e)


# def call_llm(prompt):

#     max_attempts = 4

#     for attempt in range(max_attempts):

#         try:

#             response = state["llm"].chat.completions.create(
#                 model=GROQ_MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.7,
#                 top_p=0.95,
#                 max_completion_tokens=16000,
#                 extra_body={"reasoning_effort": "high"},
#             )

#             response_text = response.choices[0].message.content

#             if not response_text:
#                 raise ValueError("Groq returned an empty response")

#             return json.loads(response_text)

#         except json.JSONDecodeError as e:

#             raise HTTPException(
#                 status_code=502,
#                 detail=f"Groq returned invalid JSON: {str(e)}",
#             )

#         except Exception as e:

#             status_code = (
#                 getattr(e, "status_code", None)
#                 or getattr(e, "code", None)
#             )

#             error_text = str(e).upper()

#             is_temporary = (
#                 status_code in (429, 500, 502, 503, 504)
#                 or any(code in error_text for code in (
#                     "UNAVAILABLE",
#                     "RESOURCE_EXHAUSTED",
#                     "RATE LIMIT",
#                 ))
#             )

#             if is_temporary and attempt < max_attempts - 1:

#                 delay = (2 ** attempt) + random.uniform(0, 1)

#                 print(
#                     f"Groq temporarily unavailable. "
#                     f"Retry {attempt + 1}/{max_attempts - 1} "
#                     f"in {delay:.1f} seconds..."
#                 )

#                 time.sleep(delay)

#                 continue

#             if is_temporary:

#                 raise HTTPException(
#                     status_code=503,
#                     detail=(
#                         "Groq is currently rate limited or busy. "
#                         "Please try generating the paper again later. "
#                         f"({error_details(e)})"
#                     ),
#                 )

#             raise HTTPException(
#                 status_code=502,
#                 detail=f"Groq generation error: {error_details(e)}",
#             )


# # ---------------- VALIDATION ----------------

# def validate(paper, allowed, req):

#     if not isinstance(paper, dict):
#         return ["Reply was not a JSON object"]

#     problems = []

#     seen = set()

#     for key, n in (
#         ("mcq", req.mcq),
#         ("short", req.short),
#         ("long", req.long),
#     ):

#         items = paper.get(key, [])

#         if not isinstance(items, list):
#             problems.append(f"{key}: expected a list")
#             continue

#         if len(items) != n:
#             problems.append(
#                 f"{key}: expected {n} questions, got {len(items)}"
#             )

#         for q in items:

#             if not isinstance(q, dict):
#                 problems.append(f"{key}: invalid question object")
#                 continue

#             if str(q.get("topic")) not in allowed:
#                 problems.append(
#                     f"{key}: topic {q.get('topic')} was not selected"
#                 )

#             norm = re.sub(
#                 r"\W+",
#                 " ",
#                 str(q.get("question", "")).lower(),
#             ).strip()

#             if not norm or norm in seen:
#                 problems.append(f"{key}: empty or duplicate question")

#             seen.add(norm)

#             if key == "mcq":

#                 opts = q.get("options", [])

#                 if (
#                     not isinstance(opts, list)
#                     or len(opts) != 4
#                     or len(set(map(str, opts))) != 4
#                 ):
#                     problems.append(
#                         f"mcq needs 4 distinct options: "
#                         f"{str(q.get('question'))[:50]}"
#                     )

#                 if (
#                     isinstance(q.get("answer_index"), bool)
#                     or q.get("answer_index") not in (0, 1, 2, 3)
#                 ):
#                     problems.append(
#                         f"mcq has no valid answer_index: "
#                         f"{str(q.get('question'))[:50]}"
#                     )

#     return problems


# # ---------------- GROUNDING CHECK ----------------

# def cosine_similarity(a, b):

#     dot = sum(x * y for x, y in zip(a, b))

#     ma = math.sqrt(sum(x * x for x in a))
#     mb = math.sqrt(sum(y * y for y in b))

#     if ma == 0 or mb == 0:
#         return 0.0

#     return dot / (ma * mb)


# def add_grounding(paper, ctx):

#     by_topic = {t["number"]: t["vecs"] for t in ctx}

#     items = [
#         (key, q)
#         for key in ("mcq", "short", "long")
#         for q in paper.get(key, [])
#         if isinstance(q, dict) and str(q.get("topic")) in by_topic
#     ]

#     if not items:
#         return []

#     vecs = embed(
#         [q["question"] for _, q in items],
#         task_type="RETRIEVAL_QUERY",
#     )

#     flagged = []

#     for (key, q), vector in zip(items, vecs):

#         grounding = max(
#             cosine_similarity(stored, vector)
#             for stored in by_topic[str(q["topic"])]
#         )

#         q["grounding"] = round(float(grounding), 3)

#         if grounding < LOW_GROUNDING:
#             flagged.append(
#                 f"{key} (topic {q['topic']}) may not follow the book "
#                 f"[grounding {q['grounding']}]: {q['question'][:70]}"
#             )

#     return flagged


# # ---------------- PDF GENERATION ----------------

# def register_fonts():
#     """Use a Unicode TrueType font when available (Greek letters, degree signs, etc.)."""

#     candidates = [
#         (os.getenv("PDF_FONT_PATH"), os.getenv("PDF_FONT_BOLD_PATH")),
#         (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
#         ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
#         ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
#     ]

#     for regular, bold in candidates:

#         if regular and os.path.exists(regular):

#             pdfmetrics.registerFont(TTFont("PaperFont", regular))

#             bold_path = bold if bold and os.path.exists(bold) else regular
#             pdfmetrics.registerFont(TTFont("PaperFont-Bold", bold_path))

#             pdfmetrics.registerFontFamily(
#                 "PaperFont",
#                 normal="PaperFont",
#                 bold="PaperFont-Bold",
#                 italic="PaperFont",
#                 boldItalic="PaperFont-Bold",
#             )

#             return "PaperFont", "PaperFont-Bold"

#     return "Helvetica", "Helvetica-Bold"


# FONT, FONT_BOLD = register_fonts()

# _SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
# _SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


# def fmt(text):
#     """Make model text safe for ReportLab Paragraphs."""

#     s = escape(str(text if text is not None else "").strip())

#     # Unicode sub/superscripts render as black boxes in some fonts
#     s = re.sub(r"[₀-₉₊₋]+", lambda m: "<sub>" + m.group().translate(_SUB) + "</sub>", s)
#     s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+", lambda m: "<super>" + m.group().translate(_SUP) + "</super>", s)

#     # **bold** from markdown
#     s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

#     return s.replace("\n", "<br/>")


# def make_styles():

#     def style(name, **kw):
#         kw.setdefault("fontName", FONT)
#         kw.setdefault("fontSize", 10.5)
#         kw.setdefault("leading", 14.5)
#         return ParagraphStyle(name, **kw)

#     return {
#         "institute": style("institute", fontName=FONT_BOLD, fontSize=16,
#                            leading=20, alignment=TA_CENTER),
#         "title": style("title", fontName=FONT_BOLD, fontSize=13,
#                        leading=17, alignment=TA_CENTER),
#         "meta": style("meta", fontSize=10.5, leading=14),
#         "section": style("section", fontName=FONT_BOLD, fontSize=11.5,
#                          leading=15, spaceBefore=10, spaceAfter=6),
#         "q": style("q", leftIndent=18, firstLineIndent=-18, spaceAfter=3),
#         "opt": style("opt", leftIndent=34, firstLineIndent=-16, spaceAfter=1),
#         "ans": style("ans", leftIndent=18, spaceAfter=7, fontSize=10),
#         "note": style("note", fontSize=9.5, leading=13,
#                       textColor=colors.HexColor("#444444")),
#     }


# def page_footer(book_label):

#     def draw(canvas, doc):

#         canvas.saveState()
#         canvas.setFont(FONT, 8.5)
#         canvas.setFillColor(colors.HexColor("#666666"))

#         w, _ = A4

#         canvas.drawString(18 * mm, 10 * mm, book_label)
#         canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")

#         canvas.restoreState()

#     return draw


# def build_pdf(req: PaperRequest, paper: dict, total_marks: int) -> bytes:

#     st = make_styles()
#     story = []

#     title_text = f"{req.class_name} \u2013 {req.book_name}"

#     # ---- Header ----

#     if req.institute_name:
#         story.append(Paragraph(fmt(req.institute_name), st["institute"]))
#         story.append(Spacer(1, 3))

#     story.append(Paragraph(fmt(title_text) + " \u2013 Question Paper", st["title"]))
#     story.append(Spacer(1, 8))

#     meta = [
#         [
#             Paragraph(f"<b>Class:</b> {fmt(req.class_name)}", st["meta"]),
#             Paragraph(f"<b>Subject:</b> {fmt(req.book_name)}", st["meta"]),
#         ],
#         [
#             Paragraph(f"<b>Time Allowed:</b> {fmt(req.time_allowed or '________')}", st["meta"]),
#             Paragraph(f"<b>Total Marks:</b> {total_marks}", st["meta"]),
#         ],
#         [
#             Paragraph("<b>Name:</b> ____________________________", st["meta"]),
#             Paragraph("<b>Roll No:</b> ______________", st["meta"]),
#         ],
#     ]

#     table = Table(meta, colWidths=[85 * mm, 85 * mm])
#     table.setStyle(TableStyle([
#         ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
#         ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
#         ("TOPPADDING", (0, 0), (-1, -1), 5),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
#     ]))

#     story.append(table)
#     story.append(Spacer(1, 6))
#     story.append(Paragraph(
#         "Attempt all questions. Write neatly and show all working for numerical problems.",
#         st["note"],
#     ))
#     story.append(Spacer(1, 4))
#     story.append(HRFlowable(width="100%", thickness=1, color=colors.black))

#     mcqs = paper.get("mcq", []) or []
#     shorts = paper.get("short", []) or []
#     longs = paper.get("long", []) or []

#     letters = "abcd"

#     # ---- Section A: MCQs ----

#     if mcqs:

#         story.append(Paragraph(
#             f"Section A \u2013 Multiple Choice Questions "
#             f"({len(mcqs)} \u00d7 {req.marks_mcq} = {len(mcqs) * req.marks_mcq} marks)",
#             st["section"],
#         ))

#         for i, q in enumerate(mcqs, 1):

#             block = [Paragraph(f"<b>{i}.</b> {fmt(q.get('question'))}", st["q"])]

#             for j, opt in enumerate(q.get("options", [])[:4]):
#                 block.append(Paragraph(f"({letters[j]}) {fmt(opt)}", st["opt"]))

#             block.append(Spacer(1, 6))
#             story.append(KeepTogether(block))

#     # ---- Section B: Short ----

#     if shorts:

#         story.append(Paragraph(
#             f"Section B \u2013 Short Questions "
#             f"({len(shorts)} \u00d7 {req.marks_short} = {len(shorts) * req.marks_short} marks)",
#             st["section"],
#         ))

#         for i, q in enumerate(shorts, 1):
#             story.append(Paragraph(f"<b>{i}.</b> {fmt(q.get('question'))}", st["q"]))
#             story.append(Spacer(1, 8))

#     # ---- Section C: Long ----

#     if longs:

#         story.append(Paragraph(
#             f"Section C \u2013 Long Questions "
#             f"({len(longs)} \u00d7 {req.marks_long} = {len(longs) * req.marks_long} marks)",
#             st["section"],
#         ))

#         for i, q in enumerate(longs, 1):
#             story.append(Paragraph(f"<b>{i}.</b> {fmt(q.get('question'))}", st["q"]))
#             story.append(Spacer(1, 10))

#     # ---- Answer key ----

#     if req.include_answer_key:

#         story.append(PageBreak())
#         story.append(Paragraph("Answer Key &amp; Marking Scheme", st["title"]))
#         story.append(Spacer(1, 6))
#         story.append(HRFlowable(width="100%", thickness=1, color=colors.black))

#         if mcqs:

#             story.append(Paragraph("Section A \u2013 MCQs", st["section"]))

#             for i, q in enumerate(mcqs, 1):

#                 opts = q.get("options", [])
#                 idx = q.get("answer_index")

#                 if isinstance(idx, int) and 0 <= idx < len(opts):
#                     correct = f"({letters[idx]}) {fmt(opts[idx])}"
#                 else:
#                     correct = "-"

#                 text = f"<b>{i}.</b> <b>{correct}</b>"

#                 if q.get("explanation"):
#                     text += f"<br/>{fmt(q['explanation'])}"

#                 story.append(Paragraph(text, st["ans"]))

#         if shorts:

#             story.append(Paragraph("Section B \u2013 Short Questions", st["section"]))

#             for i, q in enumerate(shorts, 1):
#                 story.append(Paragraph(
#                     f"<b>{i}.</b> {fmt(q.get('answer'))}", st["ans"]))

#         if longs:

#             story.append(Paragraph("Section C \u2013 Long Questions", st["section"]))

#             for i, q in enumerate(longs, 1):
#                 story.append(Paragraph(
#                     f"<b>{i}.</b> {fmt(q.get('answer'))}", st["ans"]))

#     # ---- Build ----

#     buf = BytesIO()

#     doc = SimpleDocTemplate(
#         buf,
#         pagesize=A4,
#         leftMargin=18 * mm,
#         rightMargin=18 * mm,
#         topMargin=16 * mm,
#         bottomMargin=18 * mm,
#         title=f"{title_text} Question Paper",
#         author=req.institute_name or "RAG Question Paper Generator",
#     )

#     footer = page_footer(title_text)

#     doc.build(story, onFirstPage=footer, onLaterPages=footer)

#     return buf.getvalue()


# def safe_filename(req: PaperRequest):

#     name = re.sub(r"[^A-Za-z0-9]+", "_", f"{req.class_name}_{req.book_name}_paper")

#     return name.strip("_") + ".pdf"


# # ---------------- GENERATE PAPER ENDPOINT ----------------

# def run_generation(req: PaperRequest):

#     if req.mcq + req.short + req.long == 0:
#         raise HTTPException(422, "Ask for at least one question")

#     # STEP 1: Retrieve textbook content

#     with get_conn() as conn:
#         ctx = retrieve(conn, req)

#     numbers = [t["number"] for t in ctx]

#     # STEP 2: Distribute questions across topics

#     plan = {
#         "mcq": split_counts(req.mcq, numbers),
#         "short": split_counts(req.short, numbers),
#         "long": split_counts(req.long, numbers),
#     }

#     # STEP 3: Generate and validate paper

#     paper = None
#     problems = []

#     for attempt in range(3):

#         try:
#             prompt = build_prompt(ctx, req, plan, problems)
#             paper = call_llm(prompt)

#         except HTTPException:
#             raise

#         except Exception as e:
#             raise HTTPException(
#                 502,
#                 f"Paper generation failed: {str(e)}",
#             )

#         problems = validate(paper, set(numbers), req)

#         if not problems:
#             break

#     if paper is None:
#         raise HTTPException(502, "The model did not return a valid paper")

#     # STEP 4: Check grounding

#     warnings = problems + add_grounding(paper, ctx)

#     # STEP 5: Return final result

#     return {
#         "class": req.class_name,
#         "book": req.book_name,
#         "difficulty": req.difficulty,
#         "total_marks": (
#             req.mcq * req.marks_mcq
#             + req.short * req.marks_short
#             + req.long * req.marks_long
#         ),
#         "paper": paper,
#         "sources": [
#             {
#                 "topic": t["number"],
#                 "name": t["name"],
#                 "chunks": [
#                     {
#                         "page": c["page"],
#                         "similarity": c["similarity"],
#                     }
#                     for c in t["chunks"]
#                 ],
#             }
#             for t in ctx
#         ],
#         "warnings": warnings,
#     }


# @app.post(
#     "/generate-paper",
#     response_class=Response,
#     responses={
#         200: {
#             "content": {
#                 "application/pdf": {
#                     "schema": {"type": "string", "format": "binary"}
#                 }
#             },
#             "description": "Question paper PDF (downloaded as a file)",
#         }
#     },
# )
# def generate_paper(req: PaperRequest):
#     """Generate the question paper and return it as a PDF file."""

#     result = run_generation(req)

#     try:
#         pdf_bytes = build_pdf(req, result["paper"], result["total_marks"])

#     except Exception as e:
#         raise HTTPException(500, f"PDF generation failed: {str(e)}")

#     return Response(
#         content=pdf_bytes,
#         media_type="application/pdf",
#         headers={
#             "Content-Disposition": f'attachment; filename="{safe_filename(req)}"',
#             "Access-Control-Expose-Headers": "Content-Disposition, X-Paper-Warnings",
#             "X-Paper-Warnings": str(len(result["warnings"])),
#         },
#     )


# @app.post("/generate-paper/json")
# def generate_paper_json(req: PaperRequest):
#     """Same generation, but returns raw JSON with sources, grounding and warnings."""

#     return run_generation(req)




import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai
from openai import OpenAI

from app.routes.paper import  router
from app.services.rag import (
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    state,
)


# ---------------- LIFESPAN ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from environment variables")

    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for embeddings")

    # Groq client for question generation
    state["llm"] = OpenAI(
        api_key=groq_api_key,
        base_url=GROQ_BASE_URL,
        timeout=300.0,
    )

    # Gemini client for embeddings only
    state["embed_llm"] = genai.Client(api_key=gemini_api_key)

    print("Groq and Gemini embedding clients initialized")
    print(f"Generation model: {GROQ_MODEL}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Embedding dimensions: {EMBED_DIMENSIONS}")

    yield

    try:
        state["llm"].close()
    except Exception:
        pass

    try:
        state["embed_llm"].close()
    except Exception:
        pass

    state.clear()


app = FastAPI(
    title="RAG Question Paper Generator",
    description="Generate textbook-grounded question papers using Groq and PostgreSQL",
    lifespan=lifespan,
)

app.include_router(router)