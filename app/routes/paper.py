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

# from app.services.rag import PaperRequest, build_pdf, run_generation, safe_filename

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

# from fastapi import APIRouter, HTTPException, Response

# from app.services.rag import (
#     PaperRequest,
#     build_pdf,
#     run_generation,
#     safe_filename,
# )

# router = APIRouter()


# @router.post("/generate-paper/json")
# def generate_paper_json(req: PaperRequest):
#     """
#     Same generation, but returns raw JSON
#     with sources, grounding and warnings.
#     """

#     return run_generation(req)


# @router.post(
#     "/generate-paper",
#     response_class=Response,
#     responses={
#         200: {
#             "content": {
#                 "application/pdf": {
#                     "schema": {
#                         "type": "string",
#                         "format": "binary"
#                     }
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
#         pdf_bytes = build_pdf(
#             req,
#             result["paper"],
#             result["total_marks"],
#         )

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"PDF generation failed: {str(e)}",
#         )

#     return Response(
#         content=pdf_bytes,
#         media_type="application/pdf",
#         headers={
#             "Content-Disposition": (
#                 f'attachment; filename="{safe_filename(req)}"'
#             ),
#             "Access-Control-Expose-Headers": (
#                 "Content-Disposition, X-Paper-Warnings"
#             ),
#             "X-Paper-Warnings": str(len(result["warnings"])),
#         },
#     )



from fastapi import APIRouter, HTTPException, Response

from app.services.rag import PaperRequest, build_pdf, run_generation, safe_filename

router = APIRouter()


@router.post(
    "/generate-paper",
    response_class=Response,
    responses={
        200: {
            "content": {
                "application/pdf": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "description": "Question paper PDF (downloaded as a file)",
        }
    },
)
def generate_paper(req: PaperRequest):
    """Generate the question paper and return it as a PDF file."""

    result = run_generation(req)

    try:
        pdf_bytes = build_pdf(req, result["paper"], result["total_marks"])

    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename(req)}"',
            "Access-Control-Expose-Headers": "Content-Disposition, X-Paper-Warnings",
            "X-Paper-Warnings": str(len(result["warnings"])),
        },
    )


@router.post("/generate-paper/json")
def generate_paper_json(req: PaperRequest):
    """Same generation, but returns raw JSON with sources, grounding and warnings."""

    return run_generation(req)