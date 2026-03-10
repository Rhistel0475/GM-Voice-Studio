"""
Adventure document parsing pipeline: normalize → section chunk → classify → extract → relationships → dedupe.
"""
from app.services.parsing.pipeline import run_parsing_pipeline

__all__ = ["run_parsing_pipeline"]
