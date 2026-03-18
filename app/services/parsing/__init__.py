"""
Adventure document parsing pipeline: normalize → section chunk → classify → extract → relationships → dedupe.
"""

def run_parsing_pipeline(*args, **kwargs):
    from app.services.parsing.pipeline import run_parsing_pipeline as _run_parsing_pipeline
    return _run_parsing_pipeline(*args, **kwargs)

__all__ = ["run_parsing_pipeline"]
