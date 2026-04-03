#!/usr/bin/env python3
"""
SME Knowledge Agent - Production RAG System
==========================================

This is the main entry point for the Streamlit application.
It now uses the modular pipeline architecture for production-ready RAG.

Features:
- Multi-format document ingestion (PDF, Excel, email, text)
- Dual embedding backends (OpenAI + local fallback)
- Dual vector stores (ChromaDB + FAISS)
- Conflict detection and resolution
- Multi-backend reranking (Cohere + cross-encoder)
- CRM integration with support ticket creation
- Customer-scoped knowledge bases
- Citation tracking and source attribution

For development, run: streamlit run ui/app.py
For production, deploy ui/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the UI application
from ui.app import *