#!/usr/bin/env python3
"""
Ingest sample documents for testing the RAG system.
This will add some test data so you can try the knowledge chat.
"""

import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.chunk_embed import run_phase2
from datetime import datetime

def ingest_sample_documents():
    """Ingest sample documents for testing."""

    print("🔄 Ingesting sample documents...")

    # Sample refund policy document
    refund_policy_content = """
# Refund Policy for Widget Products

## General Refund Policy
We offer a 30-day return policy for all Widget products purchased from our authorized dealers.

## Conditions for Refund
- Product must be in original packaging
- Receipt or proof of purchase required
- Product must be unused and undamaged
- Refunds processed within 7-10 business days

## Exceptions
- Custom-ordered widgets are not eligible for refund
- Opened software licenses cannot be refunded
- Refunds not available for products purchased over 30 days ago

## Return Process
1. Contact customer support at support@company.com
2. Receive return authorization number
3. Ship product back to our warehouse
4. Refund issued to original payment method

## Contact Information
Email: refunds@company.com
Phone: 1-800-WIDGETS
"""

    # Sample pricing document
    pricing_content = """
# Widget Pricing Structure

## Standard Pricing
- Widget A: $99.99 (Basic model)
- Widget B: $149.99 (Advanced model)
- Widget C: $199.99 (Premium model)

## Bulk Discounts
- 10-50 units: 5% discount
- 51-100 units: 10% discount
- 100+ units: 15% discount

## Seasonal Promotions
- Back to School: 10% off all models (August-September)
- Holiday Sale: 20% off Widget C (December)

## Warranty Information
All widgets come with 1-year manufacturer warranty covering defects in materials and workmanship.
"""

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(refund_policy_content)
        refund_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(pricing_content)
        pricing_file = f.name

    try:
        # Ingest refund policy
        count1 = run_phase2(
            file_path=refund_file,
            source_type="txt",
            document_id="refund_policy_doc",
            customer_id="acme_corp",
            doc_date=datetime.now().isoformat(),
            filename="refund_policy.pdf",
        )
        print(f"✅ Ingested refund policy: {count1} chunks")

        # Ingest pricing document
        count2 = run_phase2(
            file_path=pricing_file,
            source_type="txt",
            document_id="pricing_doc",
            customer_id="acme_corp",
            doc_date=datetime.now().isoformat(),
            filename="pricing_table.xlsx",
        )
        print(f"✅ Ingested pricing document: {count2} chunks")

        # Ingest the test email
        count3 = run_phase2(
            file_path="test_email.eml",
            source_type="txt",  # Treat as text for now
            document_id="test_email_doc",
            customer_id="acme_corp",
            doc_date="2024-01-15",
            filename="test_email.eml",
        )
        print(f"✅ Ingested test email: {count3} chunks")

        total_chunks = count1 + count2 + count3
        print(f"\n🎉 Successfully ingested {total_chunks} chunks!")
        print("You can now ask questions like:")
        print("- What is the refund policy?")
        print("- How much does Widget A cost?")
        print("- What are the pricing discounts?")
        print("- What are the refund conditions?")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up temp files
        try:
            os.unlink(refund_file)
            os.unlink(pricing_file)
        except:
            pass

if __name__ == "__main__":
    ingest_sample_documents()