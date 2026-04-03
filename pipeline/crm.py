"""
pipeline/crm.py
SupportTicket Pydantic model + mock CRM save.
Swap save_ticket() body for real HubSpot / Zoho API call when ready.
"""
import os
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


class Citation(BaseModel):
    filename:  str
    section:   str
    doc_date:  str
    excerpt:   str


class SupportTicket(BaseModel):
    ticket_id:         str
    customer_name:     str
    query:             str
    answer:            str
    citations:         List[Citation]   = Field(default_factory=list)
    conflict_detected: bool             = False
    conflict_note:     str              = ""
    agent_notes:       str              = ""
    created_at:        str              = Field(
                           default_factory=lambda: datetime.utcnow().isoformat()
                       )


def build_ticket(
    ticket_id:    str,
    customer_name: str,
    query:        str,
    result:       dict,
) -> SupportTicket:
    """Build a SupportTicket from the generate_answer() result dict."""
    raw_cites = result.get("citations", [])
    citations = [
        Citation(
            filename = c.get("filename",  "unknown"),
            section  = c.get("section",   "—"),
            doc_date = c.get("doc_date",  "—"),
            excerpt  = c.get("excerpt",   ""),
        )
        for c in raw_cites
    ]
    return SupportTicket(
        ticket_id         = ticket_id,
        customer_name     = customer_name,
        query             = query,
        answer            = result.get("answer", ""),
        citations         = citations,
        conflict_detected = result.get("conflict_detected", False),
        conflict_note     = result.get("conflict_explanation", ""),
    )


def save_ticket(ticket: SupportTicket) -> bool:
    """
    Mock CRM save — writes JSON to ./crm_tickets/.
    Replace this body with HubSpot or Zoho API call for production.

    HubSpot example (when ready):
        from hubspot import HubSpot
        api = HubSpot(access_token=os.getenv("HUBSPOT_TOKEN"))
        api.crm.tickets.basic_api.create(SimplePublicObjectInputForCreate(
            properties={"subject": ticket.query, "content": ticket.answer}
        ))
    """
    import pathlib
    out_dir = pathlib.Path("./crm_tickets")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{ticket.ticket_id}.json"
    with open(path, "w") as f:
        f.write(ticket.model_dump_json(indent=2))
    print(f"[crm] Ticket saved → {path}")
    return True