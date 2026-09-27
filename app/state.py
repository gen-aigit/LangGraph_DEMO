"""Graph state schemas, as drawn in the architecture diagram."""

from typing import Literal, TypedDict

IntentType = Literal["domain", "not_domain"]
Domain = Literal["billing", "technical", "general", "unknown"]


class OverallState(TypedDict, total=False):
    # From the diagram: query, response, intentType, domain
    query: str
    response: str
    intent_type: IntentType
    domain: Domain

    # Bookkeeping for the domain classifier <-> routing loop
    domain_attempts: int

    # Written by the branch agents, read by the response collector
    bill_response: str
    tech_response: str
    gen_response: str


class BillingState(TypedDict, total=False):
    query: str
    bill_response: str
    domain: Domain


class TechState(TypedDict, total=False):
    query: str
    tech_response: str
    domain: Domain


class GenState(TypedDict, total=False):
    query: str
    gen_response: str
    domain: Domain
