"""Class-based agent nodes for the support graph.

Every node is a callable object: LangGraph calls ``node(state)`` and merges the
returned dict into the graph state.
"""

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import BillingState, GenState, OverallState, TechState

# Node names
INTENT_CLASSIFIER = "intent_classifier"
RESPONSE_AGENT = "response_agent"
DOMAIN_CLASSIFIER = "domain_classifier"
BILLING = "billing"
TECHNICAL = "technical"
GENERAL = "general"
RESPONSE_COLLECTOR = "response_collector"


# ----------------------------------------------------------------------- base

class BaseNode(ABC):
    """A graph node. Subclasses set ``name`` and implement ``__call__``."""

    name: str

    @abstractmethod
    def __call__(self, state: dict) -> dict: ...


class LLMNode(BaseNode):
    """A node that answers with a single chat-model call using ``system_prompt``."""

    system_prompt: str

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def ask(self, prompt: str) -> str:
        response = await self.llm.ainvoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        )
        return response.text.strip()


# ---------------------------------------------------------------- classifiers

class IntentClassifier(LLMNode):
    """Decide whether the query is a support (domain) question at all."""

    name = INTENT_CLASSIFIER
    system_prompt = (
        "You classify messages sent to a customer support assistant. "
        "Answer 'domain' if the message is a support request about billing, "
        "technical problems, or general questions about the product or service. "
        "Answer 'not_domain' for greetings, small talk, or anything unrelated. "
        "Reply with exactly one word: domain or not_domain."
    )

    async def __call__(self, state: OverallState) -> dict:
        label = (await self.ask(state["query"])).lower()
        return {"intent_type": "not_domain" if "not_domain" in label else "domain"}

    @staticmethod
    def route(state: OverallState) -> str:
        return DOMAIN_CLASSIFIER if state["intent_type"] == "domain" else RESPONSE_AGENT


class DomainClassifier(LLMNode):
    """Pick the support domain that should handle the query."""

    name = DOMAIN_CLASSIFIER
    domains = (BILLING, TECHNICAL, GENERAL)
    max_attempts = 2
    system_prompt = (
        "You route customer support requests. Pick the single best team:\n"
        "- billing: invoices, payments, refunds, pricing, subscriptions\n"
        "- technical: errors, bugs, outages, setup, integrations\n"
        "- general: account info, policies, anything else support-related\n"
        "Reply with exactly one word: billing, technical, general, or unknown."
    )

    async def __call__(self, state: OverallState) -> dict:
        label = (await self.ask(state["query"])).lower()
        domain = next((d for d in self.domains if d in label), "unknown")
        return {"domain": domain, "domain_attempts": state.get("domain_attempts", 0) + 1}

    @classmethod
    def route(cls, state: OverallState) -> str:
        """Continual routing: go to the domain agent, or back here if unclear."""
        domain = state.get("domain", "unknown")
        if domain in cls.domains:
            return domain
        if state.get("domain_attempts", 0) < cls.max_attempts:
            return DOMAIN_CLASSIFIER
        return GENERAL


# --------------------------------------------------------------------- agents

class ResponseAgent(LLMNode):
    """Handle queries outside the support domain."""

    name = RESPONSE_AGENT
    system_prompt = (
        "You are a friendly customer support assistant. The user's message is not "
        "a support request. Reply briefly and politely, and mention that you can "
        "help with billing, technical, or general account questions."
    )

    async def __call__(self, state: OverallState) -> dict:
        return {"response": await self.ask(state["query"])}


class DomainAgent(LLMNode):
    """A domain specialist that writes its answer to ``output_key``."""

    output_key: str
    state_schema: type

    async def __call__(self, state: dict) -> dict:
        return {self.output_key: await self.ask(state["query"])}


class BillingAgent(DomainAgent):
    name = BILLING
    output_key = "bill_response"
    state_schema = BillingState
    system_prompt = (
        "You are a billing support specialist. Help with invoices, payments, "
        "refunds, pricing, and subscriptions. Be concise and actionable."
    )


class TechnicalAgent(DomainAgent):
    name = TECHNICAL
    output_key = "tech_response"
    state_schema = TechState
    system_prompt = (
        "You are a technical support engineer. Diagnose the problem and give clear, "
        "step-by-step troubleshooting instructions."
    )


class GeneralAgent(DomainAgent):
    name = GENERAL
    output_key = "gen_response"
    state_schema = GenState
    system_prompt = (
        "You are a general customer support agent. Answer questions about accounts, "
        "policies, and the service helpfully and concisely."
    )


# ------------------------------------------------------------------ collector

class ResponseCollector(BaseNode):
    """Copy whichever branch answered into the final ``response`` field."""

    name = RESPONSE_COLLECTOR
    output_keys = ("bill_response", "tech_response", "gen_response")

    def __call__(self, state: OverallState) -> dict:
        reply = next((state[k] for k in self.output_keys if state.get(k)), None)
        return {"response": reply or "Sorry, I couldn't produce an answer for that request."}
