"""Graph definition. Returns an *uncompiled* StateGraph; the API layer compiles it."""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.llm import get_llm
from app.nodes import (
    BillingAgent,
    DomainClassifier,
    GeneralAgent,
    IntentClassifier,
    ResponseAgent,
    ResponseCollector,
    TechnicalAgent,
)
from app.state import OverallState


def build_graph(llm: BaseChatModel | None = None) -> StateGraph:
    llm = llm or get_llm()

    intent_classifier = IntentClassifier(llm)
    response_agent = ResponseAgent(llm)
    domain_classifier = DomainClassifier(llm)
    domain_agents = [BillingAgent(llm), TechnicalAgent(llm), GeneralAgent(llm)]
    response_collector = ResponseCollector()

    graph = StateGraph(OverallState)

    graph.add_node(intent_classifier.name, intent_classifier)
    graph.add_node(response_agent.name, response_agent)
    graph.add_node(domain_classifier.name, domain_classifier)
    for agent in domain_agents:
        graph.add_node(agent.name, agent, input_schema=agent.state_schema)
    graph.add_node(response_collector.name, response_collector)

    # fast api -> intent classifier
    graph.add_edge(START, intent_classifier.name)

    # intent classifier -> domain classifier | response agent
    graph.add_conditional_edges(
        intent_classifier.name,
        IntentClassifier.route,
        [domain_classifier.name, response_agent.name],
    )
    graph.add_edge(response_agent.name, END)

    # domain classifier -> continual routing -> billing | technical | general,
    # or back to the domain classifier when the domain is unclear
    graph.add_conditional_edges(
        domain_classifier.name,
        DomainClassifier.route,
        [agent.name for agent in domain_agents] + [domain_classifier.name],
    )

    # branches -> response collector
    for agent in domain_agents:
        graph.add_edge(agent.name, response_collector.name)
    graph.add_edge(response_collector.name, END)

    return graph
