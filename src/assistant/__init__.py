"""VibeFinder AI: natural-language assistant built on the existing VibeFinder recommender."""

from src.assistant.models import AssistantResponse
from src.assistant.providers import MockProvider, OpenRouterProvider, TemplateProvider, make_provider
from src.assistant.render import render
from src.assistant.workflow import Assistant

__all__ = ["Assistant", "AssistantResponse", "MockProvider", "OpenRouterProvider", "TemplateProvider",
           "make_provider", "render"]
