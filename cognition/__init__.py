"""LLM agents for Shrooms: a Narrator and the mycelial-network keystone overlay.

Agents never touch sim state directly. They read observations and the event
log, and act only by proposing typed intents that sim/validator.py checks.
Replays apply the recorded intents and never call a model.
"""
