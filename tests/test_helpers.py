"""Tests for helper functions in extended_openai_conversation."""

import pytest

from custom_components.extended_openai_conversation.helpers import (
    get_model_config,
    resolve_param_inclusion,
)
from custom_components.extended_openai_conversation.const import (
    PARAM_OVERRIDE_AUTO,
    PARAM_OVERRIDE_ENABLED,
    PARAM_OVERRIDE_DISABLED,
)


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------


def test_standard_openai_model():
    """gpt-4o → supports both top_p and temperature."""
    config = get_model_config("gpt-4o")
    assert config["supports_top_p"] is True
    assert config["supports_temperature"] is True
    assert config["supports_max_tokens"] is True
    assert config["supports_max_completion_tokens"] is False
    assert config["supports_reasoning_effort"] is False
    assert config["supports_service_tier"] is False


def test_reasoning_model():
    """o1 → supports neither top_p nor temperature."""
    config = get_model_config("o1")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is False
    assert config["supports_max_completion_tokens"] is True
    assert config["supports_reasoning_effort"] is True
    assert config["supports_service_tier"] is True


def test_anthropic_litellm_model():
    """anthropic/claude-sonnet-4-6 via LiteLLM → no top_p, has temperature."""
    config = get_model_config("anthropic/claude-sonnet-4-6")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is True
    assert config["supports_max_tokens"] is True
    assert config["supports_max_completion_tokens"] is False
    assert config["supports_reasoning_effort"] is False
    assert config["supports_service_tier"] is False


def test_claude_direct_model():
    """claude-3-5-sonnet-20241022 (direct) → no top_p, has temperature."""
    config = get_model_config("claude-3-5-sonnet-20241022")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is True
    assert config["supports_max_tokens"] is True
    assert config["supports_reasoning_effort"] is False


def test_gemini_litellm_model():
    """gemini/gemini-2.0-flash via LiteLLM → no top_p, has temperature."""
    config = get_model_config("gemini/gemini-2.0-flash")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is True
    assert config["supports_max_tokens"] is True
    assert config["supports_reasoning_effort"] is False


def test_vertex_ai_gemini_model():
    """vertex_ai/gemini-1.5-pro via LiteLLM → no top_p, has temperature."""
    config = get_model_config("vertex_ai/gemini-1.5-pro")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is True


def test_unknown_litellm_model_default():
    """ollama/llama3 (unknown) → default config (both top_p and temperature)."""
    config = get_model_config("ollama/llama3")
    assert config["supports_top_p"] is True
    assert config["supports_temperature"] is True
    assert config["supports_max_tokens"] is True
    assert config["supports_max_completion_tokens"] is False
    assert config["supports_reasoning_effort"] is False
    assert config["supports_service_tier"] is False


def test_case_insensitive_anthropic():
    """ANTHROPIC/CLAUDE-SONNET → matches anthropic pattern (case-insensitive)."""
    config = get_model_config("ANTHROPIC/CLAUDE-SONNET")
    assert config["supports_top_p"] is False
    assert config["supports_temperature"] is True


# ---------------------------------------------------------------------------
# Override logic tests
# ---------------------------------------------------------------------------


def test_auto_defers_to_model_when_true():
    """auto + model_supports=True → True."""
    assert resolve_param_inclusion(True, PARAM_OVERRIDE_AUTO) is True


def test_auto_defers_to_model_when_false():
    """auto + model_supports=False → False."""
    assert resolve_param_inclusion(False, PARAM_OVERRIDE_AUTO) is False


def test_enabled_overrides_false():
    """enabled + model_supports=False → True."""
    assert resolve_param_inclusion(False, PARAM_OVERRIDE_ENABLED) is True


def test_enabled_keeps_true():
    """enabled + model_supports=True → True."""
    assert resolve_param_inclusion(True, PARAM_OVERRIDE_ENABLED) is True


def test_disabled_overrides_true():
    """disabled + model_supports=True → False."""
    assert resolve_param_inclusion(True, PARAM_OVERRIDE_DISABLED) is False


def test_disabled_keeps_false():
    """disabled + model_supports=False → False."""
    assert resolve_param_inclusion(False, PARAM_OVERRIDE_DISABLED) is False
