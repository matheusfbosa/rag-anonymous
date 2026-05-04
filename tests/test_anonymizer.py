"""Unit tests for ``rag_anonymous.anonymizer`` helpers.

The Presidio-backed ``Anonymizer`` itself is intentionally not exercised here:
instantiating it loads ``en_core_web_lg`` which is far too heavy for a unit
test. We focus on the pure helpers that protect the engine's input contract.
"""

from rag_anonymous.anonymizer import (
    DEFAULT_ENTITIES,
    _ensure_str,
    _entities_from_env,
)


class TestEnsureStr:
    def test_none_becomes_empty_string(self):
        assert _ensure_str(None) == ""

    def test_plain_string_returned_as_str(self):
        result = _ensure_str("hello")
        assert result == "hello"
        assert type(result) is str

    def test_list_joined_with_spaces(self):
        assert _ensure_str(["foo", "bar", "baz"]) == "foo bar baz"

    def test_tuple_joined_with_spaces(self):
        assert _ensure_str(("foo", "bar")) == "foo bar"

    def test_nested_iterables_are_flattened(self):
        assert _ensure_str(["a", ["b", "c"]]) == "a b c"

    def test_arbitrary_object_falls_back_to_str(self):
        assert _ensure_str(42) == "42"

        class Custom:
            def __str__(self) -> str:
                return "custom-repr"

        assert _ensure_str(Custom()) == "custom-repr"


class TestEntitiesFromEnv:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("RAG_ANON_ANONYMIZER_ENTITIES", raising=False)
        assert _entities_from_env() == DEFAULT_ENTITIES.split(",")

    def test_custom_value_is_split_on_commas(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_ANONYMIZER_ENTITIES", "PERSON,EMAIL_ADDRESS")
        assert _entities_from_env() == ["PERSON", "EMAIL_ADDRESS"]

    def test_single_entity(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_ANONYMIZER_ENTITIES", "PERSON")
        assert _entities_from_env() == ["PERSON"]
