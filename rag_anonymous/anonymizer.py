import os

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

DEFAULT_ENTITIES = "PERSON,LOCATION,ORGANIZATION,DATE_TIME,MONEY"

MONEY_PATTERN = Pattern(
    name="currency_pattern",
    regex=(
        r"(?:R\$|€|\$|US\$|EUR)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?"
        r"|^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s?(?:R\$|€|\$|US\$|EUR)"
    ),
    score=0.8,
)


class Anonymizer:
    def __init__(self, entities=None):
        self._entities = entities if entities is not None else _entities_from_env()
        self._analyzer = _create_analyzer_engine()
        self._anonymizer = AnonymizerEngine()

    def anonymize(self, text):
        text = _ensure_str(text)
        results = self._analyzer.analyze(
            text=text, language="en", entities=self._entities
        )
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace")},
        )
        return anonymized.text


def _create_analyzer_engine():
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "LOCATION": "LOCATION",
                "GPE": "LOCATION",
                "FAC": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
                "NORP": "NRP",
                "MONEY": "MONEY",
            },
            "labels_to_ignore": [
                "CARDINAL", "LAW",
                "WORK_OF_ART", "ORDINAL", "PERCENT",
                "PRODUCT", "LANGUAGE", "QUANTITY", "EVENT",
            ],
        },
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    money_recognizer = PatternRecognizer(
        supported_entity="MONEY",
        patterns=[MONEY_PATTERN],
    )
    registry.add_recognizer(recognizer=money_recognizer)
    engine = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return engine


def _entities_from_env() -> list[str]:
    return os.getenv("RAG_ANON_ANONYMIZER_ENTITIES", DEFAULT_ENTITIES).split(",")


def _ensure_str(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text + ""
    if isinstance(text, (list, tuple)):
        return " ".join(_ensure_str(block) for block in text)
    return str(text)
