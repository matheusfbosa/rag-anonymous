from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from rag_anonymous.config import DEFAULT_ANONYMIZER_ENTITIES, Settings

DEFAULT_ENTITIES = DEFAULT_ANONYMIZER_ENTITIES


class Anonymizer:
    def __init__(self, entities: list[str] | None = None) -> None:
        self._entities = entities if entities is not None else _entities_from_env()
        self._analyzer = _create_analyzer_engine()
        self._anonymizer = AnonymizerEngine()

    def anonymize(self, text: object) -> str:
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


def _create_analyzer_engine() -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
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
                    "DATE": "DATE_TIME",
                    "TIME": "DATE_TIME",
                },
                "labels_to_ignore": [
                    "CARDINAL",
                    "LAW",
                    "WORK_OF_ART",
                    "ORDINAL",
                    "PERCENT",
                    "PRODUCT",
                    "LANGUAGE",
                    "QUANTITY",
                    "EVENT",
                    "ORG",
                    "NORP",
                    "MONEY",
                ],
            },
        }
    )
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    engine = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return engine


def _ensure_str(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text + ""
    if isinstance(text, (list, tuple)):
        return " ".join(_ensure_str(block) for block in text)
    return str(text)


def _entities_from_env() -> list[str]:
    return list(Settings.load().anonymizer_entities)
