"""
Data Normalization Module

This module provides comprehensive data normalization and cleaning capabilities
for the Semantica framework, enabling standardization and quality improvement
of various data types.

Key Features:
    - Text normalization and cleaning (Unicode, whitespace, special characters)
    - Entity name normalization (aliases, variants, disambiguation)
    - Date and time normalization (formats, timezones, relative dates)
    - Number and quantity normalization (formats, units, currency, scientific notation)
    - Data cleaning (duplicates, validation, missing values)
    - Language detection (multi-language support)
    - Encoding handling (detection, conversion, BOM removal)

Main Classes:
    - TextNormalizer: Text cleaning and normalization
    - EntityNormalizer: Entity name normalization
    - DateNormalizer: Date and time normalization
    - NumberNormalizer: Number and quantity normalization
    - DataCleaner: General data cleaning utilities
    - TextCleaner: Text cleaning utilities
    - LanguageDetector: Language detection
    - EncodingHandler: Encoding detection and conversion

Example Usage:
    >>> from semantica.normalize import TextNormalizer, EntityNormalizer
    >>> text_norm = TextNormalizer()
    >>> normalized = text_norm.normalize_text("Hello   World")
    >>> entity_norm = EntityNormalizer()
    >>> canonical = entity_norm.normalize_entity("John Doe")

Author: Semantica Contributors
License: MIT
"""

from .text_normalizer import (
    TextNormalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    SpecialCharacterProcessor,
)
from .entity_normalizer import (
    EntityNormalizer,
    AliasResolver,
    EntityDisambiguator,
    NameVariantHandler,
)
from .date_normalizer import (
    DateNormalizer,
    TimeZoneNormalizer,
    RelativeDateProcessor,
    TemporalExpressionParser,
)
from .number_normalizer import (
    NumberNormalizer,
    UnitConverter,
    CurrencyNormalizer,
    ScientificNotationHandler,
)
from .data_cleaner import (
    DataCleaner,
    DuplicateDetector,
    DataValidator,
    MissingValueHandler,
    DuplicateGroup,
    ValidationResult,
)
from .text_cleaner import TextCleaner
from .language_detector import LanguageDetector
from .encoding_handler import EncodingHandler

__all__ = [
    "TextNormalizer",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
    "SpecialCharacterProcessor",
    "EntityNormalizer",
    "AliasResolver",
    "EntityDisambiguator",
    "NameVariantHandler",
    "DateNormalizer",
    "TimeZoneNormalizer",
    "RelativeDateProcessor",
    "TemporalExpressionParser",
    "NumberNormalizer",
    "UnitConverter",
    "CurrencyNormalizer",
    "ScientificNotationHandler",
    "DataCleaner",
    "DuplicateDetector",
    "DataValidator",
    "MissingValueHandler",
    "DuplicateGroup",
    "ValidationResult",
    "TextCleaner",
    "LanguageDetector",
    "EncodingHandler",
]
