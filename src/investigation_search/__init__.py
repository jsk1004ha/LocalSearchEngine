from .bm25 import BM25Index, build_bm25_index, search_bm25
from .contradiction import (
    ContradictionDetector,
    ContradictionPrediction,
    HeuristicContradictionDetector,
    OnnxContradictionDetectorAdapter,
)
from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts, load_local_model
from .engine import InvestigationEngine
from .entity import AliasCandidate, generate_alias_candidates, normalize_masked_entity
from .index_ann import build_index, load_index, save_index, search_index
from .learning import LearningConfig, LearningState, OnlineLearningManager
from .ollama import OllamaClient, OllamaContradictionDetector, OllamaRerankerAdapter
from .parser import DocumentParser, ParseConfig, parse_documents
from .reranker import LocalCrossEncoderReranker, OnnxRerankerAdapter
from .offline import (
    build_manifest,
    knowledge_build_id,
    load_build,
    load_sharded_build,
    write_build,
    write_build_from_documents,
)
from .retrieval import RetrievalWeights, build_bm25_for_units
from .schema import (
    EvidenceUnit,
    ScoredEvidence,
    SearchResult,
    SourceCitation,
    SourceType,
    Verdict,
    build_source_citation,
)

__all__ = [
    "InvestigationEngine",
    "BM25Index",
    "build_bm25_index",
    "search_bm25",
    "build_bm25_for_units",
    "RetrievalWeights",
    "ContradictionDetector",
    "ContradictionPrediction",
    "HeuristicContradictionDetector",
    "OnnxContradictionDetectorAdapter",
    "DocumentParser",
    "ParseConfig",
    "parse_documents",
    "LearningConfig",
    "LearningState",
    "OnlineLearningManager",
    "OllamaClient",
    "OllamaRerankerAdapter",
    "OllamaContradictionDetector",
    "AliasCandidate",
    "normalize_masked_entity",
    "generate_alias_candidates",
    "build_manifest",
    "knowledge_build_id",
    "write_build",
    "write_build_from_documents",
    "load_build",
    "load_sharded_build",
    "build_index",
    "search_index",
    "save_index",
    "load_index",
    "DEFAULT_EMBEDDING_MODEL",
    "load_local_model",
    "encode_texts",
    "LocalCrossEncoderReranker",
    "OnnxRerankerAdapter",
    "EvidenceUnit",
    "ScoredEvidence",
    "SearchResult",
    "SourceCitation",
    "build_source_citation",
    "SourceType",
    "Verdict",
]
