from pathlib import Path
from typing import Dict, List, Any

from engine.storage.indexes.fs_classifier import FileSemanticClassifier
from engine.storage.indexes.fs_semantic_aggregator import FileSemanticAggregator


class FileSystemScanner:
    """
    Scansione semplice del project_root.
    Output a struttura ad albero:
        {
            "root": "...",
            "items": [ ... ]
        }
    """

    def __init__(self, root: str, runtime: Any | None = None):
        self.root = Path(root).resolve()
        self.classifier = FileSemanticClassifier()
        self.aggregator = FileSemanticAggregator(runtime) if runtime else None

    def scan(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "items": self._walk(self.root),
        }

    def _walk(self, folder: Path) -> List[Dict[str, Any]]:
        children = []
        for item in sorted(folder.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if item.is_symlink():
                # per ora ignoriamo i symlink; in futuro potremo gestirli meglio
                continue

            node = self._make_node(item)
            children.append(node)
        return children

    def _make_node(self, item: Path) -> Dict[str, Any]:
        if item.is_dir():
            return {
                "name": item.name,
                "path": str(item),
                "type": "folder",
                "semantic": {
                    "kind": "folder",
                    "language": None,
                    "is_hidden": item.name.startswith("."),
                    "is_test": False,
                    "knowledge_refs": 0,
                    "agent_activity_score": 0.0,
                    "git_status": None,
                },
                "children": self._walk(item),
            }

        sem = self.classifier.classify(str(item))
        enriched = self.aggregator.enrich(str(item)) if self.aggregator else None

        semantic_dict = {
            "kind": sem.kind,
            "language": sem.language,
            "is_hidden": sem.is_hidden,
            "is_test": sem.is_test,
            "knowledge_refs": enriched.knowledge_refs if enriched else sem.knowledge_refs,
            "agent_activity_score": enriched.agent_activity_score if enriched else sem.agent_activity_score,
            "git_status": enriched.git_status if enriched else sem.git_status,
            "entities": enriched.entities if enriched else [],
            "relations": enriched.relations if enriched else [],
            "code_elements": enriched.code_elements if enriched else [],
            "complexity": enriched.complexity if enriched else None,
            "last_agent": enriched.last_agent if enriched else None,
            "last_operation": enriched.last_operation if enriched else None,
        }

        return {
            "name": item.name,
            "path": str(item),
            "type": "file",
            "semantic": semantic_dict,
        }
