from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str = "apple-notes"
    metadata: Dict[str, Any] = field(default_factory=dict)
