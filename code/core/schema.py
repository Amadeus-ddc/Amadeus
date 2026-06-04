import json
import os
import re
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


ReviewAction = Literal["keep", "merge", "generalize", "reject"]
SpecStatus = Literal["active", "merged", "rejected"]
RuleSource = Literal["emergence", "optimizer"]


class NodeTypeSpec(BaseModel):
    name: str = Field(..., description="Node type name")
    description: str = Field("", description="Node type description")
    when_to_create: str = Field("", description="When to create this node type")
    usage_scene: str = Field("", description="Applicable scenario")
    examples: List[str] = Field(default_factory=list, description="Natural language examples")
    status: SpecStatus = Field("active", description="Current status")
    canonical_name: Optional[str] = Field(None, description="Canonical name assigned after review merging")


class EdgeTypeSpec(BaseModel):
    name: str = Field(..., description="Edge type name")
    description: str = Field("", description="Edge type description")
    when_to_create: str = Field("", description="When to create this edge type")
    usage_scene: str = Field("", description="Applicable scenario")
    examples: List[str] = Field(default_factory=list, description="Natural language examples")
    status: SpecStatus = Field("active", description="Current status")
    canonical_name: Optional[str] = Field(None, description="Canonical name assigned after review merging")


class BuilderRuleSpec(BaseModel):
    name: str = Field(..., description="Rule name")
    rule_text: str = Field(..., description="Rule text")
    examples: List[str] = Field(default_factory=list, description="Rule examples")
    source: RuleSource = Field("emergence", description="Rule source")
    status: SpecStatus = Field("active", description="Current status")


class NodeTypeProposal(BaseModel):
    name: str
    description: str = ""
    when_to_create: str = ""
    usage_scene: str = ""
    examples: List[str] = Field(default_factory=list)


class EdgeTypeProposal(BaseModel):
    name: str
    description: str = ""
    when_to_create: str = ""
    usage_scene: str = ""
    examples: List[str] = Field(default_factory=list)


class RuleProposal(BaseModel):
    name: str
    rule_text: str
    examples: List[str] = Field(default_factory=list)
    source: RuleSource = "emergence"


class SchemaProposalBundle(BaseModel):
    node_types: List[NodeTypeProposal] = Field(default_factory=list)
    edge_types: List[EdgeTypeProposal] = Field(default_factory=list)
    rules: List[RuleProposal] = Field(default_factory=list)


class SchemaReviewResult(BaseModel):
    kind: Literal["node", "edge", "rule"]
    proposed_name: str
    action: ReviewAction
    canonical_name: Optional[str] = None
    reason: str = ""


class SchemaState(BaseModel):
    node_types: List[NodeTypeSpec] = Field(default_factory=list)
    edge_types: List[EdgeTypeSpec] = Field(default_factory=list)
    rules: List[BuilderRuleSpec] = Field(default_factory=list)
    version: int = 0
    n_selected: int = 0
    buffers_seen: int = 0
    last_replay_buffer_idx: int = -1
    has_pending_schema_change: bool = False

    def _active_node_types(self) -> List[NodeTypeSpec]:
        return [item for item in self.node_types if item.status == "active"]

    def _active_edge_types(self) -> List[EdgeTypeSpec]:
        return [item for item in self.edge_types if item.status == "active"]

    def _active_rules(self) -> List[BuilderRuleSpec]:
        return [item for item in self.rules if item.status == "active"]

    def to_prompt_context(self) -> str:
        active_node_types = self._active_node_types()
        active_edge_types = self._active_edge_types()
        active_rules = self._active_rules()

        if not active_node_types and not active_edge_types and not active_rules:
            return "No emerged schema yet. Reuse broad defaults and only propose new types if clearly necessary."

        node_lines = []
        for idx, item in enumerate(active_node_types, 1):
            example_text = "; ".join(item.examples[:2]) if item.examples else "None"
            node_lines.append(
                f"{idx}. {item.name}\n"
                f"   - desc: {item.description or 'None'}\n"
                f"   - when_to_create: {item.when_to_create or 'None'}\n"
                f"   - usage_scene: {item.usage_scene or 'None'}\n"
                f"   - examples: {example_text}"
            )

        edge_lines = []
        for idx, item in enumerate(active_edge_types, 1):
            example_text = "; ".join(item.examples[:2]) if item.examples else "None"
            edge_lines.append(
                f"{idx}. {item.name}\n"
                f"   - desc: {item.description or 'None'}\n"
                f"   - when_to_create: {item.when_to_create or 'None'}\n"
                f"   - usage_scene: {item.usage_scene or 'None'}\n"
                f"   - examples: {example_text}"
            )

        rule_lines = []
        for idx, item in enumerate(active_rules, 1):
            example_text = "; ".join(item.examples[:2]) if item.examples else "None"
            rule_lines.append(
                f"{idx}. {item.name}\n"
                f"   - rule: {item.rule_text}\n"
                f"   - examples: {example_text}"
            )

        node_block = "\n".join(node_lines) if node_lines else "None"
        edge_block = "\n".join(edge_lines) if edge_lines else "None"
        rule_block = "\n".join(rule_lines) if rule_lines else "None"

        return (
            f"Schema Version: {self.version}\n"
            f"Buffers Seen For Emergence: {self.buffers_seen}\n"
            f"Selected n: {self.n_selected}\n\n"
            f"[Node Types]\n{node_block}\n\n"
            f"[Edge Types]\n{edge_block}\n\n"
            f"[Builder Rules]\n{rule_block}"
        )

    def _find_active_by_name(self, items: List[BaseModel], name: str):
        lowered = normalize_type_name(name)
        for item in items:
            item_name = getattr(item, "name", "")
            if getattr(item, "status", "active") == "active" and normalize_type_name(item_name) == lowered:
                return item
        return None

    def apply_reviewed_proposals(
        self,
        proposals: SchemaProposalBundle,
        review_results: List[SchemaReviewResult],
        buffers_seen: Optional[int] = None,
        selected_n: Optional[int] = None,
    ) -> bool:
        changed = False
        result_map = {(r.kind, normalize_type_name(r.proposed_name)): r for r in review_results}

        for proposal in proposals.node_types:
            review = result_map.get(("node", normalize_type_name(proposal.name)))
            if review is None:
                continue
            changed = self._apply_node_review(proposal, review) or changed

        for proposal in proposals.edge_types:
            review = result_map.get(("edge", normalize_type_name(proposal.name)))
            if review is None:
                continue
            changed = self._apply_edge_review(proposal, review) or changed

        for proposal in proposals.rules:
            review = result_map.get(("rule", normalize_type_name(proposal.name)))
            if review and review.action == "reject":
                continue
            changed = self._apply_rule_proposal(proposal) or changed

        if buffers_seen is not None:
            self.buffers_seen = max(self.buffers_seen, buffers_seen)
        if selected_n is not None:
            self.n_selected = max(self.n_selected, selected_n)

        if changed:
            self.version += 1
            self.has_pending_schema_change = True
        return changed

    def _apply_node_review(self, proposal: NodeTypeProposal, review: SchemaReviewResult) -> bool:
        action = review.action
        canonical_name = review.canonical_name or proposal.name
        if action == "reject":
            return False

        if action in {"merge", "generalize"} and canonical_name:
            existing = self._find_active_by_name(self.node_types, canonical_name)
            if existing:
                return merge_node_spec(existing, proposal)
            self.node_types.append(
                NodeTypeSpec(
                    name=canonical_name,
                    description=proposal.description,
                    when_to_create=proposal.when_to_create,
                    usage_scene=proposal.usage_scene,
                    examples=proposal.examples,
                )
            )
            return True

        existing = self._find_active_by_name(self.node_types, proposal.name)
        if existing:
            return merge_node_spec(existing, proposal)

        self.node_types.append(
            NodeTypeSpec(
                name=proposal.name,
                description=proposal.description,
                when_to_create=proposal.when_to_create,
                usage_scene=proposal.usage_scene,
                examples=proposal.examples,
            )
        )
        return True

    def _apply_edge_review(self, proposal: EdgeTypeProposal, review: SchemaReviewResult) -> bool:
        action = review.action
        canonical_name = review.canonical_name or proposal.name
        if action == "reject":
            return False

        if action in {"merge", "generalize"} and canonical_name:
            existing = self._find_active_by_name(self.edge_types, canonical_name)
            if existing:
                return merge_edge_spec(existing, proposal)
            self.edge_types.append(
                EdgeTypeSpec(
                    name=canonical_name,
                    description=proposal.description,
                    when_to_create=proposal.when_to_create,
                    usage_scene=proposal.usage_scene,
                    examples=proposal.examples,
                )
            )
            return True

        existing = self._find_active_by_name(self.edge_types, proposal.name)
        if existing:
            return merge_edge_spec(existing, proposal)

        self.edge_types.append(
            EdgeTypeSpec(
                name=proposal.name,
                description=proposal.description,
                when_to_create=proposal.when_to_create,
                usage_scene=proposal.usage_scene,
                examples=proposal.examples,
            )
        )
        return True

    def _apply_rule_proposal(self, proposal: RuleProposal) -> bool:
        normalized_name = normalize_type_name(proposal.name)
        normalized_text = normalize_text(proposal.rule_text)
        for existing in self.rules:
            if existing.status != "active":
                continue
            if normalize_type_name(existing.name) == normalized_name:
                return merge_rule_spec(existing, proposal)
            if normalize_text(existing.rule_text) == normalized_text:
                changed = merge_rule_spec(existing, proposal)
                if existing.name != proposal.name and not existing.name:
                    existing.name = proposal.name
                    changed = True
                return changed

        self.rules.append(
            BuilderRuleSpec(
                name=proposal.name,
                rule_text=proposal.rule_text,
                examples=proposal.examples,
                source=proposal.source,
            )
        )
        return True

    def needs_replay(self) -> bool:
        return self.has_pending_schema_change

    def mark_replayed_until(self, buffer_idx: int) -> None:
        self.last_replay_buffer_idx = buffer_idx
        self.has_pending_schema_change = False

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class EmergenceOutput(BaseModel):
    analysis: str = ""
    stable: bool = False
    reason: str = ""
    proposals: SchemaProposalBundle = Field(default_factory=SchemaProposalBundle)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def normalize_type_name(name: str) -> str:
    normalized = normalize_text(name)
    normalized = normalized.replace("_", " ")
    return normalized


def is_specific_type_name(name: str) -> bool:
    stripped = (name or "").strip()
    if not stripped:
        return True
    if "'s" in stripped or " of " in stripped.lower():
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    if len(tokens) >= 3 and any(token[:1].isupper() for token in tokens):
        return True
    if any(char.isdigit() for char in stripped):
        return True
    return False


def merge_examples(existing: List[str], incoming: List[str]) -> List[str]:
    merged = []
    seen = set()
    for item in (existing or []) + (incoming or []):
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged[:5]


def merge_node_spec(existing: NodeTypeSpec, proposal: NodeTypeProposal) -> bool:
    changed = False
    if proposal.description and normalize_text(proposal.description) not in normalize_text(existing.description):
        existing.description = proposal.description if not existing.description else f"{existing.description} | {proposal.description}"
        changed = True
    if proposal.when_to_create and normalize_text(proposal.when_to_create) not in normalize_text(existing.when_to_create):
        existing.when_to_create = proposal.when_to_create if not existing.when_to_create else f"{existing.when_to_create} | {proposal.when_to_create}"
        changed = True
    if proposal.usage_scene and normalize_text(proposal.usage_scene) not in normalize_text(existing.usage_scene):
        existing.usage_scene = proposal.usage_scene if not existing.usage_scene else f"{existing.usage_scene} | {proposal.usage_scene}"
        changed = True
    merged_examples = merge_examples(existing.examples, proposal.examples)
    if merged_examples != existing.examples:
        existing.examples = merged_examples
        changed = True
    return changed


def merge_edge_spec(existing: EdgeTypeSpec, proposal: EdgeTypeProposal) -> bool:
    changed = False
    if proposal.description and normalize_text(proposal.description) not in normalize_text(existing.description):
        existing.description = proposal.description if not existing.description else f"{existing.description} | {proposal.description}"
        changed = True
    if proposal.when_to_create and normalize_text(proposal.when_to_create) not in normalize_text(existing.when_to_create):
        existing.when_to_create = proposal.when_to_create if not existing.when_to_create else f"{existing.when_to_create} | {proposal.when_to_create}"
        changed = True
    if proposal.usage_scene and normalize_text(proposal.usage_scene) not in normalize_text(existing.usage_scene):
        existing.usage_scene = proposal.usage_scene if not existing.usage_scene else f"{existing.usage_scene} | {proposal.usage_scene}"
        changed = True
    merged_examples = merge_examples(existing.examples, proposal.examples)
    if merged_examples != existing.examples:
        existing.examples = merged_examples
        changed = True
    return changed


def merge_rule_spec(existing: BuilderRuleSpec, proposal: RuleProposal) -> bool:
    changed = False
    if proposal.rule_text and normalize_text(proposal.rule_text) not in normalize_text(existing.rule_text):
        existing.rule_text = proposal.rule_text if not existing.rule_text else f"{existing.rule_text} | {proposal.rule_text}"
        changed = True
    merged_examples = merge_examples(existing.examples, proposal.examples)
    if merged_examples != existing.examples:
        existing.examples = merged_examples
        changed = True
    if proposal.source != existing.source:
        existing.source = proposal.source
        changed = True
    return changed
