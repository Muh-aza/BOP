"""
Configs/prompt_config.py
========================
Structural Anchor Prompt (SAP) template.
Four-component design: Role | Aims | Description | Question
"""

from dataclasses import dataclass
from Configs.config import FEW_SHOT_DEMO


@dataclass
class SAPTemplate:
    """
    Four-component SAP structure.

    Fields
    ------
    role        : Domain expert identity
    aims        : Task objective
    description : Contextual grounding (KEGG database, etc.)
    question    : User-turn template with {gene1} and {gene2} placeholders
    ascii_key   : 15-char ASCII structural anchor prepended to system prompt
    """
    role        : str
    aims        : str
    description : str
    question    : str
    ascii_key   : str = ""

    def build_system_prompt(self) -> str:
        """[ascii_key] Role. Aims. Description. [few-shot examples]"""
        base     = f"{self.role} {self.aims} {self.description}"
        anchored = f"{self.ascii_key} {base}" if self.ascii_key else base
        return f"{anchored}\n{FEW_SHOT_DEMO}"

    def build_user_question(self, gene1: str, gene2: str) -> str:
        return self.question.format(gene1=gene1, gene2=gene2)


# ── Default baseline SAP — no structural anchor ────────────────────────────────
BASELINE_SAP = SAPTemplate(
    role=(
        "You are a Gene Interaction Scientist specializing in molecular "
        "biology and signaling pathway analysis."
    ),
    aims=(
        "Your aim is to classify the directional regulatory relationship "
        "between two genes as one of: activation, inhibition, or phosphorylation."
    ),
    description=(
        "Use knowledge from the KEGG Pathway Database, focusing on experimentally "
        "validated molecular interactions. If the relationship cannot be determined, "
        "respond with 'no information'. Provide only the single relationship term."
    ),
    question  = "What effect does gene {gene1} have on gene {gene2}?",
    ascii_key = "",
)


def assemble_system_prompt(
    role: str, aims: str, description: str, ascii_key: str = "",
) -> str:
    """Build a full system-turn string from SAP components."""
    t = SAPTemplate(role=role, aims=aims, description=description,
                    question="", ascii_key=ascii_key)
    return t.build_system_prompt()
