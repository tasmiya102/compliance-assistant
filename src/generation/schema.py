"""
Defines the structured output shape every answer must follow. Using
Pydantic means we get automatic validation -- if the LLM's response
doesn't match this shape, we'll know immediately rather than silently
accepting malformed output.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(description="The document ID, e.g. 'COC-04'")
    clause_id: str = Field(description="The clause ID within that document, e.g. '2.1'")


class PolicyAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question, in plain language")
    citations: list[Citation] = Field(
        description="One or more clauses that support this answer. Empty if abstaining."
    )
    applicable_policy: str = Field(
        description="Name of the policy document this answer is based on, or 'N/A' if abstaining"
    )
    rule_or_limit: str = Field(
        description="The specific rule, limit, or number mentioned (e.g. '$75 limit'), or 'N/A'"
    )
    required_approval: str = Field(
        description="Any approval required per the policy, or 'None' if not applicable"
    )
    confidence: str = Field(
        description="One of: 'high', 'medium', 'low' -- how well the retrieved context supports this answer"
    )
    abstained: bool = Field(
        description="True if the assistant could not find enough information to answer confidently"
    )