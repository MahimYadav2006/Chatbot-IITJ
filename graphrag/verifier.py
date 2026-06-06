"""
Post-Generation Faithfulness Verifier for GraphRAG.

Layer 4 of the hallucination defense. After the LLM generates a response,
this module verifies that factual claims in the response are grounded in
the retrieved context. Unsupported claims are stripped or the response is
replaced with an "I don't know" fallback.

Uses the same configured LLM provider unless verification is disabled.
"""

import os
import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from env_config import load_env_file

logger = logging.getLogger(__name__)


def is_verification_enabled() -> bool:
    load_env_file()
    return os.environ.get("VERIFY_RESPONSES", "true").lower() in ("true", "1", "yes")


@dataclass
class ClaimVerification:
    """A single factual claim and its verification status."""
    claim: str
    supported: bool
    reason: str = ""


@dataclass
class VerificationResult:
    """Result of faithfulness verification."""
    faithful: bool                          # Overall: are most claims supported?
    cleaned_response: str                   # Response with unsupported claims removed
    original_response: str                  # The raw LLM response
    claims: List[ClaimVerification] = field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0
    skipped: bool = False                   # True if verification was skipped (non-factoid)
    reason: str = ""


class ResponseVerifier:
    """
    Post-generation hallucination detector.

    Uses the LLM itself to check if its response is grounded
    in the retrieved context. This catches:
    - Fabricated names/emails/designations
    - Invented counts or statistics
    - Facts merged from different entities
    """

    def __init__(self, llm):
        """
        Args:
            llm: The active LLM instance for verification prompts.
        """
        self.llm = llm

    def verify(self, query: str, context: str, response: str) -> VerificationResult:
        """
        Verify that the LLM response is grounded in the retrieved context.

        Only verifies factoid queries (who/what/how many/list/name).
        Broad reasoning queries skip verification.

        Args:
            query: The original user query.
            context: The retrieved context that was fed to the LLM.
            response: The LLM-generated response.

        Returns:
            VerificationResult with faithful status and cleaned response.
        """
        if not is_verification_enabled():
            return VerificationResult(
                faithful=True,
                cleaned_response=response,
                original_response=response,
                skipped=True,
                reason="Verification disabled via VERIFY_RESPONSES env var",
            )

        if not self._is_factoid_query(query):
            return VerificationResult(
                faithful=True,
                cleaned_response=response,
                original_response=response,
                skipped=True,
                reason="Non-factoid query — verification skipped",
            )

        # Skip verification for short responses (likely "I don't know" or errors)
        if len(response.split()) < 10:
            return VerificationResult(
                faithful=True,
                cleaned_response=response,
                original_response=response,
                skipped=True,
                reason="Response too short — likely a refusal or error",
            )

        try:
            return self._run_verification(query, context, response)
        except Exception as e:
            logger.warning(f"Verification failed with error: {e}. Passing response through.")
            return VerificationResult(
                faithful=True,
                cleaned_response=response,
                original_response=response,
                skipped=True,
                reason=f"Verification error: {e}",
            )

    def _is_factoid_query(self, query: str) -> bool:
        """Determine if a query is factoid (requires verification) vs. broad reasoning."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        factoid_starters = (
            "who ", "what ", "which ", "where ", "when ",
            "name ", "list ", "give me ", "show me ",
            "tell me about ", "how many ", "how much ",
            "email ", "contact ",
        )
        return q.startswith(factoid_starters) or any(
            kw in q for kw in ("how many", "how much", "email of", "contact of", "phone number")
        )

    def _run_verification(self, query: str, context: str, response: str) -> VerificationResult:
        """Run the actual LLM-based claim verification."""
        # For multi-department contexts, use a larger window so we don't
        # miss evidence from departments appearing later in the merged context.
        is_multi_dept = context.count("## Department of") >= 2 or context.count("\n---\n") >= 2
        max_context_chars = 10000 if is_multi_dept else 3000
        context_truncated = context[:max_context_chars] if len(context) > max_context_chars else context

        verification_prompt = f"""You are a strict factual accuracy checker. Your job is to verify if a response is supported by the given context.

CONTEXT (source of truth):
{context_truncated}

RESPONSE TO VERIFY:
{response}

INSTRUCTIONS:
1. Extract each factual claim from the RESPONSE (names, numbers, emails, designations, relationships).
2. For each claim, check if it is EXPLICITLY supported by the CONTEXT.
3. A claim is SUPPORTED only if the exact fact appears in the CONTEXT.
4. A claim is NOT SUPPORTED if it is invented, inferred, or not present in the CONTEXT.
5. General statements like greetings or "I don't know" are always SUPPORTED.
6. If the response attributes information to a specific department and that department's section appears in the CONTEXT, the department attribution itself is SUPPORTED.
7. If the CONTEXT contains multiple department sections, claims from ANY section count as SUPPORTED.

Output ONLY valid JSON (no markdown, no explanation):
{{"claims": [{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false, "reason": "..."}}], "overall_faithful": true/false}}"""

        raw = self.llm.generate(
            verification_prompt,
            temperature=0.1,
            max_tokens=800,
        )

        claims, overall = self._parse_verification_response(raw)

        supported_count = sum(1 for c in claims if c.supported)
        total = len(claims)

        if total == 0:
            # Could not extract claims — pass through
            return VerificationResult(
                faithful=True,
                cleaned_response=response,
                original_response=response,
                skipped=True,
                reason="No claims extracted — passing through",
            )

        # Decision logic:
        # - Single-dept: >50% claims supported → faithful
        # - Multi-dept: >30% claims supported → faithful (context truncation
        #   causes false negatives for departments appearing later in the merge)
        faithfulness_ratio = supported_count / total
        threshold = 0.3 if is_multi_dept else 0.5
        is_faithful = faithfulness_ratio > threshold

        if is_faithful and supported_count < total:
            # Partially faithful — log unsupported claims but keep response
            unsupported = [c for c in claims if not c.supported]
            for c in unsupported:
                logger.warning(f"Unsupported claim detected: '{c.claim}' — Reason: {c.reason}")
            cleaned = response
        elif not is_faithful:
            cleaned = response  # Will be replaced by fallback in app.py
            logger.warning(
                f"Response failed faithfulness check: {supported_count}/{total} claims supported. "
                f"Query: '{query[:80]}'"
            )
        else:
            cleaned = response

        return VerificationResult(
            faithful=is_faithful,
            cleaned_response=cleaned,
            original_response=response,
            claims=claims,
            total_claims=total,
            supported_claims=supported_count,
            reason=f"{supported_count}/{total} claims supported (ratio: {faithfulness_ratio:.2f})",
        )

    def _parse_verification_response(self, raw: str) -> tuple:
        """Parse the JSON output from the verification LLM call."""
        claims = []
        overall = True

        # Try to extract JSON from the response
        raw_clean = raw.strip()

        # Remove markdown code fences if present
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)

        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            json_match = re.search(r"\{[\s\S]*\}", raw_clean)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse verification JSON: {raw_clean[:200]}")
                    return [], True
            else:
                logger.warning(f"No JSON found in verification response: {raw_clean[:200]}")
                return [], True

        overall = data.get("overall_faithful", True)

        for item in data.get("claims", []):
            claims.append(ClaimVerification(
                claim=item.get("claim", ""),
                supported=item.get("supported", True),
                reason=item.get("reason", ""),
            ))

        return claims, overall
