#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_without_outer_ws(path: Path) -> str:
    return hashlib.sha256(path.read_text().strip().encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\\b[\\w’'-]+\\b", text, flags=re.UNICODE))


def standalone_i_count(text: str) -> int:
    return len(re.findall(r"(?<![\\w’'])I(?![\\w’'])", text))


def paragraph_list(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]


def duplicate_paragraph_count(text: str) -> int:
    seen: set[str] = set()
    duplicates = 0
    for paragraph in paragraph_list(text):
        normalized = re.sub(r"\\s+", " ", paragraph).lower()
        if normalized.startswith("#") and len(normalized.split()) < 12:
            continue
        if normalized in seen:
            duplicates += 1
        else:
            seen.add(normalized)
    return duplicates


def jaccard_tokens(paragraph: str) -> set[str]:
    return set(re.findall(r"[\\w’'-]+", paragraph.lower(), flags=re.UNICODE))


def near_duplicate_pairs(text: str, threshold: float) -> list[dict]:
    paragraphs = paragraph_list(text)
    pairs = []
    for i, left in enumerate(paragraphs):
        left_tokens = jaccard_tokens(left)
        if not left_tokens:
            continue
        for j in range(i + 1, len(paragraphs)):
            right_tokens = jaccard_tokens(paragraphs[j])
            union = left_tokens | right_tokens
            if not union:
                continue
            score = len(left_tokens & right_tokens) / len(union)
            if score >= threshold:
                pairs.append(
                    {
                        "paragraph_a": i + 1,
                        "paragraph_b": j + 1,
                        "jaccard": score,
                    }
                )
    return pairs


def parse_hash_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise RuntimeError(f"invalid hash inventory line {line_no}: {raw!r}")
        digest, rel = parts
        if rel in result:
            raise RuntimeError(f"duplicate hash inventory path: {rel}")
        result[rel] = digest
    return result


def actual_inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "provenance/hashes.sha256":
            continue
        result[rel] = sha256_file(path)
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_archive(archive: Path) -> None:
    manifest = load_json(archive / "manifest.json")
    config = load_json(archive / "config.json")
    qualification = load_json(archive / "provider" / "qualification.json")
    source_artifacts = load_json(archive / "provenance" / "source-artifacts.json")
    audit = load_json(archive / "provenance" / "post-qualification-audit.json")

    story_path = archive / "compiled" / "story.md"
    preaudit_path = archive / "compiled" / "story-pre-semantic-audit.md"
    synthesis_path = archive / "compiled" / "synthesis-brief.md"
    review_path = archive / "compiled" / "review.md"

    recorded = parse_hash_inventory(archive / "provenance" / "hashes.sha256")
    actual = actual_inventory(archive)
    missing = sorted(set(actual) - set(recorded))
    stale = sorted(set(recorded) - set(actual))
    require(not missing, f"hash inventory missing paths: {missing}")
    require(not stale, f"hash inventory has stale paths: {stale}")
    mismatched = sorted(
        rel for rel, digest in actual.items() if recorded.get(rel) != digest
    )
    require(not mismatched, f"hash inventory mismatches: {mismatched}")

    story = story_path.read_text()
    preaudit = preaudit_path.read_text()
    story_sha = sha256_file(story_path)
    preaudit_sha = sha256_file(preaudit_path)
    synthesis_sha = sha256_file(synthesis_path)
    review_sha = sha256_file(review_path)

    require(manifest["final_story_sha256"] == story_sha, "manifest final story hash mismatch")
    require(manifest["synthesis_sha256"] == synthesis_sha, "manifest synthesis file hash mismatch")
    require(manifest["review_sha256"] == review_sha, "manifest review file hash mismatch")
    require(
        manifest["provider_qualification_synthesis_content_sha256"]
        == qualification["synthesis_sha256"],
        "provider qualification synthesis content hash not preserved in manifest",
    )
    require(
        manifest["provider_qualification_review_content_sha256"]
        == qualification["review_sha256"],
        "provider qualification review content hash not preserved in manifest",
    )
    require(
        qualification["synthesis_sha256"] == sha256_text_without_outer_ws(synthesis_path),
        "provider qualification synthesis content hash mismatch",
    )
    require(
        qualification["review_sha256"] == sha256_text_without_outer_ws(review_path),
        "provider qualification review content hash mismatch",
    )
    require(
        qualification["final_sha256"] == preaudit_sha,
        "provider qualification final hash does not match preserved pre-audit story",
    )

    words = word_count(story)
    i_count = standalone_i_count(story)
    separators = len(re.findall(r"(?m)^\\s*---+\\s*$", story))
    anchor_resets = len(re.findall(r"the smell hit me first", story, flags=re.I))
    duplicates = duplicate_paragraph_count(story)
    acceptance = config["acceptance"]

    require(words == manifest["word_count"], "manifest word count mismatch")
    require(i_count == manifest["standalone_i_count"], "manifest standalone-I count mismatch")
    require(
        abs((i_count / words) - manifest["standalone_i_rate"]) < 1e-15,
        "manifest standalone-I rate mismatch",
    )
    require(separators == manifest["variant_separator_count"], "manifest separator count mismatch")
    require(duplicates == manifest["duplicate_paragraph_count"], "manifest duplicate count mismatch")
    require(words >= int(acceptance["minimum_words"]), "accepted story below minimum word gate")
    require(words <= int(acceptance["maximum_words"]), "accepted story above maximum word gate")
    require(i_count <= int(acceptance["maximum_standalone_i"]), "accepted story exceeds standalone-I gate")
    require(
        i_count / words <= float(acceptance["maximum_standalone_i_rate"]),
        "accepted story exceeds standalone-I rate gate",
    )
    require(
        separators <= int(acceptance["maximum_variant_separators"]),
        "accepted story exceeds separator gate",
    )
    require(
        anchor_resets <= int(acceptance["maximum_anchor_reset_count"]),
        "accepted story exceeds anchor reset gate",
    )
    require(duplicates == 0, "accepted story contains exact duplicate paragraphs")
    require(manifest["qualified"] is True, "manifest is not qualified")

    semantic_pairs = near_duplicate_pairs(story, 0.28)
    require(not semantic_pairs, f"accepted story has semantic near-duplicates: {semantic_pairs}")
    require(
        manifest["near_duplicate_paragraph_pairs_ge_0_28"] == 0,
        "manifest semantic duplicate count is nonzero",
    )

    require(
        audit["provider_pre_audit"]["story_sha256"] == preaudit_sha,
        "audit pre-audit story hash mismatch",
    )
    require(
        audit["provider_pre_audit"]["word_count"] == word_count(preaudit),
        "audit pre-audit word count mismatch",
    )
    accepted_audit = audit["accepted_post_audit"]
    require(accepted_audit["story_sha256"] == story_sha, "audit accepted story hash mismatch")
    require(accepted_audit["word_count"] == words, "audit accepted word count mismatch")
    require(accepted_audit["standalone_i_count"] == i_count, "audit accepted I count mismatch")
    require(
        accepted_audit["near_duplicate_pairs_at_jaccard_ge_0_28"] == [],
        "audit accepted semantic duplicate list is nonempty",
    )
    require(audit["correction"]["text_added"] is False, "semantic correction claims added text")

    pre_paragraphs = paragraph_list(preaudit)
    removed_indices = audit["correction"]["removed_provider_paragraph_indices"]
    removed_paragraphs = audit["correction"]["removed_paragraphs"]
    require(
        len(removed_indices) == len(removed_paragraphs),
        "audit removed index/text cardinality mismatch",
    )
    for index, paragraph in zip(removed_indices, removed_paragraphs):
        require(
            1 <= int(index) <= len(pre_paragraphs)
            and pre_paragraphs[int(index) - 1] == paragraph,
            f"audit removed paragraph does not match pre-audit paragraph {index}",
        )
    removed_set = {int(index) for index in removed_indices}
    reconstructed = "\n\n".join(
        paragraph
        for index, paragraph in enumerate(pre_paragraphs, 1)
        if index not in removed_set
    ).strip()
    require(
        reconstructed == story.strip(),
        "accepted story is not exactly pre-audit story minus recorded paragraphs",
    )

    for name, source in source_artifacts["source_hashes"].items():
        source_path = ROOT / source["path"]
        require(source_path.is_file(), f"missing source artifact {name}: {source_path}")
        require(
            sha256_file(source_path) == source["sha256"],
            f"source artifact hash drift: {name}",
        )
        require(
            source_path.stat().st_size == source["bytes"],
            f"source artifact byte-size drift: {name}",
        )

    print(
        "STORY-001-V2 archive verification: PASS "
        f"files={len(actual)} words={words} I={i_count} "
        f"story_sha256={story_sha}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "datasets" / "story-001-v2",
    )
    args = parser.parse_args()
    verify_archive(args.archive)


if __name__ == "__main__":
    main()
