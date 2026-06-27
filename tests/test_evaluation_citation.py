from ai_engineer_app.evaluation.metrics import citation_present


def test_citation_faithfulness_requires_supported_page_and_row_provenance():
    chunks = [
        {
            "chunk_id": "chunk_001",
            "source_type": "circonscription",
            "source_id": "001",
            "source_page": 1,
            "chunk_text": "Circonscription 001...",
        }
    ]

    result = citation_present(
        "La circonscription est documentée dans le PDF (page 1).",
        chunks,
        expected_pages=[1],
        expected_source_ids=["001"],
    )

    assert result["pass"] is True
    assert result["unsupported_cited_pages"] == []
    assert result["missing_expected_pages"] == []
    assert result["missing_expected_source_ids"] == []
    assert result["provenance_complete"] is True


def test_citation_faithfulness_rejects_unsupported_page():
    chunks = [
        {
            "chunk_id": "chunk_001",
            "source_type": "circonscription",
            "source_id": "001",
            "source_page": 1,
            "chunk_text": "Circonscription 001...",
        }
    ]

    result = citation_present("Réponse avec une mauvaise citation (page 99).", chunks)

    assert result["pass"] is False
    assert result["unsupported_cited_pages"] == [99]
