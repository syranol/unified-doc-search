def test_search_endpoint_combines_sources(flask_app, monkeypatch):
    slack_payload = {
        "Slack_0": {
            "text": "Capybaras in Slack updates.",
            "link": "https://slack.example.com/message/456",
            "source": "Slack",
        }
    }
    confluence_payload = {
        "Confluence_0": {
            "text": "Capybaras in Confluence docs.",
            "link": "https://example.atlassian.net/wiki/pages/capybara",
            "source": "Confluence",
        }
    }

    def fake_transform(query, combined):
        assert query == "capybara"
        assert combined == {**slack_payload, **confluence_payload}
        return ["combined-result"]

    monkeypatch.setattr("unified_doc_search.app.slack_search", lambda query: (slack_payload, 200))
    monkeypatch.setattr("unified_doc_search.app.confluence_search", lambda query: (confluence_payload, 200))
    monkeypatch.setattr("unified_doc_search.app.transform_result", fake_transform)

    client = flask_app.test_client()
    response = client.get("/search?q=capybara")

    assert response.status_code == 200
    assert response.get_json() == ["combined-result"]


def test_search_endpoint_allows_partial_success(flask_app, monkeypatch):
    confluence_payload = {
        "Confluence_0": {
            "text": "Capybaras in Confluence docs.",
            "link": "https://example.atlassian.net/wiki/pages/capybara",
            "source": "Confluence",
        }
    }

    def fake_transform(query, combined):
        return combined

    monkeypatch.setattr("unified_doc_search.app.slack_search", lambda query: ("error", 500))
    monkeypatch.setattr("unified_doc_search.app.confluence_search", lambda query: (confluence_payload, 200))
    monkeypatch.setattr("unified_doc_search.app.transform_result", fake_transform)

    client = flask_app.test_client()
    response = client.get("/search?q=capybara")

    assert response.status_code == 200
    assert response.get_json() == confluence_payload


def test_slack_search_helper_returns_error(flask_app, monkeypatch):
    from unified_doc_search.app import slack_search

    monkeypatch.setattr(
        "unified_doc_search.app.slack_controller",
        lambda query: (None, "boom"),
    )

    result, status = slack_search("capybara")

    assert status == 500
    assert result == "boom"
