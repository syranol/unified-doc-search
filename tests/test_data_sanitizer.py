from unified_doc_search.controllers.utils.data_sanitizer import sanitize_confluence, sanitize_slack


def test_sanitize_confluence_happy_path():
    raw_payload = {
        "results": [
            {
                "content": {"_links": {"webui": "/wiki/pages/test"}},
                "excerpt": "Matched text snippet",
            }
        ]
    }

    sanitized = sanitize_confluence(raw_payload)

    assert "Confluence_0" in sanitized
    entry = sanitized["Confluence_0"]
    assert entry["link"].endswith("/wiki/pages/test")
    assert entry["text"] == "Matched text snippet"
    assert entry["source"] == "Confluence"


def test_sanitize_confluence_with_no_results():
    sanitized = sanitize_confluence({"results": []})

    assert sanitized == {}


def test_sanitize_slack_happy_path():
    raw_payload = {
        "messages": {
            "matches": [
                {
                    "permalink": "https://slack.example.com/message/123",
                    "blocks": [
                        {
                            "elements": [
                                {
                                    "elements": [
                                        {
                                            "text": "Capybaras are excellent swimmers."
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    }

    sanitized = sanitize_slack(raw_payload)

    assert "Slack_0" in sanitized
    entry = sanitized["Slack_0"]
    assert entry["link"] == "https://slack.example.com/message/123"
    assert entry["text"] == "Capybaras are excellent swimmers."
    assert entry["source"] == "Slack"


def test_sanitize_slack_with_no_matches():
    sanitized = sanitize_slack({"messages": {"matches": []}})

    assert sanitized == {}
