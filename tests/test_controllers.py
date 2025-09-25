from unittest.mock import Mock

import pytest

from unified_doc_search.controllers import confluence, slack


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_TOKEN", "dummy-token")
    monkeypatch.setenv("CONFLUENCE_USERNAME", "dummy-user")
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "dummy-slack-token")


def test_confluence_controller_success(monkeypatch):
    mocked_response = Mock()
    mocked_response.status_code = 200
    mocked_response.json.return_value = {
        "results": [
            {
                "content": {"_links": {"webui": "/wiki/pages/test"}},
                "excerpt": "Matched text snippet",
            }
        ]
    }

    monkeypatch.setattr(confluence.requests, "get", lambda *args, **kwargs: mocked_response)

    sanitized, error = confluence.confluence_controller("capybara")

    assert error is None
    assert sanitized["Confluence_0"]["source"] == "Confluence"
    assert sanitized["Confluence_0"]["link"].endswith("/wiki/pages/test")


def test_confluence_controller_error(monkeypatch):
    mocked_response = Mock()
    mocked_response.status_code = 502
    mocked_response.text = "Bad gateway"

    monkeypatch.setattr(confluence.requests, "get", lambda *args, **kwargs: mocked_response)

    sanitized, error = confluence.confluence_controller("capybara")

    assert sanitized is None
    assert error == "Bad gateway"


def test_slack_controller_success(monkeypatch):
    mocked_response = Mock()
    mocked_response.status_code = 200
    mocked_response.json.return_value = {
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

    monkeypatch.setattr(slack.requests, "post", lambda *args, **kwargs: mocked_response)

    sanitized, error = slack.slack_controller("capybara")

    assert error is None
    assert sanitized["Slack_0"]["text"] == "Capybaras are excellent swimmers."


def test_slack_controller_error(monkeypatch):
    mocked_response = Mock()
    mocked_response.status_code = 403
    mocked_response.text = "Forbidden"

    monkeypatch.setattr(slack.requests, "post", lambda *args, **kwargs: mocked_response)

    sanitized, error = slack.slack_controller("capybara")

    assert sanitized is None
    assert error == "Forbidden"
