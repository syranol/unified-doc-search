import pytest


def test_rank_result_prioritizes_matching_text(stubbed_transformer):
    combined_data = {
        "Confluence_0": {
            "text": "Capybaras love swimming in warm lakes.",
            "link": "https://example.atlassian.net/wiki/pages/capybara",
            "source": "Confluence",
        },
        "Slack_0": {
            "text": "Random update about unrelated topics.",
            "link": "https://slack.example.com/message/123",
            "source": "Slack",
        },
    }

    ranking = stubbed_transformer.rank_result("capybara", combined_data)

    assert ranking[0][0] == "Confluence_0"
    assert len(ranking) == 2


def test_sort_result_appends_scores(stubbed_transformer):
    combined_data = {
        "Slack_0": {
            "text": "Capybaras love swimming in warm lakes.",
            "link": "https://slack.example.com/message/123",
            "source": "Slack",
        }
    }

    ranking = [("Slack_0", 0.75)]

    sorted_result = stubbed_transformer.sort_result(combined_data, ranking)

    assert sorted_result == [
        {
            "text": "Capybaras love swimming in warm lakes.",
            "link": "https://slack.example.com/message/123",
            "source": "Slack",
            "score": pytest.approx(0.75),
        }
    ]


def test_transform_result_runs_end_to_end(stubbed_transformer):
    combined_data = {
        "Confluence_0": {
            "text": "Capybaras love swimming in warm lakes.",
            "link": "https://example.atlassian.net/wiki/pages/capybara",
            "source": "Confluence",
        },
        "Slack_0": {
            "text": "Random update about unrelated topics.",
            "link": "https://slack.example.com/message/123",
            "source": "Slack",
        },
    }

    result = stubbed_transformer.transform_result("capybara", combined_data)

    assert result[0]["source"] == "Confluence"
    assert all("score" in item for item in result)
