"""Markdown to Slack mrkdwn (moved from correspond)."""

import pytest

from dn.to import markdown_to_mrkdwn


@pytest.mark.parametrize(
    "markdown, mrkdwn",
    [
        ("**bold** and __bold__", "*bold* and *bold*"),
        ("*italic* and _italic_", "_italic_ and _italic_"),
        ("~~gone~~", "~gone~"),
        ("[docs](https://example.org/d)", "<https://example.org/d|docs>"),
        ("<https://example.org/x>", "<https://example.org/x>"),
        ("![chart](https://example.org/c.png)", "<https://example.org/c.png|chart>"),
        ("## Summary ##", "*Summary*"),
        ("- one\n  * two\n+ three", "• one\n  • two\n• three"),
        ("1. first\n2. second", "1. first\n2. second"),
        ("> a quote\nand 1 > 0", "> a quote\nand 1 &gt; 0"),
        ("Tom & Jerry <3", "Tom &amp; Jerry &lt;3"),
        ("`a < b` and `**x**`", "`a &lt; b` and `**x**`"),
        ("```\nif a < b:\n    **x**\n```", "```\nif a &lt; b:\n    **x**\n```"),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"),
        ("2*3*4 and snake_case_name", "2*3*4 and snake_case_name"),
        ("---", "──────────"),
    ],
)
def test_markdown_renders_as_mrkdwn(markdown, mrkdwn):
    assert markdown_to_mrkdwn(markdown) == mrkdwn


def test_a_draft_cannot_carry_a_mention_that_notifies_people():
    assert markdown_to_mrkdwn("<!channel> <!here> <@U123> <#C1>") == (
        "&lt;!channel&gt; &lt;!here&gt; &lt;@U123&gt; &lt;#C1&gt;"
    )





@pytest.mark.parametrize(
    "markdown",
    [
        "[ping](!channel)",
        "[x](@U123ABC)",
        "[](!here)",
        "[x](!subteam^S1)",
        "[x](#C1)",
        "[x](<!channel>)",
        "<!everyone>",
        "[mail](mailto:<!channel>)",
    ],
)
def test_no_link_can_become_a_mention(markdown):
    rendered = markdown_to_mrkdwn(markdown)
    for live in ("<!", "<@", "<#"):
        assert live not in rendered, rendered


@pytest.mark.parametrize(
    "markdown", ["`\x000\x00`", "a \x000\x00 b", "`a` \x000\x00", "\x00\x00\x001\x00"]
)
def test_input_cannot_forge_the_renderers_placeholders(markdown):
    rendered = markdown_to_mrkdwn(markdown.encode().decode("unicode_escape"))
    assert "\x00" not in rendered


@pytest.mark.parametrize(
    "markdown",
    ["[" * 40_000, "[a](" * 10_000, "*_" * 20_000, "**" * 20_000],
    ids=["brackets", "unclosed-links", "emphasis-runs", "bold-runs"],
)
def test_long_adversarial_input_renders_quickly(markdown):
    import time

    start = time.perf_counter()
    markdown_to_mrkdwn(markdown)
    assert time.perf_counter() - start < 2


def test_a_table_inside_a_code_fence_stays_in_the_fence():
    assert markdown_to_mrkdwn("```\n| a | b |\n|---|---|\n```\n**after**") == (
        "```\n| a | b |\n|---|---|\n```\n*after*"
    )


def test_a_line_of_pipes_without_a_separator_is_not_a_table():
    assert markdown_to_mrkdwn("|a|") == "|a|"


@pytest.mark.parametrize(
    "markdown, mrkdwn",
    [
        ("[wiki](https://example.org/Foo_(bar))", "<https://example.org/Foo_(bar)|wiki>"),
        ("[q](https://example.org/a?x=1&y=2)", "<https://example.org/a?x=1&y=2|q>"),
        ("[p](https://example.org/a|b)", "<https://example.org/a%7Cb|p>"),
        ("# **Head** _x_", "*Head _x_*"),
        ("***both***", "*_both_*"),
        ("\\*not italic\\*", "*not italic*"),
        ("- - -", "──────────"),
        ("Tom &amp; Jerry", "Tom &amp; Jerry"),
        (
            "https://example.org/a_b_c and snake_case",
            "https://example.org/a_b_c and snake_case",
        ),
    ],
)
def test_review_cases_render_as_slack_expects(markdown, mrkdwn):
    assert markdown_to_mrkdwn(markdown) == mrkdwn
