"""TF-IDF 的 max_len：按神经网络同一套分词截断后再拼回字符串。"""

from data import tokenize, truncate_text


def test_truncate_text_keeps_exact_token_count():
    text = "Hello, world! This is a test."
    tokens = tokenize(text)
    assert len(tokens) == 9

    out = truncate_text(text, 5)
    assert tokenize(out) == tokens[:5]
    assert len(tokenize(out)) == 5


def test_truncate_text_none_means_no_truncation():
    text = "A long review with <br /> a break and punctuation!"
    assert truncate_text(text, None) == text
    assert tokenize(truncate_text(text, None)) == tokenize(text)


def test_truncate_text_longer_than_text_is_noop_tokens():
    text = "short text"
    out = truncate_text(text, 100)
    assert tokenize(out) == tokenize(text)
