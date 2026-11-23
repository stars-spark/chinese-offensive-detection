"""数据清洗、分词、分层抽样的测试（对应任务 9 的前两项）。"""

from data import clean_text, subsample, tokenize


def test_clean_removes_br():
    cleaned = clean_text("good<br /><br />movie", "imdb")
    assert "<br" not in cleaned.lower()
    assert cleaned == "good movie"


def test_clean_handles_html_entities_and_ag_backslash():
    # html.unescape 应该把 &amp; 还原成 &，AG News 的字面反斜杠应变成空格
    assert clean_text("A &amp; B", "imdb") == "A & B"
    assert clean_text("one\\two", "ag_news") == "one two"


def test_tokenize_contracts_and_punctuation():
    assert tokenize("It's GREAT!") == ["it's", "great", "!"]


def test_subsample_is_stratified():
    import pandas as pd

    df = pd.DataFrame({"text": ["x"] * 100, "label": [0] * 50 + [1] * 50})
    small = subsample(df, 0.10, seed=42)
    # 每类各抽 10%，比例应保持 1:1
    assert len(small) == 10
    assert small["label"].value_counts().to_dict() == {0: 5, 1: 5}
