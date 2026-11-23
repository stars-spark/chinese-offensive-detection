"""模型测试：位置编码公式、注意力权重、mask 不变性、词序敏感性、输出形状。"""

import math

import pytest
import torch

from models import BiLSTM, MultiHeadSelfAttention, PositionalEncoding, TextCNN, TransformerClassifier


def small_transformer(**kwargs):
    return TransformerClassifier(
        vocab_size=50, num_classes=3, d_model=16, layers=2, heads=2,
        d_ff=32, dropout=0.0, max_len=32, **kwargs
    )


def test_sinusoidal_positional_encoding_values():
    module = PositionalEncoding(d_model=8, max_len=20, mode="sin", dropout=0.0)
    pe = module.pe[0]  # (max_len, d_model)
    # pos=0: sin(0)=0（偶数维），cos(0)=1（奇数维）
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 1::2], torch.ones(4))
    # 与公式手算值比对 pos=3, i=2（即第 4 维）
    pos, i = 3, 2
    angle = pos / (10000 ** (2 * i / 8))
    assert abs(pe[pos, 2 * i].item() - math.sin(angle)) < 1e-6
    assert abs(pe[pos, 2 * i + 1].item() - math.cos(angle)) < 1e-6


def test_attention_rows_sum_to_one_and_padding_is_zero():
    torch.manual_seed(0)
    attn_module = MultiHeadSelfAttention(d_model=8, heads=2, dropout=0.0)
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False  # 第一条后两个位置是 padding
    mask[1, 5:] = False

    _, attn = attn_module(x, mask, need_weights=True)  # (batch, heads, seq, seq)
    assert attn.shape == (2, 2, 6, 6)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 2, 6), atol=1e-6)
    for b in range(2):
        pad_cols = ~mask[b]
        assert attn[b][:, :, pad_cols].abs().max().item() < 1e-7


def test_padding_does_not_change_outputs():
    torch.manual_seed(1)
    ids = torch.randint(2, 50, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    # 同一批句子后面补 3 个 padding
    ids_pad = torch.cat([ids, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(2, 3, dtype=torch.long)], dim=1)

    models = {
        "transformer": small_transformer(pos_enc="sin"),
        "textcnn": TextCNN(50, 3, embed_dim=16, num_filters=8, dropout=0.5),
        "bilstm": BiLSTM(50, 3, embed_dim=16, hidden_dim=8, dropout=0.5),
    }
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
            out_pad = model(ids_pad, mask_pad)
        assert out.shape == (2, 3), name
        # 容差 1e-5：只允许浮点求和顺序带来的误差，不允许 padding 真的参与计算
        assert torch.allclose(out, out_pad, atol=1e-5), f"{name} 对 padding 不不变"


@pytest.mark.parametrize("pos_enc,expect_same", [("none", True), ("sin", False)])
def test_position_encoding_controls_word_order_sensitivity(pos_enc, expect_same):
    torch.manual_seed(2)
    ids = torch.randint(2, 50, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)
    order = torch.arange(7, -1, -1)  # 把词序完全倒过来

    model = small_transformer(pos_enc=pos_enc)
    model.eval()
    with torch.no_grad():
        out = model(ids, mask)
        out_perm = model(ids[:, order], mask)
    if expect_same:
        # 没有位置编码时，注意力 + 平均池化对词序不敏感
        assert torch.allclose(out, out_perm, atol=1e-5)
    else:
        assert not torch.allclose(out, out_perm, atol=1e-5)


def test_all_models_output_shape():
    torch.manual_seed(3)
    ids = torch.randint(2, 50, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.long)
    models = [
        small_transformer(pos_enc="sin"),
        TextCNN(50, 3, embed_dim=16, num_filters=8, dropout=0.5),
        BiLSTM(50, 3, embed_dim=16, hidden_dim=8, dropout=0.5),
    ]
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(ids, mask)
        assert logits.shape == (4, 3)
