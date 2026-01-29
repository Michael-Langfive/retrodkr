import torch
import torch.nn as nn

from onmt.decoders.transformer import TransformerDecoderLayerBase, TransformerDecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class RetroDKRDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
            self,
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type="scaled-dot",
            max_relative_positions=0,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_heads=0,
            pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(RetroDKRDecoderLayer, self).__init__(
            d_model, heads, d_ff, dropout, attention_dropout,
            self_attn_type, max_relative_positions, aan_useffn,
            full_context_alignment, alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
        )

        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.ref_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.frag_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)

        # Layer Norms
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        super(RetroDKRDecoderLayer, self).update_dropout(dropout, attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.ref_attn.update_dropout(attention_dropout)
        self.frag_attn.update_dropout(attention_dropout)

    def _forward(
            self,
            inputs,
            memory_bank,
            src_pad_mask,
            tgt_pad_mask,
            ref_memory_bank=None,
            ref_pad_mask=None,
            sim=None,
            frag_memory_bank=None,
            frag_pad_mask=None,
            frag_sim=None,
            layer_cache=None,
            step=None,
            future=False,
    ):

        dec_mask = self._compute_dec_mask(tgt_pad_mask, future) if inputs.size(1) > 1 else None

        inputs_norm = self.layer_norm_1(inputs)
        query, _ = self._forward_self_attn(
            inputs_norm, dec_mask, layer_cache, step
        )
        query = self.drop(query) + inputs

        if ref_memory_bank is not None:
            query_norm = self.layer_norm_2(query)
            mid, ref_attns = self.ref_attn(
                ref_memory_bank,
                ref_memory_bank,
                query_norm,
                mask=ref_pad_mask,
                attn_type="ref",
            )
            sim = sim.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]
            mid = mid * sim
            query = self.drop(mid) + query
        if frag_memory_bank is not None:
            query_norm = self.layer_norm_3(query)
            mid, frag_attns = self.frag_attn(
                frag_memory_bank,
                frag_memory_bank,
                query_norm,
                mask=frag_pad_mask,
                attn_type="frag",
            )
            mid = frag_sim * mid
            query = self.drop(mid) + query
        query_norm = self.layer_norm_4(query)
        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )

        output = self.feed_forward(self.drop(mid) + query)

        return output, attns


class RetroDKRDecoder(TransformerDecoderBase):
    def __init__(
            self,
            num_layers,
            d_model,
            heads,
            d_ff,
            copy_attn,
            self_attn_type,
            dropout,
            attention_dropout,
            embeddings,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            pos_ffn_activation_fn=ActivationFunction.relu,
            lambda1=1.0,
            lambda2=0.0,
            lambda3=0.0
    ):
        super(RetroDKRDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.transformer_layers = nn.ModuleList(
            [
                RetroDKRDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def detach_state(self):
        if self.state["cache"] is not None:
            self.state = self.state.copy()
            self.state["cache"] = {
                k: v.detach()
                for k, v in self.state["cache"].items()
            }

            if "src" in self.state:
                self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank=None, memory_lengths=None,
                ref_memory_bank=None, ref_memory_lengths=None, sim=None,
                frag_memory_bank=None, frag_memory_lengths=None, frag_sim=None,
                step=None, **kwargs):
        if memory_bank is None:
            memory_bank = self.embeddings(tgt)
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        use_ref = self.lambda2 > 0.0 and ref_memory_bank is not None
        use_frag = self.lambda3 > 0.0 and frag_memory_bank is not None
        if use_ref:
            ref_memory_bank = ref_memory_bank.transpose(0, 1).contiguous()
            ref_max_len = ref_memory_bank.size(1)
            ref_pad_mask = ~sequence_mask(ref_memory_lengths, ref_max_len).unsqueeze(1)
        else:
            ref_memory_bank = None
            ref_pad_mask = None
        if use_frag:
            frag_memory_bank = frag_memory_bank.transpose(0, 1).contiguous()
            frag_max_len = frag_memory_bank.size(1)
            frag_pad_mask = ~sequence_mask(frag_memory_lengths, frag_max_len).unsqueeze(1)
        else:
            frag_memory_bank = None
            frag_pad_mask = None
        pad_idx = self.embeddings.word_padding_idx
        src_lens = memory_lengths
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)

        with_align = kwargs.pop("with_align", False)
        attn_aligns = []
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                ref_memory_bank=ref_memory_bank,
                ref_pad_mask=ref_pad_mask,
                sim=sim,
                frag_memory_bank=frag_memory_bank,
                frag_pad_mask=frag_pad_mask,
                frag_sim=frag_sim,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)
        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]

        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
            }
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros(
                    (batch_size, 1, depth), device=memory_bank.device
                )
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"][f"layer_{i}"] = layer_cache
