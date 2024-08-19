import logging
import os

import torch
from torch import nn
from transformers import BertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from .arguments import ModelArguments
from .enhancedDecoder import BertLayerForDecoder

logger = logging.getLogger(__name__)

def _print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable: {name}")
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}"
    )


class RetroMAEForPretraining(nn.Module):
    def __init__(
            self,
            bert: BertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.model_args = model_args
        self.lm = bert
        self.layer_freeze(self.lm)
        _print_trainable_parameters(self.lm)

        if hasattr(self.lm, 'bert'):
            self.decoder_embeddings = self.lm.bert.embeddings
        elif hasattr(self.lm, 'roberta'):
            self.decoder_embeddings = self.lm.roberta.embeddings
        else:
            self.decoder_embeddings = self.lm.bert.embeddings

        self.c_head = BertLayerForDecoder(bert.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()


    def gradient_checkpointing_enable(self, **kwargs):
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # decoder_position_ids = self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        # decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
        # query = decoder_position_embeddings + cls_hiddens

        cls_hiddens = cls_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=cls_hiddens)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, 'cls'):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, 'lm_head'):
            pred_scores = self.lm.lm_head(hiddens)
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(os.path.join(output_dir, "encoder_model"))
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    def layer_freeze(self, model):
        if self.model_args.freeze_input_embeddings or self.model_args.freeze_mlm_decoder:
            for param in model.parameters():
                param.requires_grad = False
            if self.model_args.freeze_mlm_decoder:
                print("freeze mlm decoder")
                model.lm_head.decoder.weight.requires_grad = True
                if hasattr(model.lm_head.decoder, "bias"):
                    model.lm_head.decoder.bias.requires_grad = True
            if self.model_args.freeze_input_embeddings:
                print("freeze input embeddings")
                model.get_input_embeddings().weight.requires_grad = True
                if hasattr(model.get_input_embeddings(), "bias"):
                    model.get_input_embeddings().bias.requires_grad = True


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model
