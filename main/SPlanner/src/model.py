from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput

from .modules import Biaffine


@dataclass
class PlannerOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class TaskPlannerModel(nn.Module):
    def __init__(self, plm_dir: str, vocab_size: int = None):
        super().__init__()

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        hsz = self.plm.config.hidden_size

        

        self.start_mlp = nn.Linear(hsz, hsz)
        self.end_mlp = nn.Linear(hsz, hsz)
        self.biaffine = Biaffine(hsz, 2)

        self.loss_fn = nn.CrossEntropyLoss()

    def encoding(self, input_ids, attn_mask):
        outs = self.plm(input_ids, attn_mask, return_dict=True)
        return outs.last_hidden_state

    def get_logits(self, input_ids, mask):
        # mask: cls task sep assumption sep [step] step [step] step sep pad
        #       1   2    3   4          5   6      7    6      7    8   0
        # hidden: (bsz, seq_len, hsz)
        hidden = self.encoding(input_ids, mask.gt(0))
        start = self.start_mlp(hidden)
        end = self.end_mlp(hidden)
        # logits: (bsz, 2, seq_len, seq_len)
        logits = self.biaffine(start, end)
        return logits

    def forward(self, input_ids, mask, labels=None):
        logits = self.get_logits(input_ids, mask)
        _logits = logits.permute(0, 2, 3, 1).reshape(-1, 2)
        loss = self.loss_fn(_logits, labels.reshape(-1))

        return PlannerOutput(loss=loss, logits=logits)
