import torch
import torch.nn as nn
import torchaudio
try:
    from speechbrain.nnet.loss.transducer_loss import TransducerLoss
except ImportError:
    print("speechbrain is not installed. FastRNNTLoss will not be available.")

class CTCLoss(nn.Module):
    def __init__(self, blank_id, reduction='mean'):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.ctc = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc(log_probs, targets, input_lengths, target_lengths)

    def __str__(self):
        return "CTCLoss"

class FocalCTCLoss(nn.Module):
    def __init__(self, blank_id, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.blank_id = blank_id
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ctc = nn.CTCLoss(blank=blank_id, reduction='none', zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 1. Calculate standard CTC Loss (unreduced)
        # log_probs: (T, B, C)
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        # 2. Calculate "confidence" (p_t) of the model
        # We estimate p_t using exp(-loss) because CTC loss is -log(p(target))
        p_t = torch.exp(-ctc_loss)

        # 3. Apply Focal Weighting: alpha * (1 - p_t)^gamma * loss
        # This reduces loss for examples where model is already confident (p_t -> 1)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ctc_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def __str__(self):
        return "FocalCTCLoss"

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits, targets):
        return self.ce(logits, targets)

    def __str__(self):
        return "CrossEntropyLoss"

class RNNTLoss(nn.Module):
    def __init__(self, blank_id, reduction='mean'):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.rnnt = torchaudio.transforms.RNNTLoss(
            blank=blank_id,
            reduction=reduction,
            fused_log_softmax=True,
            clamp=1.0
        )

    def forward(self, logits, targets, input_lengths, target_lengths):
        return self.rnnt(logits, targets, input_lengths, target_lengths)

    def __str__(self):
        return "RNNTLoss"

# class FastRNNTLoss(nn.Module):
#     def __init__(self, blank_id, reduction='mean'):
#         super().__init__()
#         self.blank_id = blank_id
#         self.reduction = reduction
#         self.rnnt = TransducerLoss(blank=blank_id, reduction=reduction)
#
#     def forward(self, logits, targets, input_lengths, target_lengths):
#         return self.rnnt(
#             logits=logits,
#             labels=targets,
#             T=input_lengths,
#             U=target_lengths
#         )
#
#     def __str__(self):
#         return "FastRNNTLoss"