import numpy as np
import os
from typing import List, Dict

from huggingface_hub import hf_hub_download
import pyctcdecode
import torch

from config import Configurator
from pyctcdecode import build_ctcdecoder
from llm_utils import LocalLLMCorrector


def greedy_decode(logits: torch.Tensor, blank_id=0, batch_first=False) -> List[str]:
    """
    logits: (T, B, C) raw output or (B, T, C) if batch_first
    Returns list of lists of predicted token IDs (per batch)
    """
    # We don't need to take log_softmax here, as argmax is invariant to monotonic transformations
    if batch_first:
        B, T, C = logits.shape
    else:
        T, B, C = logits.shape
    best = logits.argmax(dim=-1)  # (T, B) or (B, T)
    best = best.cpu().numpy()
    results = []
    for b in range(B):
        seq = []
        prev = blank_id
        for t in range(T):
            p = best[t, b]
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        results.append(seq)
    return results

def beam_search_decode(
        decoder: pyctcdecode.BeamSearchDecoderCTC,
        logits,
        beam_width=10,
        batch_first: bool = False,
        is_log_probs: bool = False
) -> List[str]:
    """
    Args:
        decoder: An instance of pyctcdecode's BeamSearchDecoder
        logits: The raw output from your Conformer.
                       Shape: (Time_Steps, Num_Phonemes)
                       Type: Numpy array or Torch Tensor
        beam_width: How many candidate paths to keep alive (Higher = slower but better)
        batch_first: Whether the input logits have batch dimension first
        is_log_probs: Whether the input logits are already log probabilities

    Returns:
        str: The best predicted phoneme sequence
    """
    # Convert to numpy (T', V)
    if is_log_probs:
        # Already log_softmax-ed by EnsembleModel
        log_probs = logits.cpu().detach().numpy()
    else:
        # Compute log_softmax if input is raw logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu().detach().numpy()

    # Perform beam search decoding on batch
    if log_probs.ndim == 3:
        beam_results = []
        B = log_probs.shape[0] if batch_first else log_probs.shape[1]
        for b in range(B):
            single_log_probs = log_probs[b] if batch_first else log_probs[:, b, :]
            result = decoder.decode(single_log_probs, beam_width=beam_width)
            beam_results.append(result)
    else:
        beam_results = decoder.decode(log_probs, beam_width=beam_width)

    return beam_results

def run_evaluate_text(model, dataloader, blank_id=0, device='cpu'):
    model.eval()
    cer_total, wer_total = 0.0, 0.0
    n = 0
    results = {
        "references": [],
        "hypotheses": [],
        "llm_corrected": []  # Add this to track changes
    }

    # 1. Initialize Decoder (Standard KenLM)
    local_dir = "../corpuses"
    os.makedirs(local_dir, exist_ok=True)
    if Configurator.USE_BEAM_SEARCH_DECODING and not os.path.exists(os.path.join(local_dir, "en.arpa.bin")):
        print("Downloading KenLM model for beam search decoding...")
        repo_id = "edugp/kenlm"
        filenames = ["wikipedia/en.arpa.bin", "wikipedia/en.sp.model", "wikipedia/en.sp.vocab"]
        for filename in filenames:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # Set to False to get the actual file, not a symlink
            )
            print(f"Downloaded {filename} to {model_path}")

    kenlm_model = os.path.join(local_dir, "wikipedia/en.arpa.bin")
    labels = [chr(i) for i in range(128)] + ['']  # ASCII + blank
    decoder = build_ctcdecoder(
        labels=labels,  # ASCII + blank
        kenlm_model_path=kenlm_model,
        alpha=0.5,  # Weights for LM
        beta=1.0,  # Bonus for word length (can help prevent very short outputs)
    )

    # 2. Initialize LLM (New)
    llm_corrector = None
    if Configurator.USE_LLM_CORRECTION:
        try:
            llm_corrector = LocalLLMCorrector(Configurator.LLM_MODEL_PATH)
        except Exception as e:
            print(f"Skipping LLM: {e}")

    with torch.no_grad():
        for x_pad, x_len, targets, target_len in tqdm(dataloader, desc="Evaluating"):
            # move to device
            x_pad = x_pad.to(device)
            x_len = x_len.to(device)

            # forward
            logits, _ = model(x_pad, x_len)

            # decode IDs (already CTC-decoded: blanks removed in greedy_decode)
            if Configurator.USE_BEAM_SEARCH_DECODING:
                pred_ids_batch = beam_search_decode(decoder, logits, batch_first=Configurator.BATCH_FIRST, is_log_probs=False)
            else:
                pred_ids_batch = greedy_decode(logits, blank_id=blank_id)

            # build reference & hypothesis text sequences
            for i in range(len(x_len)):
                start = sum(target_len[:i])
                end = start + target_len[i]

                ref_ids = targets[start:end].cpu().tolist()
                hyp_ids = pred_ids_batch[i]
                if isinstance(hyp_ids, str):
                    # beam search returns string, no need to convert
                    hyp_text = hyp_ids
                else:
                    hyp_text = ascii_ids_to_text(hyp_ids)

                # Apply LLM correction if enabled
                final_text = hyp_text
                if llm_corrector:
                    # Only correct if the sentence is somewhat long to avoid noise
                    if len(hyp_text) > 3:
                        corrected = llm_corrector.correct_text(hyp_text)
                        # Fallback: if LLM returns empty or hallucinated short text, keep original
                        if len(corrected) > 0:
                            final_text = corrected

                ref_text = ascii_ids_to_text(ref_ids)
                results["references"].append(ref_text)
                results["hypotheses"].append(hyp_text) # Raw beam search
                results["llm_corrected"].append(final_text)  # Corrected

                cer_total += character_error_rate(ref_text, final_text)
                wer_total += word_error_rate(ref_text, final_text)
                n += 1

    avg_cer = cer_total / n
    avg_wer = wer_total / n
    return avg_cer, avg_wer, results

def levenshtein(a, b):
    dp = np.zeros((len(a)+1, len(b)+1), dtype=np.int32)
    dp[:,0] = np.arange(len(a)+1)
    dp[0,:] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[len(a), len(b)]

def character_error_rate(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(1, len(ref))

def word_error_rate(ref: str, hyp: str) -> float:
    ref_w, hyp_w = ref.split(), hyp.split()
    return levenshtein(ref_w, hyp_w) / max(1, len(ref_w))