from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.models

from torchaudio.models import Conformer

# --- Transducer Models --_

# Wrapper for Encoder
class MyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, n_days, dropout=0.1, conformer=False):
        super().__init__()
        self.conformer = conformer
        self.num_layers = layers
        self.hidden_dim = hidden_dim
        self.n_days = n_days

        # --- Day-Specific Input Layer (Based on Baseline) ---
        # Initialize with Identity so it starts neutral
        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.eye(input_dim)) for _ in range(n_days)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, input_dim)) for _ in range(n_days)
        ])
        self.day_activation = nn.Softsign()  # Baseline uses Softsign
        self.day_dropout = nn.Dropout(dropout)
        # ----------------------------------------------------

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        if self.conformer:
            self.encoder = torchaudio.models.Conformer(
                input_dim=hidden_dim,  # Must match output of projection
                num_heads=4,
                ffn_dim=hidden_dim * 4,  # Standard is 4x hidden_dim
                num_layers=layers,
                depthwise_conv_kernel_size=31,  # Good for 20ms bins (~600ms context)
                dropout=dropout
            )
        else:
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )

        self.output_norm = nn.Linear(hidden_dim if conformer else hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, day_indices: torch.Tensor):
        """
        x: (Batch, Time, Features)
        day_indices: (Batch,) -> The ID of the day for each sample in batch
        """
        # 1. Apply Day-Specific Transformation
        # Gather the weights/biases for the days in this batch
        # day_weights shape: (Batch, Features, Features)
        # day_biases shape:  (Batch, 1, Features)
        batch_weights = torch.stack([self.day_weights[i] for i in day_indices], dim=0)
        batch_biases = torch.stack([self.day_biases[i] for i in day_indices], dim=0).unsqueeze(1)

        # Apply: x * W + b
        # x is (B, T, D), W is (B, D, D) -> einsum 'btd,bdk->btk'
        x = torch.einsum("btd,bdk->btk", x, batch_weights) + batch_biases
        x = self.day_activation(x)
        x = self.day_dropout(x)

        # 2. Standard Projection & Encoding
        x = self.input_proj(x)

        if self.conformer:
            x, lengths = self.encoder(x, lengths)
        else:
            x, _ = self.encoder(x)

        x = self.output_norm(x)
        x = self.layer_norm(x)

        return x, lengths

    def __str__(self):
        return f"{'Conformer' if self.conformer else 'LSTM'} Encoder(layers={self.num_layers}, hidden_dim={self.hidden_dim})"

# Wrapper for Predictor
class MyPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_dim, blank_id):
        super().__init__()
        self.blank_id = blank_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=blank_id)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor, state=None):
        if state is not None:
            # Just embed and pass to LSTM. Do NOT add SOS here.
            x = self.embedding(input)
            x, state = self.lstm(x, state)
            # x = self.layer_norm(x)
            return x, lengths, state

        # Torchaudio expects: (input, lengths, state) -> (output, lengths, state)
        # 1. Create a tensor of SOS tokens (B, 1)
        sos = torch.full(
            (input.size(0), 1),
            self.blank_id,
            dtype=input.dtype,
            device=input.device
        )

        # 2. Concatenate [SOS, Input] -> Shape becomes (B, L+1)
        # Example: Targets [A, B, C] become [SOS, A, B, C]
        dec_input = torch.cat([sos, input], dim=1)

        # 3. Pass through Embedding and LSTM
        x = self.embedding(dec_input)
        x, state = self.lstm(x, state)

        # x = self.layer_norm(x)

        # Output x shape is now (B, L+1, Hidden)
        return x, lengths, state

# Wrapper for your Joiner
class MyJoiner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, source_encodings, source_lengths, target_encodings, target_lengths):
        # Helper to broadcast dimensions automatically
        # source: (B, T, D) -> (B, T, 1, D)
        # target: (B, U, D) -> (B, 1, U, D)
        src_expand = source_encodings.unsqueeze(2)
        tgt_expand = target_encodings.unsqueeze(1)

        # Joint operation: GELU(Enc + Pred)
        joint = F.gelu(src_expand + tgt_expand)
        output = self.linear(joint)
        return output, source_lengths, target_lengths

class RNNT(torchaudio.models.RNNT):
    def __init__(self, transcriber, predictor, joiner):
        super().__init__(transcriber, predictor, joiner)
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner


    def forward(self, source, source_lengths, target, target_lengths, day_indices, state=None):
        # Pass day_indices to the transcriber (Encoder)
        # source becomes the "x" input, day_indices is the new arg
        src_enc, src_len = self.transcriber(source, source_lengths, day_indices)

        # Calculate Predictor output (Target embedding)
        tgt_enc, tgt_len, _ = self.predictor(target, target_lengths, state)

        # Joint
        return self.joiner(src_enc, src_len, tgt_enc, tgt_len)

    def __str__(self):
        return "rnnt_model"

    def describe(self):
        return ("RNN-T Model with "
                f"Transcriber: {self.transcriber}, "
                f"Predictor: {self.predictor.__class__.__name__}, "
                f"Joiner: {self.joiner.__class__.__name__}")


def predict_on_sample(model, x, vocab_map, device='cuda'):
    """
    Args:
        model: The trained RNNT model
        x: Input features (Tensor shape: [T, Input_Dim])
        vocab_map: Dictionary mapping ID -> Char
        device: 'cuda' or 'cpu'
    """
    model.eval()
    x = x.to(device)

    # 1. ENCODER PASS
    # Fake a batch dimension: (T, D) -> (1, T, D)
    x_in = x.unsqueeze(0)
    x_len = torch.tensor([x_in.size(1)], dtype=torch.int32, device=device)

    with torch.no_grad():
        encoder_out, _ = model.transcriber(x_in, x_len)

    # 2. INITIALIZE PREDICTOR STATE MANUALLY
    # We must construct the initial state (h_0, c_0) to force MyPredictor
    # into "inference mode" (skipping the automatic SOS prepend).

    # LSTM State shape: (num_layers * num_directions, batch, hidden_size)
    # Assuming standard LSTM params from your Config: 1 layer, Unidirectional
    hidden_dim = model.predictor.lstm.hidden_size
    h_0 = torch.zeros(1, 1, hidden_dim, device=device)
    c_0 = torch.zeros(1, 1, hidden_dim, device=device)
    predictor_state = (h_0, c_0)

    # Create the first input: [[blank_id]]
    blank_id = model.predictor.blank_id
    last_token = torch.full((1, 1), blank_id, dtype=torch.long, device=device)

    # Get first embedding using the MANUAL state
    # pred_out shape: (1, 1, Hidden)
    pred_out, _, predictor_state = model.predictor(last_token, None, state=predictor_state)

    decoded_indices = []
    T = encoder_out.size(1)
    t = 0
    max_symbols_per_step = 30

    # 1. Check Predictor Embedding
    pred_vocab = model.predictor.embedding.num_embeddings
    assert pred_vocab == len(vocab_map), "ERROR: Predictor Embedding size does not match vocab_map size."

    # 2. MAIN LOOP
    while t < T:
        symbols_emitted = 0

        while symbols_emitted < max_symbols_per_step:
            # A. JOINER
            # Slice encoder to keep dimensions: (1, 1, Enc_Dim)
            enc_frame = encoder_out[:, t:t + 1, :]

            # logits shape: (1, 1, 1, Vocab)
            logits, _, _ = model.joiner(enc_frame, None, pred_out, None)

            # --- CRITICAL DEBUG BLOCK ---
            # 1. Check for NaNs
            if torch.isnan(logits).any():
                print(f"!!! FATAL: NaNs detected in Logits at step t={t} !!!")
                print(f"Encoder Frame Max: {enc_frame.max()}")
                print(f"Predictor Out Max: {pred_out.max()}")
                raise ValueError("Model is outputting NaNs. Weights likely corrupted.")

            # B. GREEDY SEARCH
            # View as (Vocab) to be safe against squeezing dims with size 1
            log_probs = torch.log_softmax(logits.view(-1), dim=0)
            predicted_id = torch.argmax(log_probs).item()

            # 1. Handle BLANK first
            if predicted_id == blank_id:
                t += 1
                break # Break inner loop, advance time t

            # 2. Safety Check: Vocab Limit
            vocab_limit = model.predictor.embedding.num_embeddings
            if predicted_id >= vocab_limit:
                 print(f"PREDICTION ERROR: Model predicted {predicted_id}, limit is {vocab_limit}")
                 return ""

            # 3. Handle Valid Token (Non-Blank)
            if predicted_id in vocab_map:
                decoded_indices.append(predicted_id)
            else:
                # Only warn here if it's NOT blank and NOT in map
                print(f"WARNING: Predicted ID {predicted_id} not in vocab_map at step t={t}")
                continue  # Skip adding to decoded indices

            # Feed new token to predictor
            next_input = torch.tensor([[predicted_id]], dtype=torch.long, device=device)
            pred_out, _, predictor_state = model.predictor(
                next_input,
                None,
                state=predictor_state
            )

            symbols_emitted += 1

        if symbols_emitted >= max_symbols_per_step:
            t += 1

    # 3. CONVERT TO TEXT
    predicted_text = "".join([vocab_map.get(i, '') for i in decoded_indices])

    return predicted_text