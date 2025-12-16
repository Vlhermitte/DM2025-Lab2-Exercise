import torch
from pathlib import Path


class Configurator:
    """Configuration class for application settings."""

    file_localpath = Path(__file__).resolve()

    BLANK_ID = 0

    # Note: We use a string tag for Blank to keep the list consistent, 
    # but we know index 0 is blank.
    # VOCABULARY = CHARACTER
    # VOCAB_SIZE = len(VOCABULARY)

    # CHAR_TO_ID = {c: i for i, c in enumerate(VOCABULARY)}
    # ID_TO_CHAR = {i: c for i, c in enumerate(VOCABULARY)}

    PAD_ID = BLANK_ID

    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # Force CPU on Mac for now to avoid MPS ops errors

    DEBUG = True
    TRAIN = False
    DATA_PATH = str(
        Path(file_localpath).parent.parent
        / "brain-to-text-25"
        / "t15_copyTask_neuralData"
        / "hdf5_data_final"
    )

    BATCH_SIZE = 8
    FEATURE_LEN = 512
    BATCH_FIRST = True

    # LLM config
    USE_LLM_CORRECTION = False  # Keep FALSE during training
    LLM_MODEL_PATH = str(
            Path(file_localpath).parent.parent /
            "models/llama3-8b-instruct-gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    )
    MODEL_STR = "seq2seq_model.pt"
    MODEL_PATH = str(Path(file_localpath).parent.parent / "models" / "text_models")

    def __init__(self):
        self.ID_TO_CHAR = None
        self.CHAR_TO_ID = None
        self.VOCABULARY = None
        self.VOCAB_SIZE = None

    def set_vocabulary(self, unique_ids):
        """
        Creates a DENSE mapping.
        0 -> <blank>
        1 -> first char
        2 -> second char
        ...
        """
        # 1. Reset maps
        self.ID_TO_CHAR = {0: '<blank>'}
        self.CHAR_TO_ID = {'<blank>': 0}

        # 2. Sort unique characters (excluding 0 if it exists in raw data)
        # This ensures 'a' always gets the same ID every run
        valid_ascii = sorted([x for x in unique_ids if x != 0])

        # 3. Create Dense Mapping
        # ASCII 97 becomes ID 1, ASCII 98 becomes ID 2, etc.
        current_id = 1
        for ascii_val in valid_ascii:
            char = chr(ascii_val)
            self.ID_TO_CHAR[current_id] = char
            self.CHAR_TO_ID[char] = current_id

            # OPTIONAL: Map the raw integer too, for safety in Dataset
            self.CHAR_TO_ID[ascii_val] = current_id

            current_id += 1

        self.VOCABULARY = list(self.CHAR_TO_ID.keys())
        self.VOCAB_SIZE = len(self.ID_TO_CHAR)

        print(f"Vocabulary Fixed. Size: {self.VOCAB_SIZE}")
        print(f"Mapping Check: ASCII 97 ('a') maps to Dense ID {self.CHAR_TO_ID.get(97, 'N/A')}")