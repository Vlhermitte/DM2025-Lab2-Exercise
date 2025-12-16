import logging
import shutil
import os
import pandas as pd

try:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
except ImportError:
    Llama = None


class LocalLLMCorrector:
    def __init__(self, model_path: str = None, context_window: int = 2048):
        if Llama is None:
            raise ImportError("Please run: pip install llama-cpp-python")
        if model_path is None or not os.path.exists(model_path):
            print("No model path provided. Loading default Llama 3 8B Instruct model from Hugging Face...")
            repo_id = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
            filename = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            local_dir = "../models/llama3-8b-instruct-gguf"
            if os.path.exists(local_dir):
                print(f"Model found at {local_dir}. Using cached version.")
                model_path = os.path.join(local_dir, filename)
            else:
                os.makedirs(local_dir, exist_ok=True)
                print(f"Downloading {filename} from {repo_id}...")
                # Download the file
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,  # Set to False to get the actual file, not a symlink
                )
                print(f"Download complete! File saved to: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_gpu_layers=-1,  # Set to 0 if no GPU, -1 for all on GPU
            verbose=False
        )

    def correct_text(self, noisy_text: str) -> str:
        """
        Uses Llama to fix grammar/spelling of the noisy text.
        """
        # System prompt prevents the model from being chatty
        system_prompt = (
            "You are a helpful assistant that corrects text output from a "
            "Brain-Computer Interface. The input text is phonetically correct "
            "but may have grammar or spelling errors. \n"
            "INSTRUCTIONS:\n"
            "1. Fix spelling and grammar.\n"
            "2. Do not change the meaning.\n"
            "3. Do not add new words.\n"
            "4. Output ONLY the corrected text. Do not add quotes or explanations."
        )

        user_prompt = f"Correct this text: {noisy_text}"

        # Llama 3 instruct format
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        output = self.llm(
            prompt,
            max_tokens=200,  # B2T sentences usually aren't huge paragraphs
            stop=["<|eot_id|>", "\n"],
            temperature=0.1,  # Low temp for deterministic corrections
        )

        result = output['choices'][0]['text'].strip()
        return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv("../submissions/submissions.csv")
    # Example usage
    corrector = LocalLLMCorrector()


    for sentence in df['text'].tolist():
        corrected_sentence = corrector.correct_text(sentence)
        print("Noisy Sentence: ", sentence)
        print("Corrected Sentence: ", corrected_sentence)
        df.loc[df['text'] == sentence, 'corrected_text'] = corrected_sentence

    df.to_csv("../submissions/submissions_corrected.csv", index=False)

    # noisy_sentence = "Like brick d be and that kind of thing."
    # corrected_sentence = corrector.correct_text(noisy_sentence)

    # print("Noisy Sentence: ", noisy_sentence)
    # print("Corrected Sentence: ", corrected_sentence)
