from pathlib import Path

from module.inference import TransformerInference


model_folder = Path(__file__).parent / 'model'

model = TransformerInference(
    input_tokenizer_path=model_folder / 'source_tokenizer.tok',
    output_tokenizer_path=model_folder / 'target_tokenizer.tok',
    model_state_dict_path=model_folder / 'best_model_state_dict.pt',
    device='cuda'
)
