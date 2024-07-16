import pprint
from module.inference import TransformerInference


def translate(input: str) -> str | list[dict]:
    model = TransformerInference(
        input_tokenizer_path=r'C:\Users\Ноутбук\Desktop\enviroment\transformer\model\source_tokenizer.tok',
        output_tokenizer_path=r'C:\Users\Ноутбук\Desktop\enviroment\transformer\model\target_tokenizer.tok',
        device='cuda',
        model_state_dict_path=r'C:\Users\Ноутбук\Desktop\enviroment\transformer\model\best_model_state_dict.pt'
    )
    
    return model.beam(input)
    

if __name__ == '__main__':
    text = 'I participate there.'
    pprint.pprint(translate(text))
