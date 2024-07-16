import uvicorn
from fastapi import FastAPI, Query

from model import model
from schemas import Translated


app = FastAPI()


@app.get('/translate', response_model=Translated)
def translate(
    text: str = Query(..., min_length=1, max_length=300)
) -> Translated:
    return Translated(input=text, output=model.generate(text))


@app.get('/translate/beam')
def translate_beam(
    text: str = Query(..., min_length=1, max_length=300),
    beam_size: int = 5
) -> Translated:
    return Translated(input=text, output=model.beam(text, beam_size))


@app.get('/translate/text', response_model=Translated)
def translate_text(text: str = Query(..., min_length=1, max_length=10000)):
    generated = model.generate_many(text)
    return {'input': text, 'output': generated} 


if __name__ == '__main__':
    uvicorn.run(app)
