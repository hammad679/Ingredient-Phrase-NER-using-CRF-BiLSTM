# Ingredient Phrase NER using CRF-BiLSTM Model

## Deploying using Docker
- `docker build --tag ner-model-api . --file Dockerfile.dev`
- `docker run --name my_api -p 8000:80 -d ner-model-api`

## How to use
Send `POST` requests to `<hostname>:<port>/api/predict` with a JSON request body

### JSON Body Example
```
{
	"inputs": ["2 cloves of ginger", "2 tablespoons of garlic paste", "2 teaspoons of hot milk", "salt"]
}
```

## Known limitations
- Works well on single ingredient phrases, a single sentence with multiple ingredient phrases is prone to mispredictions
- Expected ingredient phrase structure is, `<QUANTITY> ...`, the rest of the tokens should be followed by the `<QUANTITY>` for more accurate predictions

## Authors
