#On fait un test avec phi4

import transformers

pipelin = transformers.pipeline(
	"text-generation",
	model="microsoft/phi4",
	model_kwargs="{'torch_dtype':"auto"},
	device_map:'auto'
)

messages = {
	    {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."}
}

outputs = pipeline(messages, max_new_tokens=128)
print(outputs[0]["genrated_text"][-1])


