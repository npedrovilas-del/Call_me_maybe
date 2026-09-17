from llm_sdk import Small_LLM_Model
import numpy as no
import json

m = Small_LLM_Model()
ids = m.encode("mae").tolist()[0]
logits = m.get_logits_from_input_ids(ids)
print(len(logits) == 151000)
next_id = int(no.argmax(logits))
print("len(logits):", len(logits))  
print("before:", m.decode([ids]))
print("next id", next_id)
print("model: ", m.decode([next_id]))
