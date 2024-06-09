from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
# processor = Blip2Processor.from_pretrained('/home/SceneSketch/blip2_processor')
# self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
"Salesforce/blip2-opt-2.7b",load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)  