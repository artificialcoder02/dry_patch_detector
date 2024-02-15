#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import onnxruntime
def load_onnx_model(model_path):
       # Load the ONNX model
       session = onnxruntime.InferenceSession(model_path)
       return session

