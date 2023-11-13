# Model Directory Structure

This repository follows a specific directory structure for organizing models based on the MODEL_PATH and MODEL_TYPE specified in the environment (env) file. Please ensure that your models are placed in the correct directories to ensure proper functionality.

## Directory Structure

The general structure is as follows:

In the example below ./models/ is set in the .env as MODEL_PATH
```
./models/
│
├── ollama/
│ ├── model1.bin
│ ├── model2.bin
│ └── ...
│
├── gpt4all/
│ ├── modelA.bin
│ ├── modelB.bin
│ └── ...
│
└── other_model_type/
├── modelX.bin
├── modelY.bin
└── ...
```