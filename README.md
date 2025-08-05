## News
- **[2025.07.20]** Release the scripts and the remaining code.
- **[2025.01.21]** Release the code for inference and evaluation.

## 🔨 Preparations

```bash
$ git clone https://github.com/zxthesky/ParaBench.git
$ cd ParaBench
$ pip install -r requirements.txt
```

## 🍰 Get started

Our test data can be found in `data/test.json`.
Our train data can be found in `data/train_data_3000.json`.

### Data construction

If you want to experience our data construction method, please follow the steps:
1. Arrive at the corresponding directory

```bash
cd main/src/create_data.
```

2. You need to use 'python use_api_create_data.py' to generate some data as an icl example.

```bash
python use_api_create_data.py
```

3. You need to use the data generated in step 2 and combine it with ‘generate_data_dynamic_icl.py’ to generate the final data.

```bash
python generate_data_dynamic_icl.py
```

### Build evaluation settings

1. First you need to download the corresponding test data.
2. You need to get the inference results of the small model first through the following code

```bash
cd main/SPlanner
bash scripts/train.sh
```

### Evaluation
1. First you need to determine the prediction results of the small model and obtain
2. Start the Evaluation process:

```bash
cd main/src/eval
python test_LLM.py
```



