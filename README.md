# DapStep: Deep Assignee Prediction for Stack Trace Error rePresentation

# Data

## Format 

The data consists of bug reports, annotations, and labels.
The **bug report** is presented in the following JSON:
```
{
  "id": 4524,
  "timestamp": "1235540669458",
  "elements": [
    {
      "name": "javax.swing.Timer.fireActionPerformed",
      "file_name": "Timer.java",
      "line_number": 313,
      "commit_hash": 8138bd7e342810cfdc5174dcb222d136bd229288,
      "subsystem": "javax.swing"
    },
    {
      "name": "java.awt.event.InvocationEvent.dispatch",
      "file_name": "InvocationEvent.java",
      "line_number": 311,
      "commit_hash": eb2vb45e75108e0c8d0c44e80c78e9d061a11448,
      "subsystem": "java.awt.event"
    },
  ]
}
```
where
- `id` – identifier of report
- `timestamp` – timestamp of report creation (Unix time)
- `elements` – sequence of stack frames starting from the top of the stack

The **annotation** is presented in CSV file: 
```
commit_hash,author,timestamp
580dsbcfd18374a14575b65085ba46adea8b015d,24,1108830319000
9d78b417254b099c9d3ae349cfbeed0e4d8efa42,631,1341273406000
9d26b217254b099c9d3ae919cfbvbd0e4d8efa42,123,1641999406000
```
where 
- `commit_hash` – commit hash of last edit
- `author` – identifier of developer
- `timestamp` – timestamp of last edit (Unix time)

The **labels** CSV file looks like this:
```
rid,uid
234,116
54,47
4,116
```
where 
- `rid` – identifier of report
- `uid` – identifier of developer

# Usage

### Install
```
pip install -r requirements.txt
```

### Train ranking DL-based models

The example of train config is in ```src/scripts/configs/dl_ranking.yaml``` and has the following form:
```
data_dir: data_dir
features_dir: features_dir
save_dir: save_dir

data_split:
  val_size: 0
  test_size: 1500

coder:
  entry_coder_type: file_name
  cased: True
  trim_len: 0
  rem_equals: False

model:
  emb_type: cnn

optimizer:
  lr: 0.001
  weight_decay: 0.001

train:
  epochs: 10
  update_every: 4
```
where 
- `data_dir` – directory where data is stored. 
  The directory should have the following structure: 
  - `reports` folder with all reports in JSON format 
  - `files` folder with all annotations in CSV format
  - `labels.csv` file with report labels
- `features_dir` – directory with features. 
  The example of features directory is ```src/scripts/features_examples```. 
  The lists of all features are presented in ```src/features/features_maps.py```
- `save_dir` – directory where the model files will be saved
- `data_split` – time-based data splitting
- `coder` – parameters of sequence coder 
- `model` – model type (```cnn``` or ```rnn```)
- `optimizer` – parameters of optimizer
- `train` – parameters of training

Run train: 
```
cd src/scripts/train/
python dl_manual_features_ranking.py
```
or 
```
cd src/scripts/train/
python dl_neural_features_ranking.py
```

The trained model will be saved in ```save_dir```.

### Run ranking DL-based models

Run eval: 

```
cd src/scripts/eval/
python eval.py --data_dir=<DATA_DIR> 
--model_dir=<MODEL_DIR> --features_dir=<FEATURES_DIR> 
--test_size=<TEST_SIZE>
```