# CVLP: Contrastive Visual Linguistic Pretraining

#### 1. Specification of dependencies

1.1 The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```
1.2 The data download method can refer to https://github.com/airsplay/lxmert/blob/master/README.md

1.3 After downloading the data and features from the drives, please re-organize them under data/ folder according to the following example:
```
REPO ROOT
 |
 |-- data                  
 |    |-- vqa
 |    |    |-- train.json
 |    |    |-- minival.json
 |    |    |-- nominival.json
 |    |    |-- test.json
 |    |
 |    |-- mscoco_imgfeat
 |    |    |-- train2014_obj36.tsv
 |    |    |-- val2014_obj36.tsv
 |    |    |-- test2015_obj36.tsv
 |    |
 |    |-- vg_gqa_imgfeat -- *.tsv
 |    |-- gqa -- *.json
 |    |-- nlvr2_imgfeat -- *.tsv
 |    |-- nlvr2 -- *.json
 |    |-- lxmert -- *.json          # Pre-training data
 | 
 |-- snap
 |-- src
```

1.4 Download Pre-trained model is available at :coming soon
After downloading, please move the model_LXRT.pth to the folder data/snap/pretrained/


#### 2. Training code

### Fine-tune on VQA GQA NLVR2
The fine-tuning operation process is very similar to LXMERT

### VQA

Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.

```
    bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
```

If no bug came out, then the model is ready to be trained on the whole VQA corpus:

```
    bash run/vqa_finetune.bash 0 vqa_lxr955
```

It takes around 10 hours (2 hours per epoch * 5 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/vqa/vqa_lxr955`. 
The validation result after training will be around BEST--70.30 BEST_siam--70.50

### NLVR2

Before fine-tuning on whole NLVR2 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `nlvr2_lxr955_tiny` is the name of this experiment.
Do not worry if the result is low (50~55) on this tiny split, 
the whole training data would bring the performance back.

```
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955_tiny --tiny
```

If no bugs are popping up from the previous step, it means that the code, the data, and image features are ready.
Please use this command to train on the full training set. 
The validation result after training will be around BEST--75.28 BEST_siam--76.47

```
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955
```

### GQA

Before fine-tuning on whole GQA training+validation set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `gqa_lxr955_tiny` is the name of this experiment.

```
    bash run/gqa_finetune.bash 0 gqa_lxr955_tiny --tiny
```

If no bug came out, then the model is ready to be trained on the whole GQA corpus (train + validation), and validate on 
the testdev set:

```
    bash run/gqa_finetune.bash 0 gqa_lxr955
```

#### 3. Evaluation code

### Test on VQA GQA NLVR2

### VQA

Since VQA submission system requires submitting whole test data, we need to run inference over all test splits 
(i.e., test dev, test standard, test challenge, and test held-out). 
It takes around 10~15 mins to run test inference (448K instances to run).

```
    bash run/vqa_test.bash 0 vqa_lxr955_results --test test --load snap/vqa/vqa_lxr955/BEST
```

The test results will be saved in `snap/vqa_lxr955_results/test_predict.json`. 
 
You can also use the model with moving average operation(BEST_siam.pth), It will be 0.1 more on the test-dev set than the version without moving average(BEST.pth)
 
```
    bash run/vqa_test.bash 0 vqa_lxr955_results --test test --load snap/vqa/vqa_lxr955/BEST_siam
```


### NLVR2

#### Inference on Public Test Split
1. Download NLVR2 image features for the public test split (1.6 GB).

```
    wget nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip -P data/nlvr2_imgfeat
    unzip data/nlvr2_imgfeat/test_obj36.zip -d data/nlvr2_imgfeat && rm data/nlvr2_imgfeat/test_obj36.zip
```

2. Test on the public test set (corresponding to 'test-P' on [NLVR2 leaderboard](http://lil.nlp.cornell.edu/nlvr/)) with:

```
    bash run/nlvr2_test.bash 0 nlvr2_lxr955_results --load snap/nlvr2/nlvr2_lxr955/BEST --test test --batchSize 1024
```

3. The test accuracy would be shown on the screen after around 5~10 minutes.
It also saves the predictions in the file `test_predict.csv` 
under `snap/nlvr2_lxr955_reuslts`, which is compatible to NLVR2 [official evaluation script](https://github.com/lil-lab/nlvr/tree/master/nlvr2/eval).
The official eval script also calculates consistency ('Cons') besides the accuracy.
We could use this official script to verify the results by running:

```
    python data/nlvr2/nlvr/nlvr2/eval/metrics.py snap/nlvr2/nlvr2_lxr955_results/test_predict.csv data/nlvr2/nlvr/nlvr2/data/test1.json
```

The accuracy of public test ('test-P') set should be BEST.pth--76.20, BEST_siam.pth--76.81

### GQA

Since GQA submission system requires submitting the whole test data, 
we need to run inference over all test splits.
It takes around 30~60 mins to run test inference (4.2M instances to run).

```
    bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST --test submit --batchSize 1024
```

You can also use the model with moving average operation(BEST_siam.pth), It will be 0.23 more on the test-dev set than the version without moving average(BEST.pth)

```
    bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST_siam --test submit --batchSize 1024
```

#### 4. Pre-trained model and Fine-tuning models

#### Pre-trained model  

#### VQA Fine-tuning model 

#### NlVR2 Fine-tuning model 

#### GQA Fine-tuning model 

You can download these models at this link

```
    coming soon
```

After downloading the pre-trained model and (VQA.NLVR2,GQA) fine-tuning model from the drives, please re-organize them under data/ folder according to the following example:

```
REPO ROOT
 |
 |-- data                  
 | 
 |-- snap
 |    |--pretrained
 |         |--model_LXRT.pth  
 |    |-- vqa
 |         |-- vqa_lxr955
 |                 |-- BEST.pth
 |                 |-- BEST_siam.pth 
 |    |-- gqa
 |         |-- gqa_lxr955
 |                 |-- BEST.pth
 |                 |-- BEST_siam.pth 
 |    |-- nlvr2
 |         |-- nlvr2_lxr955
 |                 |-- BEST.pth
 |                 |-- BEST_siam.pth 
 |-- src
```


#### 5. README file includes table of results accompanied by precise command to run to produce those results

## Results (with this Github version)

| Split            | [VQA](https://visualqa.org/)     | [GQA](https://cs.stanford.edu/people/dorarad/gqa/)     | [NLVR2](http://lil.nlp.cornell.edu/nlvr/)  |
|-----------       |:----:   |:---:    |:------:|
| LXMERT  | 72.42%(Test-Dev)  | 61.39%(Test-Dev)  | 74.45% (Test-P)|
| CVLP    | 72.87%(Test-Dev)  | 61.78%(Test-Dev)  | 76.81% (Test-P) |


