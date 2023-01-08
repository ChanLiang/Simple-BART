# BART
A simple tool to finetune BART (on Persona-Chat dataset).

#### Environment
```bash
source env.sh
```

#### Preprocessing script

```bash
bash train_bart.sh # single round dialogue history
bash train_bart_3turn.sh # three rounds of dialogue history
```

#### Training script

```bash
bash train_bart.sh
```


#### Evaluation script
To find the best checkpoint after training
```bash
bash eval_bart.sh
```

#### Inference script
To decode response

```bash
bash infer_bart.sh
```
