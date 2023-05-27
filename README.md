# Solving NP-hard Min-max Routing Problems as Sequential Generation with Equity Context

## Dependencies

* Python>=3.8
* NumPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm

## Training

To train the model on MTSP and MPDP instances with 50 nodes and a variable number of agents ranging from 2 to 10:

```bash
python run.py --graph_size 50 --problem mtsp --run_name 'mtsp50' --agent_min 2 --agent_max 10
```

```bash
python run.py --graph_size 50 --problem mpdp --run_name 'mpdp50' --agent_min 2 --agent_max 10
```

## Generating data
To generate validation and test data (same as used in the paper) for mtsp and mpdp problems
```bash
python generate_data.py --problem mtsp --graph_sizes 50 --dataset_size 100 --name test --seed 3333 
python generate_data.py --problem mpdp --graph_sizes 50 --dataset_size 100 --name test --seed 3333
```

## Finetuning
You can finetuning using a pretrained model by using the `--load_path` option:

```bash
python run.py --graph_size 200 --problem mtsp --load_path pretrained/mtsp/mtsp50/epoch-100.pt --batch_size 64 --agent_min 10 --agent_max 20 --ft Y
```

## Evalutation

### Serialization


- MTSP
    ```bash
    python eval.py data/mtsp/mtsp200_test_seed3333.pkl --problem mtsp --model pretrained/mtsp/mtsp200/epoch-4.pt --decode_strategy greedy --agent_num 10  --is_serial True --val_size 100 --ft Y
    ```
- MPDP
    ```bash
    python eval.py data/mpdp/mpdp200_test_seed3333.pkl --problem pdp --model pretrained/mpdp/mpdp200/epoch-0.pt --decode_strategy greedy --agent_num 10  --is_serial True --val_size 100 --ft Y
    ```

### Parallelization

You can adjust batch_size to `--eval_batch_size` argument if memory is insufficient when you run parrellel

- MTSP
    ```bash
    python eval.py data/mtsp/mtsp200_test_seed3333.pkl --problem mtsp --model pretrained/mtsp/mtsp200/epoch-4.pt --decode_strategy greedy --agent_num 10  --is_serial False --val_size 100 --eval_batch_size 25 --ft Y 
    ```
- MPDP
    ```bash
    python eval.py data/mpdp/mpdp200_test_seed3333.pkl --problem pdp --model pretrained/mpdp/mpdp200/epoch-0.pt --decode_strategy greedy --agent_num 10 --eval_batch_size 25 --is_serial False --val_size 100 --ft Y
    ```

 