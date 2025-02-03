# PUORL

Official implementation of Positive and Unlabeled Offline Reinforcement Learning (PUORL). Baseline implementation is [here](https://github.com/nissymori/cdorl_baseline.git).

### How to install
```
git clone https://github.com/nissymori/PUORL.gitn
```

```
pip install -r requirements.txt
```

### How to reproduce

1. Download data
```
python download.py
```

2. Remove prefix
```
./remove_prefix.sh
```

3. Train classifier at `sh/classifier`
```
./pu.sh
```

4. Train agent  at `sh/offline/...`
```
./body_mass.sh
```

### Citeation


