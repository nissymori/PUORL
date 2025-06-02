# Offline Reinforcement Learning with Domain-Unlabeled Data

Official implementation of [Offline Reinforcement Learning with Domain-Unlabeled Data](https://arxiv.org/abs/2404.07465). The offline RL implementations are borrowed from [JAX-CORL](https://github.com/nissymori/JAX-CORL).

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

### Wandb results for reported result
- [td3bc](https://api.wandb.ai/links/nissymori/bggvrl71)
- [iql](https://api.wandb.ai/links/nissymori/y1araz5i)
- [classifier](https://api.wandb.ai/links/nissymori/sp87jczy)

### Citeation
```
@inproceedings{nishimorioffline,
  title={Offline Reinforcement Learning with Domain-Unlabeled Data},
  author={Nishimori, Soichiro and Cai, Xin-Qiang and Ackermann, Johannes and Sugiyama, Masashi},
  year={2025},
  booktitle={Reinforcement Learning Conference}
}
```


