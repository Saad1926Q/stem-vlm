# STEM VLM

## what's this about?

**end goal:** build a vision language model that can reason through and answer STEM questions, and can actually run in constrained environments (not just on massive GPUs)

this is basically me running a bunch of experiments, asking tons of questions, and figuring stuff out as I go. First time doing a lot of this stuff so it's gonna be messy but that's the point.


## the plan

rough roadmap of what I'm working through:

1. **baseline inference** - run existing models (qwen2-vl-2b) on mathvista/scienceqa, see what they can and can't do
2. **normal finetuning** - standard supervised finetuning experiments
3. **reasoning with RL** - look into making models actually reason better using reinforcement learning
4. **model compression** - quantization, pruning, distillation - make it small
5. **high performance inference** - make it fast

## file structure

```
stem-vlm/
    configs/              # yaml configs for experiments
        baseline.yaml
    data/                 # dataset download scripts
        download.py
    scripts/              # main experiment scripts
        run_baseline.py
    notebooks/            # jupyter notebooks for colab/analysis
        baseline_colab.ipynb
    experiments/          # all results saved here
        baseline/
```