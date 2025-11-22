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

### quick summary of the experiments so far

- **baseline (no finetuning):**  
  tested qwen2-vl-2b straight out of the box.  
  **ScienceQA: 2.68%**, **MathVerse: 11.02%** → basically guessing.

- **finetuning on just the correct option:**  
  trained the model to predict only the answer label.  
  **ScienceQA: 29.95%**, **MathVerse: 0.71%**.  
  I noticed that the model was outputting “1” a lot, so I had my doubts.

- **cot (reasoning) analysis:**  
  forced the model to produce reasoning + evaluated it separately.  
  answer accuracy ~27%, reasoning ~23% on ScienceQA.
  Saw that whenever the answer is wrong, the reasoning is also wrong.

- **finetuning on full solutions:**
  trained the model on both the reasoning process and the final answer for ScienceQA.
  **Total samples: 2017**
  **Answer Accuracy: 11.20%**, **Reasoning Accuracy: 18.00%**
  Breakdown: Both Correct: 114 (5.7%), Answer Correct + Reasoning Wrong: 112, Answer Wrong + Reasoning Correct: 249, Both Wrong: 1542

- **next steps:**
  move on to bigger models, also explore the idea from MM-COT paper
