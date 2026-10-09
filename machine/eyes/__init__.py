"""His eyes (L15.2): two models of his "strong zone" answers, measured honestly.

He answered 195 zone charts on Telegram and 4 of his drawn cards (L15.1, T9): "is the brown band YOUR kind of zone -
the quiet one you trade?" yes / no. Two models learn those answers:

    model A   LightGBM on numbers that describe the chart (machine/eyes/features.py)
    model B   ResNet18 - a picture network trained on ImageNet photos - taught again on the pictures he answered
              (machine/eyes/model_b.py, PyTorch + torchvision)

Measured by TIME (machine/eyes/split.py): the charts of 2018 are guessed by models that learned only from 2016-2017,
2019 from 2016-2018, 2020 from 2016-2019 - 120 held-out answers. Next to them, always: "always no" and his own written
rule on the same chart (calm at most 0.74). Every accuracy with its 90% interval (machine/eyes/measure.py).

    python -m machine.eyes exams     made-up answers first: noise, and his written rule planted (plan/work/eyes/exams.md)
    python -m machine.eyes plan      writes plan/work/eyes/plan.json + plan.md - COMMIT IT before the run
    python -m machine.eyes run       scores his held-out answers ONCE (refuses without the committed plan) -> the trial
                                     files, plan/work/eyes/runs.jsonl, plan/work/eyes-model.md + pictures of mistakes
    python -m machine.eyes final     both models taught on all 199 answers -> data/eyes/ (for later prompts)
"""
