"""Sampling probabilities, kept independent of Torch for deterministic tests."""
from collections import defaultdict
import math


def probabilities(names, seconds, family_weights):
    def family(name):
        return "talha" if name == "talha_ahmad" else "single" if name == "single_speaker" else "podcast"
    totals = defaultdict(float)
    scores = [math.sqrt(s) for s in seconds]
    for name, score in zip(names, scores):
        totals[family(name)] += score
    if set(totals) != set(family_weights) or any(s <= 0 for s in scores):
        raise ValueError("Training groups must contain all three families with positive durations")
    norm = sum(family_weights.values())
    result = [family_weights[family(name)]/norm * score/totals[family(name)]
              for name, score in zip(names, scores)]
    wesam = [i for i, name in enumerate(names) if name == "wesam_ali"]
    guests = [i for i, name in enumerate(names) if family(name) == "podcast" and name != "wesam_ali"]
    # Give Wesam half the podcast weight; dozens of guest IDs should not drown him out.
    if wesam and guests:
        for indices in (wesam, guests):
            subtotal = sum(scores[i] for i in indices)
            for i in indices:
                result[i] = family_weights["podcast"]/norm*.5*scores[i]/subtotal
    return result
