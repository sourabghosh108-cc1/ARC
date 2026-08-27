import os
import bz2
import pickle
import numpy as np


def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    scores = {}
    for g in guesses.values():
        h = hashable(g["solution"])
        x = scores.setdefault(h, [[], g["solution"]])
        x[0].append(g)
    ranked = [(getter(sc), output) for sc, output in scores.values()]
    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    return [output for _, output in ranked]


def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline - g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


def score_ensemble(guesses):
    """Interleave the two proven public rankings to increase attempt diversity."""
    ranked_prob = score_full_probmul_3(guesses)
    ranked_vote = score_kgmon(guesses)
    seen, result = set(), []
    for pair in zip(ranked_prob, ranked_vote):
        for cand in pair:
            h = hashable(cand)
            if h not in seen:
                seen.add(h)
                result.append(cand)
    for ranked in (ranked_prob, ranked_vote):
        for cand in ranked:
            h = hashable(cand)
            if h not in seen:
                seen.add(h)
                result.append(cand)
    return result


def getter_portfolio(guesses):
    """Consensus ranking for attempt_2: votes plus augmented NLL verification."""
    vote_count = len(guesses)
    beam_mean = float(np.mean([g["beam_score"] for g in guesses]))
    aug_mean = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
    aug_best = float(np.mean([np.min(g["score_aug"]) for g in guesses]))
    return (2.0 * vote_count) - (0.35 * aug_mean) - (0.15 * beam_mean) - (0.10 * aug_best)


def score_portfolio(guesses):
    return score_sum(guesses, getter_portfolio)


def score_ensemble_plus(guesses):
    """
    Public-33.89-style portfolio selector.

    Keep the strongest ensemble top guess, then choose a distinct attempt_2 from
    the portfolio ranking. This only re-ranks existing decoded beams; it does
    not add model calls or DFS time.
    """
    base = score_ensemble(guesses)
    if not base:
        return score_portfolio(guesses)

    result = [base[0]]
    seen = {hashable(base[0])}
    for ranked in (score_portfolio(guesses), score_full_probmul_3(guesses), score_kgmon(guesses), base):
        for cand in ranked:
            h = hashable(cand)
            if h in seen:
                continue
            seen.add(h)
            result.append(cand)
            if len(result) >= 2:
                break
        if len(result) >= 2:
            break

    for cand in base:
        h = hashable(cand)
        if h not in seen:
            seen.add(h)
            result.append(cand)
    return result


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
    score_ensemble,
    score_portfolio,
    score_ensemble_plus,
]


class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_ensemble_plus):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0
        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():
            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)
            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():
                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"
                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")
        else:
            print(" avg correct beam score:      n/a")
            print(" max correct beam score:      n/a")

        num_puzzles = len(num_tasks_per_puzzle)
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {
                k for k, v in selected.items()
                if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])
            }
            print(correct_puzzles)
            score = sum(1 / num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
