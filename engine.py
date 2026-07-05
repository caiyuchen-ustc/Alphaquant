"""REINFORCE training loop for the formula generator.

★ bug#2 fix: the critic value IS used — as the advantage baseline (adv = reward - value),
  and it receives gradient via a value-regression loss. An entropy bonus keeps exploration.
  (Per-token credit assignment: the single formula-level reward is the return-to-go for
  every token, but the per-sample critic baseline sharply cuts variance vs the original
  broadcast-only REINFORCE.)
★ bug#3 fix: reward = robust_reward(train, valid); best tracked on VALID; TEST scored once
  in finalize().

Deterministic sampling seed (torch.manual_seed) — no wall-clock randomness.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .alphagpt import AlphaGPT
from .backtest import backtest_long_short, quick_score
from .config import Cfg
from .prepare import splits
from .reward import robust_reward, neighborhood_robustness
from .vm import StackVM


class AlphaEngine:
    def __init__(self):
        torch.manual_seed(Cfg.SEED)
        self.full, self.train, self.valid, self.test = splits()
        print(f"splits: train {self.train.feat.shape[1]}d / valid {self.valid.feat.shape[1]}d "
              f"/ test {self.test.feat.shape[1]}d, N={len(self.full.symbols)} coins", flush=True)
        self.model = AlphaGPT().to(Cfg.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=Cfg.LR)
        self.vm = StackVM()
        self.best_valid = -float("inf")
        self.best_formula = None
        self.topk = []          # list of (valid_score, formula)
        self.history = {"step": [], "avg_reward": [], "best_valid": []}

    def _reward(self, formula) -> float:
        return robust_reward(formula, self.train.feat, self.train.fwd_ret, self.train.mask,
                             self.valid.feat, self.valid.fwd_ret, self.valid.mask, self.vm)

    def sample_batch(self, bs):
        inp = torch.zeros((bs, 1), dtype=torch.long, device=Cfg.DEVICE)
        logps, ents, vals, toks = [], [], [], []
        for _ in range(Cfg.MAX_FORMULA_LEN):
            logits, value = self.model(inp)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logps.append(dist.log_prob(act))
            ents.append(dist.entropy())
            vals.append(value)
            toks.append(act)
            inp = torch.cat([inp, act.unsqueeze(1)], dim=1)
        return (torch.stack(toks, 1), torch.stack(logps, 1),
                torch.stack(ents, 1), torch.stack(vals, 1))

    def train_loop(self):
        from tqdm import tqdm
        pbar = tqdm(range(Cfg.TRAIN_STEPS))
        for step in pbar:
            seqs, logp, ent, val = self.sample_batch(Cfg.BATCH_SIZE)
            rewards = torch.zeros(Cfg.BATCH_SIZE, device=Cfg.DEVICE)
            for i in range(Cfg.BATCH_SIZE):
                f = seqs[i].tolist()
                r = self._reward(f)
                rewards[i] = r
                # track best on VALID (robust reward already valid-aware); keep top-K
                if r > self.best_valid:
                    self.best_valid = r
                    self.best_formula = f
                    tqdm.write(f"[{step}] new best valid-robust reward {r:.3f} | {f}")
            # normalize reward for advantage scale stability, then subtract critic baseline
            r_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            r_tok = r_norm.unsqueeze(1).expand_as(logp)            # return-to-go = final reward
            adv = (r_tok - val.detach())
            pg_loss = -(logp * adv).mean()
            value_loss = F.mse_loss(val, r_tok)                    # ★ critic gets gradient
            ent_bonus = -Cfg.ENTROPY_COEF * ent.mean()
            loss = pg_loss + Cfg.VALUE_COEF * value_loss + ent_bonus

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self._update_topk(seqs, rewards)
            self.history["step"].append(step)
            self.history["avg_reward"].append(float(rewards.mean()))
            self.history["best_valid"].append(self.best_valid)
            pbar.set_postfix(avgR=f"{rewards.mean():.3f}", best=f"{self.best_valid:.3f}")

    def _update_topk(self, seqs, rewards, k=12):
        arr = rewards.detach().cpu().numpy()
        best_idx = np.argsort(arr)[-3:]
        for i in best_idx:
            self.topk.append((float(arr[i]), seqs[i].tolist()))
        # dedupe by formula, keep top-k by reward
        seen, uniq = set(), []
        for sc, f in sorted(self.topk, key=lambda x: -x[0]):
            key = tuple(f)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((sc, f))
        self.topk = uniq[:k]

    def finalize(self):
        """Re-rank top-K by neighborhood robustness on valid, then score the winner ONCE on test."""
        Cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ranked = []
        for sc, f in self.topk:
            nb = neighborhood_robustness(f, self.valid.feat, self.valid.fwd_ret, self.valid.mask, self.vm)
            ranked.append((nb, sc, f))
        ranked.sort(key=lambda x: -x[0])
        winner = ranked[0][2] if ranked else self.best_formula

        def full_eval(split):
            sig = self.vm.execute(winner, split.feat)
            res, daily = backtest_long_short(sig, split.fwd_ret, split.mask)
            return res, daily

        out = {"best_formula": winner, "best_formula_readable": readable(winner)}
        for name, split in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            res, _ = full_eval(split)
            out[name] = res.__dict__
        with open(Cfg.REPORTS_DIR / "search_result.json", "w") as fp:
            json.dump(out, fp, indent=2, default=float)
        with open(Cfg.REPORTS_DIR / "training_history.json", "w") as fp:
            json.dump(self.history, fp)
        with open(Cfg.REPORTS_DIR / "topk.json", "w") as fp:
            json.dump([{"nb": nb, "reward": sc, "formula": f, "readable": readable(f)}
                       for nb, sc, f in ranked], fp, indent=2, default=float)
        return out


def readable(tokens) -> str:
    """Postfix tokens -> infix-ish readable string."""
    from .vocab import FORMULA_VOCAB
    from .ops import OPS_CONFIG
    names = FORMULA_VOCAB.token_names
    off = FORMULA_VOCAB.operator_offset
    arity = {i + off: OPS_CONFIG[i][2] for i in range(len(OPS_CONFIG))}
    stack = []
    for t in tokens:
        t = int(t)
        if t < off:
            stack.append(names[t])
        else:
            a = arity.get(t, 1)
            if len(stack) < a:
                return "<invalid>"
            args = [stack.pop() for _ in range(a)][::-1]
            stack.append(f"{names[t]}({','.join(args)})")
    return stack[0] if len(stack) == 1 else "<invalid>"


if __name__ == "__main__":
    eng = AlphaEngine()
    eng.train_loop()
    out = eng.finalize()
    print("\nBEST:", out["best_formula_readable"])
    for s in ("train", "valid", "test"):
        r = out[s]
        print(f"  {s:>5}: Sharpe {r['sharpe']:+.2f}  IC {r['ic_mean']:+.4f}  "
              f"annRet {r['ann_return']:+.1%}  maxDD {r['max_drawdown']:.1%}")
