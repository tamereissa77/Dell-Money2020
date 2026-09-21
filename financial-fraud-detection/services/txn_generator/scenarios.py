"""Scripted typologies the presenter can inject on demand.

Every scenario is a **curated sequence of real rows from the test set**, not a
fabricated transaction. That constraint is not cosmetic: scoring-svc looks up
feature vectors by row index, so a synthetic transaction would have no features
and could not be scored. Selecting real rows that already exhibit a pattern
keeps the demo honest — what you see scored is a transaction the model was
genuinely evaluated on.

Each builder returns a list of row indices, emitted back-to-back in order.

`mule_fanin_fanout` carries acceptance criterion 8, so it is built to a harder
spec than the others: the rows must be ones the incumbent's rules do NOT fire
on, otherwise the "caught by the model, missed by screening" claim is not
actually demonstrated. See `_evades_rules`.
"""
import collections
import numpy as np

# Kept in step with services/screening_stub/screening.py. If the stub is
# retuned, this must follow, or mule-fanin-fanout silently stops evading it.
STUB_AMOUNT = 500.0
STUB_ROUND = {100.0, 200.0, 250.0, 500.0, 1000.0}
STUB_WATCHLIST = {"Rome", "Naples", "Moscow", "Lagos", "Caracas",
                  "Tehran", "Karachi", "Odessa"}


class Catalogue:
    """Indexes the test set once at startup so injection is instant."""

    def __init__(self, disp, y, e_ut, e_tm, seed=20260920):
        self.disp, self.y = disp, y
        self.user = e_ut[0]
        self.merch = e_tm[1]
        self.rng = np.random.RandomState(seed)

        self.by_user = collections.defaultdict(list)
        self.by_merch_users = collections.defaultdict(set)
        for i in range(len(disp)):
            self.by_user[int(self.user[i])].append(i)
            self.by_merch_users[int(self.merch[i])].add(int(self.user[i]))

        d = disp
        self.amount = d["Amount"].astype(float).values
        self.city = d["City"].astype(str).str.strip().values
        self.chip = d["Chip"].astype(str).str.strip().values
        self.errors = d["Errors"].astype(str).str.strip().values
        self.mcc = d["MCC"].fillna(0).astype(int).values

        self._mule_tier = None
        self.built = {}
        for name, fn in (("geo-velocity", self._geo_velocity),
                         ("cnp-burst", self._cnp_burst),
                         ("account-takeover", self._account_takeover),
                         ("mule-fanin-fanout", self._mule),
                         ("merchant-city-anomaly", self._city_anomaly)):
            try:
                self.built[name] = fn()
            except Exception as e:                       # a thin slice is better
                self.built[name] = []                    # than a crashed startup
                print(f"[gen] scenario {name} unavailable: {e}", flush=True)

    # --- helpers ----------------------------------------------------------
    def _present(self, i):
        return "online" not in self.chip[i].lower()

    def _evades_rules(self, i):
        """True when the incumbent's rules would NOT fire on this row.

        Velocity (R003) is not checked here: fan-in spreads its rows across
        distinct cards, so per-card velocity stays low by construction.
        """
        if self.amount[i] >= STUB_AMOUNT:            # R001
            return False
        if self.city[i] in STUB_WATCHLIST:           # R002
            return False
        if self.amount[i] in STUB_ROUND:             # R004
            return False
        err = self.errors[i] not in ("XX", "nan", "")
        if "online" in self.chip[i].lower() and err:  # R005
            return False
        return True

    # --- typologies -------------------------------------------------------
    def _geo_velocity(self):
        """Impossible travel: two card-present transactions, same card,
        different cities, emitted back-to-back."""
        for u, rows in self.by_user.items():
            pres = [i for i in rows if self._present(i)
                    and self.city[i] not in ("ONLINE", "", "nan")]
            if len(pres) < 2:
                continue
            for a in pres:
                for b in pres:
                    if a != b and self.city[a] != self.city[b] \
                       and (self.y[a] == 1 or self.y[b] == 1):
                        return [a, b]
        return []

    def _cnp_burst(self):
        """Card-not-present testing burst: same card, online, escalating
        amounts — the classic card-testing shape.

        Only 1,608 of 25,803 rows are online, spread over 4,795 cards, so
        insisting on a card with 3+ online rows AND a fraud among them finds
        nothing. Prefer a card that has both; fall back to the longest online
        run available, which still shows the escalating-amount shape.
        """
        with_fraud, longest = [], []
        for u, rows in self.by_user.items():
            cnp = [i for i in rows if "online" in self.chip[i].lower()]
            if len(cnp) < 2:
                continue
            cnp.sort(key=lambda i: self.amount[i])
            if any(self.y[i] == 1 for i in cnp) and len(cnp) > len(with_fraud):
                with_fraud = cnp[:6]
            if len(cnp) > len(longest):
                longest = cnp[:6]
        return with_fraud or longest

    def _account_takeover(self):
        """Behavioural break: a card's ordinary activity, then a sharp change.

        Two of the card's clean transactions first so the audience sees the
        baseline, then its fraudulent ones — the break is the point.
        """
        best = []
        for u, rows in self.by_user.items():
            clean = [i for i in rows if self.y[i] == 0]
            fraud = [i for i in rows if self.y[i] == 1]
            if len(clean) >= 2 and len(fraud) >= 2:
                cand = clean[:2] + fraud[:3]
                if len(cand) > len(best):
                    best = cand
        return best

    def _mule(self):
        """Fan-in / fan-out: many distinct cards converging on one merchant.

        This is the scenario that justifies the GNN. It is selected in tiers,
        and `describe()` reports which tier was reached, because what it proves
        depends entirely on the dataset:

          tier 1  frauds that the incumbent's rules do NOT fire on
                  -> the full claim: model catches, screening misses
          tier 2  frauds at a high fan-in merchant, rules may also fire
                  -> shows the network structure, but proves nothing about
                     rules evasion

        On the blueprint's 2019 split only tier 2 is reachable: every fraud
        there is in Rome, so any watchlist containing Rome catches all of them
        and zero of the 22,787 rule-evading rows are fraudulent. Tier 1 becomes
        reachable on a split where fraud geography varies. Never claim the
        tier-1 story while running tier 2.
        """
        by_merch = collections.defaultdict(list)
        for i in range(len(self.y)):
            by_merch[int(self.merch[i])].append(i)

        def pick(rows):
            """One row per distinct card, frauds first, so per-card velocity
            stays low and the fan-in is across accounts rather than within one."""
            seen, out = set(), []
            for i in sorted(rows, key=lambda i: -self.y[i]):
                u = int(self.user[i])
                if u in seen:
                    continue
                seen.add(u)
                out.append(i)
                if len(out) >= 8:
                    break
            return out

        # tier 1: fraudulent AND invisible to the incumbent's rules
        for mm in sorted(by_merch, key=lambda m: -sum(
                1 for i in by_merch[m] if self.y[i] == 1 and self._evades_rules(i))):
            cand = [i for i in by_merch[mm] if self._evades_rules(i)]
            if sum(1 for i in cand if self.y[i] == 1) < 2:
                break
            got = pick(cand)
            if len(got) >= 4:
                self._mule_tier = 1
                return got

        # tier 2: widest fan-in that still carries frauds
        for mm in sorted(by_merch, key=lambda m: -(
                sum(1 for i in by_merch[m] if self.y[i] == 1),
                len({int(self.user[i]) for i in by_merch[m]}))[0]):
            frauds = [i for i in by_merch[mm] if self.y[i] == 1]
            if len(frauds) < 2:
                break
            got = pick(by_merch[mm])
            if len(got) >= 4 and any(self.y[i] == 1 for i in got):
                self._mule_tier = 2
                return got
        return []

    def _city_anomaly(self):
        """Small amount, geographically wrong — reproduces the Rome $98.04
        case the v1 booth demo is known for."""
        cand = [i for i in range(len(self.y))
                if self.y[i] == 1 and self.amount[i] < 150
                and self.city[i] not in ("ONLINE", "", "nan")]
        return [int(self.rng.choice(cand))] if cand else []

    # --- description for the control API ----------------------------------
    def describe(self, name):
        rows = self.built.get(name, [])
        if not rows:
            return {"status": "unavailable", "rows": 0}
        evades = bool(all(self._evades_rules(i) for i in rows))
        d = {
            "status": "ready",
            "rows": len(rows),
            "frauds": int(sum(self.y[i] for i in rows)),
            "distinct_cards": len({int(self.user[i]) for i in rows}),
            "distinct_merchants": len({int(self.merch[i]) for i in rows}),
            "evades_incumbent_rules": evades,
            "amounts": [round(float(self.amount[i]), 2) for i in rows][:8],
            "cities": sorted({self.city[i] for i in rows})[:6],
        }
        if name == "mule-fanin-fanout":
            d["tier"] = self._mule_tier
            d["demonstrates"] = (
                "model catches it, incumbent screening does not"
                if self._mule_tier == 1 else
                "network structure across accounts; on this dataset the "
                "incumbent's rules also fire, so this does NOT demonstrate "
                "rules evasion - see docs/LIMITATIONS.md")
        return d
