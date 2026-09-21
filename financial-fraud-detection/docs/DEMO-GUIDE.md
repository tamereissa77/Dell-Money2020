# Demo guide

For whoever is presenting. Everything here has been run on the machine; no step
is aspirational.

---

## Before anyone arrives

```bash
cd ~/APPS/Money2020
make fraud-v2                       # ~48 s, exit 0
cd financial-fraud-detection
make prewarm                        # 15 s - NOT optional
make which                          # confirm: v2, FEED_SOURCE=kafka
```

Open **`http://<host>:8090`** and check the status bar along the bottom reads
`HEALTHY`. One URL; every view is a tab.

On a multi-day event, **restart the stack each morning**. Triton's latency
drifts over long runs — `docs/OPERATIONS.md` has the checklist.

Set the pace before the first visitor:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"tps":400}' \
  http://localhost:8090/api/presenter/rate
```

400 TPS reads well. 5,000 is a blur — use it only to make a throughput point.

---

## The three-minute demo

**Tab: LIVE DETECTION.**

> "This is card traffic being scored on a single NVIDIA GB10, on this desk.
> 400 a second here, and it sustains just under 7,000. Nothing leaves the box —
> the network cable can come out and nothing changes."

Point at the status bar: end-to-end latency, model latency, pipeline healthy.

**Press Ctrl+Shift+P, inject `merchant-city-anomaly`, close the panel.**

> "A transaction just arrived that the model flagged. Small amount, wrong
> geography."

**Tab: CASES.** Pick the top alert.

> "The existing screening engine raised this. Our model scored it 0.99 and put
> it at the top of a queue of eighty thousand. Here is why it scored it — this
> attribution is for *this* transaction, not an average over a batch."

**Press "Draft narrative".**

> "That is a first draft for the investigator, in English and Arabic, and every
> sentence cites the record it rests on. It is marked machine-drafted and forms
> no part of the case until a named analyst adopts it."

**Close on the approval gate.**

> "And no case closes without two named people — the investigator who submits
> and a different person who approves. That is enforced in the service, not in
> the screen, so it holds if you call the API directly."

---

## The eight-minute technical demo

For a data science, risk or architecture audience. The three-minute version is
the opening; this adds four movements.

### 1. The queue argument — **Tab: SCREENING COMPARISON**

The commercially important screen.

> "Left is the incumbent's queue in its own order. Right is the same alerts,
> reordered by the model. Nothing is removed — the rule and reason stay
> attached, and the original position is recoverable."

Switch the depth selector to **500**.

> "Working 500 alerts instead of eighty thousand, the re-ranked queue surfaces
> around ninety percent of the real frauds. The incumbent's own top 500 gets
> under ten. Same investigator effort."

Point at the amber strip across the top.

> "Those numbers mean nothing without this. That is the fraud rate of the
> stream, the true base rate, and the incumbent's false-positive rate — 98 to
> 99 percent, which is where real engines live."

**Switch to the true base rate** (presenter panel → *true rate*). The screen
goes quiet.

> "That is what 0.122 percent looks like. We show it deliberately — in demo
> mode we run the fraud rate 33 times higher so something happens while you are
> standing here, and the screen says so at all times."

### 2. Why a graph — **Tab: ENTITY GRAPH**

Inject `mule-fanin-fanout` first (Ctrl+Shift+P).

> "Nine unrelated cards, one merchant. Each transaction on its own is
> unremarkable — small, ordinary city, ordinary entry mode. The structure only
> exists *across* accounts, which is exactly what a row-at-a-time rules engine
> has nowhere to see."

**Be precise here** — see *Straight answers*, question 4. On this dataset the
incumbent's rules also fire on these, so this shows the structure, not rules
evasion.

### 3. Evidence — **Tab: CASES**, then **AUDIT TRAIL**

Walk a case through assign → investigate → submit, then try to close it as the
same person.

> "Four eyes. The submitter cannot approve their own case, and this is the
> service refusing, not the button being greyed out."

Close it as a second name, then press **Evidence pack**.

> "That is the whole history of one case — alert, score, model and data
> versions, every state change, every named human action — with the chain
> verified and the pack sealed with its own digest."

**Tab: AUDIT TRAIL.**

> "Append-only by construction. Records arrive only from the event stream;
> there is no write endpoint on this service at all. Each record carries the
> hash of the one before it, so altering any record breaks every hash after it —
> and verification tells you *which* record, not just that something is wrong."

### 4. Resilience — optional, and it lands well

```bash
docker compose stop redpanda
```

> "That is the event bus gone, mid-demo."

The banner appears, no view white-screens.

```bash
docker compose start redpanda
```

> "Five seconds, unattended. Nothing was lost."

---

## The presenter panel

**Ctrl+Shift+P.** No visible button — a customer must never see it.

| Control | Use |
|---|---|
| 5 / 60 / 400 / 5000 TPS | 5 to narrate a single transaction; 5000 for throughput |
| demo rate / true rate | the honesty switch. Show both |
| pause / resume | freeze the feed to talk over it (Space also pauses) |
| scenario buttons | inject a typology on demand |

### The five typologies

| Scenario | What it shows |
|---|---|
| `merchant-city-anomaly` | the classic single flagged transaction |
| `geo-velocity` | impossible travel, two card-present transactions |
| `cnp-burst` | card-testing: same card, online, escalating amounts |
| `account-takeover` | two ordinary transactions, then the behavioural break |
| `mule-fanin-fanout` | nine cards, one merchant — the graph argument |

All five appear in the UI in about **0.3 seconds**. They are curated sequences
of **real rows** from the test set, never fabricated transactions.

**After any `make stream` / `make classic` switch, inject once and discard the
result.** The UI's consumer joins at the latest offset, and anything produced
during that join window is never seen.

---

## Straight answers to the questions you will get

### 1. "What accuracy does it get?"

Do not quote 0.9578. On the benchmark's standard test split **every fraud is in
one city**, so `if city == "Rome"` scores F1 0.99 and beats the model. That is a
property of IBM's synthetic data — its fraud generator clusters by era.

The honest figure is **F1 0.78 at precision 0.82**, measured on a period where
fraud spans 730 cities, against 0.68 for the best single rule.

Open the showcase page — *Reading the benchmark* — and walk them through it.
Being the people who found it is a much stronger position than being the people
who shipped it.

### 2. "Can the model see the label?"

No, and it is checkable three ways: labels are on a different topic the scoring
service does not subscribe to; the label CSV is not mounted into that container
at all; and the service asserts this at startup and refuses to run if it ever
appears.

### 3. "Does this replace our screening engine?"

No. It never touches it. The incumbent keeps screening, every alert it raises
is preserved, and this reorders its queue. The platform performs no sanctions or
AML screening — say that plainly.

### 4. "Does it catch things our rules miss?"

**On this dataset, do not claim that.** Every fraud in the test split trips the
geography rule, so there is nothing the model catches that the incumbent
misses. `mule-fanin-fanout` shows the *structure* a graph sees; the tooling
reports which of the two it is demonstrating and will not let you claim the
stronger version by accident.

The honest framing: *"on your data, with your networks, this is where we would
expect the graph to earn its keep — and that is a short proof of concept, not a
slide."*

### 5. "What would it do on our data?"

Better in some ways, harder in others. Richer features and a real graph — shared
devices, addresses, beneficiaries — versus late and partial labels, harsher
imbalance and continuous drift. Expect absolute F1 **below** 0.78, not above.

But F1 is the wrong yardstick. Banks buy on *detection at fixed review
capacity*, which is what the comparison screen measures, and which a bank can
validate against its own historical alerts in about a week without deploying
anything.

### 6. "Is the explanation real, or a batch average?"

Per-decision. A single-transaction subgraph with Shapley computed on demand,
about 3.3 seconds. Two different transactions give different attributions —
show it.

### 7. "What happens when the model is wrong?"

Nothing automatic. The model prioritises and explains; a named analyst decides.
The draft narrative is marked machine-drafted until adopted, and the system
records the machine text, the human edits and the approving identity
separately.

Worth volunteering: during testing the language model read a 0.9955 fraud score
as *"strong likelihood of being legitimate"*. A fluent, well-cited narrative can
still be exactly wrong — which is precisely why nothing reaches a case file
without a person adopting it.

### 8. "Can it run air-gapped?"

Yes, and it is the only way it runs. No CDN, no external fonts, no telemetry.
The offline bundle is 49 GB and carries images, model weights and the dataset.
Provisioning takes about 11 minutes; starting takes 13 seconds.

---

## When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| Feed frozen, latency climbing | Triton drift on a long run | `docker compose restart triton && make prewarm` |
| UI looks like v1, tabs missing | `FEED_SOURCE` reverted to `file` | `make stream` |
| Red degraded banner | broker or a service down | usually self-heals in ~5 s; else `docker compose restart redpanda` |
| Injection does nothing | first one after a restart | inject again |
| Cases tab empty | alerts not joined yet | wait ~30 s after start |
| Copilot slow or template-only | LLM still loading | 43 s from cold; template output is fine to show |

**Nothing above needs a rebuild.** If you are reaching for `make redeploy`,
stop — it wipes `generated.env`, volumes and the data directory.

---

## Things not to say

- **"Apache Kafka."** It is Redpanda. Say *Kafka-compatible event stream*.
- **"Suspicious transaction report", "SAR", "STR", "filing."** This produces an
  investigator's draft narrative. The software refuses to emit those words.
- **"99.6% accurate."** See question 1.
- **"It replaces your screening engine."** It augments it.
- **Any vendor name for the cloud endpoint** in the other demos — say *cloud API
  endpoint*.
