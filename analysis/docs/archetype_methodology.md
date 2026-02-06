# Shot Archetype Methodology

> **Key Insight**: These archetypes describe **shot distribution patterns**, not offensive skill, versatility, or player quality.

## What the Model Clusters On

| Feature | Description |
|---------|-------------|
| `pct_rim` | % of shots from restricted area + paint |
| `pct_3pt` | % of shots from 3PT range (all zones) |
| `pct_midrange` | % of mid-range shots |
| `avg_sdi` | Average Shot Difficulty Index from distance, clock, shot type, zone, and angle |
| `distance_entropy` | How spread out shot locations are |
| `usage_pct` | Volume/frequency of attempts |

## Archetype Labels

| Archetype | Meaning | Example Players |
|-----------|---------|-----------------|
| **Paint-Dominant** | Cluster rim share above seasonal `q70` (`pct_rim`) | Gobert, Capela, Drummond |
| **Perimeter-Focused** | Cluster 3PT share above seasonal `q70` (`pct_3pt`) | Kennard, Merrill, Sam Hauser |
| **Mid-Range Specialist** | Cluster mid-range share above seasonal `q70` (`pct_midrange`) | DeRozan, Durant, Ingram |
| **Shot Creator (Mixed)** | Mixed profile with cluster SDI above seasonal median | Ant, SGA, Brunson |
| **Role Player (Mixed)** | Mixed profile with cluster SDI at/below seasonal median | Combo guards, role players |

### Usage Suffixes
- **(High Usage)** — Top 30% in attempts/game or usage%
- **(Low Usage)** — Bottom 30%
- *(No suffix)* — Middle 40%

## Why "Mixed" ≠ "Balanced Scorer"

The old "Balanced" label was misleading because it implied skill-based versatility.

**What "Mixed Shot Profile" actually means:**
- Shots come from multiple zones (high entropy)
- No single zone dominates (neither rim-heavy nor perimeter-heavy)
- Does **NOT** indicate scoring efficiency or skill

**The SDI-based split addresses this:**
- High SDI → **Shot Creator** (difficult, likely self-created shots)
- Low SDI → **Role Player** (easier, scheme-generated looks)
- Usage does not decide creator vs role-player; it only adds a suffix label.

## Cluster Validation

```
┌──────────────────────────────────────────────────────┐
│  GMM Selection: BIC-optimal k chosen per role group  │
│  Cluster Confidence: In-sample GMM posterior score    │
│  Role Assignment: Rim% + distance thresholds         │
└──────────────────────────────────────────────────────┘
```

## Limitations

> [!CAUTION]
> **What this model does NOT capture:**
> - Play type (ISO, PnR, spot-up, transition)
> - Defender context or closeout pressure
> - Passing/creation for others
> - Defensive attention level
> - Post skill, touch, footwork

## Reviewer Q&A

**Q: Why is [star player] in "Role Player (Mixed)"?**
A: The cluster is purely spatial. A star may shoot from diverse zones but take easier looks (low SDI) due to scheme or gravity.

**Q: Why are bigs and guards in separate clusters?**  
A: Role-aware clustering prevents bigs from being compared to guards on the same feature axes.

**Q: How do I validate cluster quality?**  
A: Check `gmm_bic_by_role.csv` for BIC curves and `cluster_confidence` column for assignment certainty.
