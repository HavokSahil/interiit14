# ARISTA NETWORKS TECHMEET 14.0
## RRM-PLUS: AI-ASSISTED, CLIENT-AWARE RADIO RESOURCE MANAGEMENT

---

## About the Company

Arista Networks is an industry leader in data-driven, client to cloud networking for large data center/AI, campus and routing environments. Arista's award-winning platforms deliver availability, agility, automation, analytics and security through an advanced network operating stack.

Arista was founded by industry luminaries Andy Bechtolsheim, Ken Duda and David Cheriton, launched in 2008 and is led by CEO Jayshree Ullal. Its seasoned leadership team is globally recognized as respected leaders and visionaries with a rich and extensive history in networking and innovation.

The company went public in June 2014, is listed with NYSE (ANET) with a market cap of 179B $ as of October 2025, and currently has more than 10,000+ cloud customers worldwide and has deployed 100M ports.

---

## Introduction and Motivation

Radio Resource Management (RRM) in enterprise Wi-Fi has historically relied on AP-side observations (e.g., periodic scanning, co-channel/adjacent-channel interference, airtime utilization) to decide channel, transmit power, channel width, CCA/OBSS-PD thresholds, and client steering hints. While effective, this AP-centric view can be:

- **Lagging**: Sparse scans miss fast interference dynamics (e.g., duty-cycled IoT, short-burst microwave/BLE activity).

- **Partial**: Hidden-node and multipath conditions at the client's location often diverge from what the AP sees.

- **Static**: Rule-based policies struggle with dense, high-variance environments and diverse client chipsets/OS behaviors.

Arista's additional (dedicated) radio provides continuous environmental visibility without sacrificing serving capacity. This opens the door for:

1. Faster, richer spectrum intelligence (including non-Wi-Fi classification)
2. Closed-loop multi-timescale control
3. An augmented client view that blends standards-based telemetry (802.11k/v/r/mc) with crowdsourced and synthetic measurements.

This Problem Statement defines a two-phase program, **Mid-Term** and **End-Term**, to design, prototype, and validate **RRM-Plus**, an AI-assisted, client-aware RRM system leveraging Arista's additional radio.

---

## Objectives

1. Raise real-world client QoE (not just AP KPIs) via client-aware decisions.

2. Exploit the additional radio for continuous, low-latency, low-overhead spectrum sensing.

3. Apply AI/ML to learn site-specific interference/topology patterns and optimize RRM parameters safely.

4. Blend AP and client views using standards and optional clients/agents to eliminate blind spots.

5. Deliver control with guardrails through reproducible changes, bounded churn, and rapid rollback.

---

## What Makes this Hard

1. Client diversity (chipsets/drivers/OS power policies) and opaque roaming decisions.

2. Short-duty, non-Wi-Fi interferers (BLE, Zigbee, cordless phones, analog video, microwave) with spectral signatures that are easy to miss with infrequent scans.

3. Multi-band, multi-width planning (20/40/80/160 MHz; 2.4/5/6 GHz) under DFS/AFC/regulatory constraints.

4. Evolving PHY/MAC features (11ax/11be OFDMA, MU-MIMO, SR with OBSS-PD) that add new control levers.

5. Stability vs agility: avoiding configuration flaps while reacting fast to real issues.

---

## Scope

1. **Bands**: 2.4/5 GHz initially; design extensible to 6 GHz/AFC where available.

2. **Levers**: channel, power (TPC), channel width, OBSS-PD/SR thresholds, target RSSI, roam/steer hints (802.11k/v), band-steering, load-balancing, admission control.

3. **Signals**: spectrum FFTs/IQ features, Wi-Fi counters, MCS/PER/retries, RU allocation/airtime, client RSSI/SNR, 802.11k neighbor/reports/Beacon/Link Measurement, 802.11mc RTT, TCP/QUIC RTT/Jitter, app QoS (optional).

4. **Guardrails**: change budgets, backoff timers, locality constraints, blast-radius isolation, SLO-driven policy.

---

## Problem Statement

### A. Mid Term

**Goal**: Prove value of the additional radio + client-aware signals for safer, better RRM; ship a pilot feature set.

#### 1. Additional-Radio Sensing & Scheduling

Design a sensing orchestrator that uses the additional radio to:

- **Adaptive dwell & scheduling**: Dynamic per-channel dwell times using multi-armed bandit to focus on likely-noisy channels; respect DFS pre-scan requirements.

- **Non-Wi-Fi classifier**: On-device lightweight CNN/feature-engine to classify BLE/Zigbee/microwave/FHSS; export confidence + duty cycle + center frequency + bandwidth.

- **Change detection**: online CUSUM/EWMA on airtime/CCA busy/Noise Floor to flag meaningful shifts in seconds, not minutes.

- **Zero-impact serving**: ensure sensing never drains client airtime beyond an SLA (e.g., <2%).

**Deliverables**: Sensing pipeline, classifier metrics (precision/recall per class), API for downstream RRM.

#### 2. Client-View Acquisition (Standards First)

- **802.11k**: Implement measurement requests (Beacon/Link/Neighbor) to obtain client-side RSSI/SNR and neighbor rankings.

- **802.11v BSS-TM**: Use client-view + AP load to craft ranked neighbor lists; log client acceptance rate & post-roam QoE deltas.

- **Passive client inference**: Estimate client perspective via uplink MCS vs downlink, ACK timing variance, retry asymmetry (hidden-node proxy).

- **Optional synthetic clients**: leverage a few USB/SoC probes per site (if available) for ground-truth.

**Deliverables**: Telemetry schema, success matrix by device OUI/OS, privacy posture, MDM guidance (no agent mandated).

#### 3. Policy & Guardrails (Rules + First ML)

- **Policy engine**: Expressible SLOs, e.g., maximize P50 throughput while keeping P95 retries < 8% and AP config churn ≤ 0.2/day.

- **Bayesian optimizer (BO)**: Offline sim + limited online BO to tune channel width, power, and OBSS-PD per AP-cell; respect regulatory/EIRP and DFS.

- **Churn control**: Change budgets, hysteresis, time-of-day windows, "cool-off" after client complaints.

**Deliverables**: Safe-change planner, A/B toggle, rollback, first BO results on pilot floor.

#### 4. KPIs & Acceptance Criteria (Mid-Term)

- **QoE lift**: +15-20% median downlink throughput for edge clients (RSSI -70 to -65 dBm).

- **Reliability**: P95 retry rate reduced by ≥20%; P95 uplink PER reduced by ≥15%.

- **Steering efficacy**: >85% BSS-TM acceptance where supported; P90 post-steer SINR +3 dB.

- **Stability**: config churn ≤ 0.2 changes/AP/day; DFS hits without client impact.

- **Overhead**: additional-radio sensing airtime cost <2%

#### 5. Mid-Term Submission (What to Submit)

- **Part 1**: Additional-radio sensing orchestrator & non-Wi-Fi classifier design + results (precision/recall, confusion matrix, CPU/RAM budget).

- **Part 2**: Safe-change planner, A/B toggle, rollback, first BO results on pilot floor.

- **Part 3**: Client-view acquisition via 802.11k/v, telemetry schema, acceptance metrics by device class.

- **Artifacts**: design doc, APIs, dashboards, offline replay sim with 24-hour log, and pilot site report.

---

### B. End Term

**Goal**: Deliver an AI-assisted, client-aware closed-loop RRM-Plus with multi-timescale control, production guardrails, and site-specific optimization.

#### 1. Multi-Timescale Control Loops

- **Fast loop (seconds-minutes)**: interference/reactive channel change on DFS/non-Wi-Fi spikes, transient width tightening, OBSS-PD adaptation based on real-time interference graph.

- **Slow loop (hours-days)**: global plan, graph coloring for channels, BO/RL to co-tune power/width/OBSS-PD per cell; align with occupancy patterns.

- **Event loop**: incident-aware, auto-mitigate microwave/BLE burst near cafeteria, exam hall quiet hours, etc.

#### 2. AI Methods & Data

- **Interference graph learning**: Graph Neural Network (GNN) fed by additional-radio edges (measured coupling) + client roam/throughput outcomes.

- **Safe RL**: Conservative Q-Learning or Reward-Constrained Policy Optimization with change budgets and do-no-harm baselines; evaluate vs BO.

- **Causal inference**: uplift modeling to validate that a change caused QoE improvement (counterfactual holdouts).

- **Explainability**: per-change reason codes ("50% OBSS detected; OBSS-PD raised from -82 to -74 dBm; predicted P95 retries -6%").

#### 3. Advanced Client-View

- **802.11mc RTT** (where supported) for location-aware interference hot-spots; privacy-preserving coarse bins.

- **Transport-layer QoE**: passive TCP/QUIC RTT and loss variance to detect client-side problems when MAC counters look clean.

- **Crowdsourced app probes (optional)**: tiny background pings during idle windows with rate limiting.

#### 4. Policy, SLOs & Compliance

- **SLO catalog** per SSID/role (exam hall, voice WLAN, guest) with distinct constraints (e.g., max width=20 MHz for voice).

- **Regulatory**: DFS/AFC, per-country EIRP, 6 GHz PSC channels; audit trail for every automated change.

- **Privacy**: no PII; hashed MACs; opt-out controls; data retention windows.

#### 5. KPIs & Acceptance Criteria (End-Term)

- **QoE**: +25-35% median throughput for edge clients; P95 latency -20%; P95 retries -30% network-wide.

- **Roaming**: >90% successful steers on capable clients; P50 roam time < 100 ms for FT-capable clients.

- **Airtime efficiency**: RU utilization +15% on 11ax cells with OFDMA scheduling improvements.

- **Stability**: mean time between config changes ≥ 4 hours; site change budget respected 99th percentile.

#### 6. End-Term Submission (What to Submit)

- All remaining sections: multi-timescale controller, AI methods (GNN + Safe RL), global planner, policy engine with guardrails and audit trail.

- **Simulation + A/B**: 7-day replay with injected interferers; live A/B at two sites with holdout floors.

- **Deliverables**: final report, dashboards, sim code, exportable policy templates.

---

## Reference Architecture

**On-AP (Edge)**: lightweight classifiers, change detection, and fast-loop actions with bounded CPU/RAM.

**Cloud/controller**: data lake, feature store, BO/RL training and validation.

**Site-broker**: graph planner and change rollout with blast-radius control.

**Telemetry bus**: protobuf/Parquet with time sync.

---

## Key APIs

- **/sensing**: additional-radio scans: spectral/non-Wi-Fi events.

- **/client_view**: 802.11k/v results, passive inferences, optional RTT bins.

- **/planner/propose**: candidate changes with predicted delta and confidence.

- **/planner/commit**: executes change with guardrails; records audit trail & rollback token.

- **/metrics**: KPIs by SSID/role/device-class; SLO compliance.

---

## Validation Plan

1. **Offline**: 24h/7d log replay; measure predicted vs actual deltas; backtest safety.

2. **Pilot**: two contrasting sites (dense office, education); define success per KPIs.

3. **Risk drills**: forced DFS, synthetic BLE floods, microwave bursts; verify stability and guardrails.

4. **Device diversity**: segment metrics by OUI/OS; identify cohorts needing special handling.

---

## Risks & Mitigations

1. **Client heterogeneity**: maintain OUI profiles and per-class policies.

2. **Privacy concerns**: no PII; hashed identifiers; explicit retention windows; admin controls.

3. **Over-automation**: human-in-the-loop mode, change budgets, one-click rollback.

4. **False positives in classification**: dual thresholds (confidence + impact), cooldown timers, ensemble models.

---

## Submission Format

1. **Mid-Term Submission**: Part 1, 2 and Part 3 policy/guardrails + initial KPIs.

2. **End-Term Submission**: All remaining sections; live demo or replay-based demo acceptable.

3. **Format**: ZIP with detailed report, dashboards/screenshots, and simulation/analysis notebooks.

---

## Evaluation Criteria

### Mid-Term (20%)
- Additional-radio sensing efficacy & classifier quality (8%)
- Client-view acquisition coverage & telemetry design (6%)
- Policy/guardrail design & initial KPI lift (6%)

### End-Term (70%)
- Multi-timescale controller & global planner quality (10%)
- AI methods (GNN/Safe-RL) and explainability (20%)
- QoE gains & stability vs control churn (20%)
- Operational readiness (audit, rollback, SLOs, privacy) (20%)

### Presentation (10%)

---

## Resources

- IEEE 802.11k/v/r/mc specs; 802.11ax/be features (OFDMA, SR, MU-MIMO, BSS Coloring, OBSS-PD)
- QUIC/TCP RTT and loss-based QoE estimation literature
- Online change detection (CUSUM/EWMA), Bayesian Optimization, Safe RL, GNNs for interference graphs

---

## Appendix A - KPI Definitions

- **Edge-Client Throughput**: median per-client downlink over 60s windows on RSSI ∈ [-70, -65] dBm.

- **Retry Rate**: MAC retries / MPDUs; reported as P95 per SSID.

- **Uplink PER**: per-client P95 across 1-min bins.

- **Config Churn**: mean automated changes per AP per day; excludes operator-initiated workflows.

- **Steering Acceptance**: fraction of BSS-TM responses with target match.

- **Airtime Cost of Sensing**: additional-radio scan airtime ÷ total AP airtime.

---

## Appendix B - Change Guardrails

- **Blast-Radius**: limit simultaneous changes to < N APs per RF domain.

- **Change Budgets**: ≤ 1 channel/power/width change per AP per 4h window unless incident.

- **Hysteresis**: min delta thresholds (e.g., power ≥ 2 dB, width ≥ 1 step) and cool-offs.

- **Time Windows**: avoid changes during known peak hours unless risk of SLO breach.

- **Rollback**: automatic reversion if KPIs worsen by X% across Y minutes.
