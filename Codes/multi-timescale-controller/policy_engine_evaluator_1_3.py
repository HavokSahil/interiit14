import json
import uuid
import hmac
import hashlib


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def sign_object(obj, signing_key: str):
    msg = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hmac.new(signing_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------

def normalize_metrics(ap_dict):
    metrics = {}

    metrics["RSSI_dBm"] = {
        "P50": float(ap_dict["rssi_dbm_p50"]),
        "P95": float(ap_dict["rssi_dbm_p95"]),
    }

    metrics["SNR_dB"] = {
        "P50": float(ap_dict["sinr_db_p50"]),
        "P95": float(ap_dict["sinr_db_p95"]),
    }

    metrics["PER_pct"] = {"P50": None, "P95": None}

    metrics["Retry_pct"] = {
        "P50": float(ap_dict["retry_rate_p50"]),
        "P95": float(ap_dict["retry_rate_p95"]),
    }

    metrics["Airtime_util_pct"] = {
        "P50": float(ap_dict["airtime_fraction_p50"]) * 100,
        "P95": float(ap_dict["airtime_fraction_p95"]) * 100,
    }

    cca = float(ap_dict["cca_busy_percentage"])
    metrics["CCA_busy_pct"] = {"P50": cca, "P95": cca}

    metrics["station_count"] = {"P50": None}

    return metrics


# -------------------------------------------------------------------
# QoE computation
# -------------------------------------------------------------------

def compute_qoe_components(metrics, normalizers):
    RSSI_min = normalizers["RSSI_min"]
    RSSI_max = normalizers["RSSI_max"]
    PER_max  = normalizers["PER_max_pct"]
    RET_max  = normalizers["Retry_max_pct"]

    S = clamp((metrics["RSSI_dBm"]["P95"] - RSSI_min) / (RSSI_max - RSSI_min))

    per_p95 = metrics["PER_pct"]["P95"]
    T = clamp(1 - per_p95 / PER_max) if per_p95 is not None else 0.0

    R = clamp(1 - metrics["Retry_pct"]["P95"] / RET_max)

    L = clamp(1 - metrics["Airtime_util_pct"]["P50"] / 100)
    A = clamp(1 - metrics["CCA_busy_pct"]["P50"] / 100)

    return {"S": S, "T": T, "R": R, "L": L, "A": A}


# -------------------------------------------------------------------
# Penalty engine
# -------------------------------------------------------------------

def compute_penalty(observed, slo, operator, clip_max):
    if observed is None:
        return 0.0, 0.0

    if operator == "<=":
        margin = observed - slo
        penalty = max(0, margin / max(1, slo))

    elif operator == ">=":
        margin = slo - observed
        penalty = max(0, margin / max(1, abs(slo)))

    else:
        return 0.0, 0.0

    return clamp(penalty, 0.0, clip_max), margin


def pick_actions(metric_name, penalty, role, explicit):
    if penalty <= 0:
        return []

    actions = []

    if explicit:
        actions += explicit.split("/")

    if "Retry" in metric_name or "PER" in metric_name:
        actions += ["Penalty", "SteerClients"]

    if "CCA" in metric_name:
        if "ExamHall" in role:
            actions.append("EnforceAirtimeFairness")
        actions += ["SteerClients", "ChannelChange"]

    return sorted(set(a.strip() for a in actions if a.strip()))


# -------------------------------------------------------------------
# CORE ENGINE FUNCTION
# -------------------------------------------------------------------

def evaluate_policy_for_ap(ap_dict, role, slo_catalog, signing_key):
    role_slo = slo_catalog["roles"][role]
    normalizers = slo_catalog["global"]["normalizers"]
    defaults = slo_catalog["global"]["defaults"]
    enforcement = role_slo["enforcement"]

    metrics = normalize_metrics(ap_dict)
    qoe_components = compute_qoe_components(metrics, normalizers)

    w = role_slo["qos_weights"]
    QoE = clamp(
        qoe_components["S"] * w["ws"] +
        qoe_components["T"] * w["wt"] +
        qoe_components["R"] * w["wr"] +
        qoe_components["L"] * w["wl"] +
        qoe_components["A"] * w["wa"]
    )

    penalties, margins, actions = {}, {}, []

    for metric_key, rule in enforcement.items():
        obs = metrics[metric_key][rule.get("authority", "P50")]

        penalty, margin = compute_penalty(
            observed=obs,
            slo=rule["value"],
            operator=rule["operator"],
            clip_max=defaults["penalty_clip_max"]
        )

        if penalty > 0:
            penalties[metric_key] = penalty
            margins[f"{metric_key}_margin"] = margin
            actions += pick_actions(metric_key, penalty, role, rule.get("action", ""))

    actions = sorted(set(actions))
    reward = QoE - defaults["penalty_scale_for_BO_reward"] * sum(penalties.values())

    audit = {
        "audit_id": str(uuid.uuid4()),
        "ap_id": ap_dict.get("ap_id"),
        "role": role,
        "penalties": penalties,
        "constraint_margins": margins,
        "actions": actions,
        "rollback_token": str(uuid.uuid4()),
    }
    audit["signature"] = sign_object(audit, signing_key)

    return {
        "qoe_components": qoe_components,
        "QoE": QoE,
        "penalties": penalties,
        "constraint_margins": margins,
        "actions_recommended": actions,
        "reward_for_BO": reward,
        "audit": audit
    }


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def run_policy_engine(ap_data: dict, role: str, slo_catalog: dict, signing_key: str):
    return evaluate_policy_for_ap(
        slo_catalog=slo_catalog,
        ap_dict=ap_data,
        role=role,
        signing_key=signing_key #bxc doesnt like
    )
