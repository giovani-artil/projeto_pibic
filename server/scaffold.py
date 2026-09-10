import pickle
from collections import defaultdict

def scaffold(model_paths, output_path):

    agg_weights_x = defaultdict(float)
    agg_weights_y = defaultdict(float)

    agg_control_x = defaultdict(float)
    agg_control_y = defaultdict(float)

    agg_intercept_x = 0.0
    agg_intercept_y = 0.0

    num_models = 0

    for path in model_paths:
        with open(path, "rb") as f:
            state = pickle.load(f)

        weights_x = state.get("weights_x", {})
        weights_y = state.get("weights_y", {})

        control_x = state.get("control_x", {})
        control_y = state.get("control_y", {})

        for k, v in weights_x.items():
            agg_weights_x[k] += v

        for k, v in weights_y.items():
            agg_weights_y[k] += v

        for k, v in control_x.items():
            agg_control_x[k] += v

        for k, v in control_y.items():
            agg_control_y[k] += v

        agg_intercept_x += state.get("intercept_x", 0.0)
        agg_intercept_y += state.get("intercept_y", 0.0)

        num_models += 1

    for k in agg_weights_x:
        agg_weights_x[k] /= num_models

    for k in agg_weights_y:
        agg_weights_y[k] /= num_models

    for k in agg_control_x:
        agg_control_x[k] /= num_models

    for k in agg_control_y:
        agg_control_y[k] /= num_models

    agg_intercept_x /= num_models
    agg_intercept_y /= num_models

    global_state = {
        "weights_x": dict(agg_weights_x),
        "weights_y": dict(agg_weights_y),
        "intercept_x": agg_intercept_x,
        "intercept_y": agg_intercept_y,
        "control_x": dict(agg_control_x),
        "control_y": dict(agg_control_y)
    }

    with open(output_path, "wb") as f:
        pickle.dump(global_state, f)

    return global_state

def scaffold_from_states(states):
    agg_weights_x = defaultdict(float)
    agg_weights_y = defaultdict(float)
    agg_control_x = defaultdict(float)
    agg_control_y = defaultdict(float)
    agg_intercept_x = 0.0
    agg_intercept_y = 0.0

    for state in states:
        for k, v in state.get("weights_x", {}).items():
            agg_weights_x[k] += v
        for k, v in state.get("weights_y", {}).items():
            agg_weights_y[k] += v
        for k, v in state.get("control_x", {}).items():
            agg_control_x[k] += v
        for k, v in state.get("control_y", {}).items():
            agg_control_y[k] += v
        agg_intercept_x += state.get("intercept_x", 0.0)
        agg_intercept_y += state.get("intercept_y", 0.0)

    return _average(agg_weights_x, agg_weights_y, agg_control_x, agg_control_y, agg_intercept_x, agg_intercept_y, len(states))


def _average(agg_weights_x, agg_weights_y, agg_control_x, agg_control_y, agg_intercept_x, agg_intercept_y, n):
    return {
        "weights_x": {k: v / n for k, v in agg_weights_x.items()},
        "weights_y": {k: v / n for k, v in agg_weights_y.items()},
        "intercept_x": agg_intercept_x / n,
        "intercept_y": agg_intercept_y / n,
        "control_x": {k: v / n for k, v in agg_control_x.items()},
        "control_y": {k: v / n for k, v in agg_control_y.items()}
    }