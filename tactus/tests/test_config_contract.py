"""Static contract test: every key in configs/*.yaml is a real constructor argument.

Pure AST -- needs neither torch nor omegaconf, so it runs anywhere and catches a
bad config in milliseconds instead of at hour zero of a 40-fold sweep.
"""
import ast, sys, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
fails = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra != "" else ""))
    if not cond:
        fails.append(name)


def registry(pkg_dir: Path, decorator: str):
    """{registered_name: (class_node, module_path)} for @<decorator>("name")."""
    out = {}
    for py in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == decorator
                        and dec.args and isinstance(dec.args[0], ast.Constant)):
                    for a in dec.args:
                        if isinstance(a, ast.Constant):
                            out[a.value] = (node, py)
    return out


def init_params(cls_node):
    for n in cls_node.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__init__":
            a = n.args
            names = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs} - {"self"}
            return names, bool(a.kwarg)
    return set(), False


losses = registry(REPO / "tactus" / "losses", "register_loss")
check("loss registry discovered", len(losses) >= 8, sorted(losses))

encoders = registry(REPO / "tactus" / "models" / "eeg", "register_eeg_encoder")
check("encoder registry discovered", len(encoders) >= 2, sorted(encoders))

# aliases the encoder builder normalizes before it hits the constructor
ENC_ALIASES = {"name", "arch", "encoder", "encoder_name", "embed_dim", "d_embed", "d_out",
               "out_dim", "dim", "n_times", "n_samples", "T", "n_channels", "n_chans", "C",
               "subject_cond", "subject_conditioning", "conditioning", "subject_mechanism",
               "subject_cond_kwargs", "subject_conditioning_kwargs", "params"}
COMPOSITE_META = {"weight", "warmup_steps", "start_step", "type", "name"}


def check_loss_block(cfg_name, block, prefix="loss"):
    name = block.get("name")
    if name == "composite":
        top, kw = init_params(losses["composite"][0])
        for k in block:
            if k in ("name", "components"):
                continue
            check(f"{cfg_name}: composite kwarg '{k}' exists", k in top or kw, sorted(top))
        for comp, spec in (block.get("components") or {}).items():
            if isinstance(spec, (int, float)):
                sub_name, keys = comp, set()
            else:
                sub_name = spec.get("name") or spec.get("type") or comp
                keys = set(spec) - COMPOSITE_META
            check(f"{cfg_name}: component '{comp}' -> registered loss '{sub_name}'",
                  sub_name in losses, sorted(losses))
            if sub_name not in losses:
                continue
            params, has_kw = init_params(losses[sub_name][0])
            bad = sorted(k for k in keys if k not in params and not has_kw)
            check(f"{cfg_name}: component '{comp}' kwargs are all real", not bad,
                  f"unknown={bad} valid={sorted(params)}" if bad else "")
        return
    check(f"{cfg_name}: loss '{name}' is registered", name in losses, sorted(losses))
    if name not in losses:
        return
    params, has_kw = init_params(losses[name][0])
    bad = sorted(k for k in block if k != "name" and k not in params and not has_kw)
    check(f"{cfg_name}: loss kwargs are all real", not bad,
          f"unknown={bad} valid={sorted(params)}" if bad else "")


def merged(path):
    """Resolve the base: chain the same way run.load_config does."""
    def deep(a, b):
        o = dict(a)
        for k, v in b.items():
            o[k] = deep(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
        return o
    def layer(a, b):  # mirrors run._merge_layer's loss-ownership rule
        old = (a.get("loss") or {}).get("name")
        new = (b.get("loss") or {}).get("name")
        out = deep(a, b)
        if new is not None and old is not None and str(new) != str(old):
            out["loss"] = dict(b["loss"])
        return out

    chain, cur = [], Path(path)
    while cur is not None:
        node = yaml.safe_load(cur.read_text(encoding="utf-8"))
        chain.append(node)
        b = node.get("base")
        cur = (cur.parent / b) if b else None
    out = {}
    for node in reversed(chain):
        out = layer(out, node)
    out.pop("base", None)
    return out


for cfg_file in sorted((REPO / "configs").glob("*.yaml")):
    cfg = merged(cfg_file)
    tag = cfg_file.name
    check_loss_block(tag, cfg["loss"])

    enc = cfg["model"]["eeg"]
    enc_name = enc.get("name")
    check(f"{tag}: encoder '{enc_name}' is registered", enc_name in encoders, sorted(encoders))
    if enc_name in encoders:
        params, has_kw = init_params(encoders[enc_name][0])
        keys = set(enc) | set(enc.get("params") or {}) | set(cfg["model"]) - {"eeg", "projector"}
        # build_eeg_encoder drops unknown keys with a warning instead of raising,
        # so this is advisory -- but a silently dropped key is still a silent bug
        unknown = sorted(k for k in (set(enc.get("params") or {}))
                         if k not in params and k not in ENC_ALIASES and not has_kw)
        check(f"{tag}: encoder params are accepted (else silently dropped)", not unknown,
              f"dropped={unknown} valid={sorted(params)}" if unknown else "")

# the projector block must reach in_dim/out_dim after aliasing
PROJ_ALIASES = {"in_dim": {"d_in", "d_video", "d_vid", "video_dim", "vid_dim", "input_dim"},
                "out_dim": {"d_out", "d_embed", "embed_dim", "dim"},
                "hidden_dim": {"d_hidden", "hidden"}}
proj_src = ast.parse((REPO / "tactus" / "models" / "heads.py").read_text(encoding="utf-8"))
vp = next(n for n in ast.walk(proj_src) if isinstance(n, ast.ClassDef) and n.name == "VideoProjector")
vp_params, vp_kw = init_params(vp)
for cfg_file in sorted((REPO / "configs").glob("*.yaml")):
    cfg = merged(cfg_file)
    pp = (cfg["model"].get("projector") or {}).get("params") or {}
    canon = set()
    for k in pp:
        canon.add(next((c for c, al in PROJ_ALIASES.items() if k in al), k))
    bad = sorted(k for k in canon if k not in vp_params and not vp_kw)
    check(f"{cfg_file.name}: projector params resolve to real args", not bad,
          f"unknown={bad} valid={sorted(vp_params)}" if bad else "")

print()
print("=" * 60)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else " -- all green"))


def test_no_failures():
    """pytest entry point; the checks above run at import."""
    assert not fails, f"{len(fails)} check(s) failed: {fails}"


if __name__ == "__main__":
    sys.exit(1 if fails else 0)
