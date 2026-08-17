# TACTUS convenience targets.
#
#   make help                 # this list
#   make download-meta        # ~50 MB: events, phenotype, VTD.csv  (start here)
#   make audit                # the three Phase-0 audits
#   make download-raw         # ~110 GB in the background; a hard dependency
#   make preprocess           # BDF -> ICA -> autoreject -> epochs + trials.parquet
#   make embed                # frozen video-tower embeddings (360 + 90)
#   make baseline-mvpa        # reproduce the companion paper's decoding curves
#   make train-infonce        # the reference model over every fold
#   make eval-all             # aggregate every finished run
#
# Everything is resumable: re-running a target picks up where it stopped.
# Override any variable on the command line, e.g.
#   make train-infonce REGIME=within_subject GPUS=4 TAG=pilot

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY          ?= python
DATA_ROOT   ?= data
BIDS        ?= $(DATA_ROOT)/ds005662
S3          ?= s3://openneuro.org/ds005662
SYNC        ?= aws s3 sync --no-sign-request
WINDOW      ?= w0600
SUBJECTS    ?= all
MODEL_TAG   ?= siglip2-base
# registry tag understood by tactus.models.video.encode --model
VIDEO_MODEL ?= siglip2-base
CONF        ?= configs/nice_infonce.yaml
REGIME      ?= double_disjoint
GPUS        ?= 1
NJOBS       ?= 8
TAG         ?=
EXTRA       ?=

# Entry points owned by sibling modules.  If a module exposes a different
# __main__, change it here once rather than in every target below.
DOWNLOAD_CMD  ?= $(PY) -m tactus.data.download
EVENTS_CMD    ?= $(PY) -m tactus.data.events
PREPROCESS_CMD?= $(PY) -m tactus.data.preprocess
EMBED_CMD     ?= $(PY) -m tactus.models.video.encode
AUDIT_CMD     ?= $(PY) ../server_handoff/audit_phase0.py
# the census now consumes the .npz caches written by EMBED_CMD rather than
# re-encoding; ../server_handoff/encoder_census.py is the stand-alone variant.
CENSUS_CMD    ?= $(PY) -m tactus.eval.census

TAGFLAG := $(if $(TAG),--tag $(TAG),)

.PHONY: help env download-meta download-stim download-derivatives download-raw \
        audit census preprocess trials embed \
        preprocess-ica preprocess-autoreject \
        baseline-mvpa baseline-mvpa-video baseline-align baseline-corrca baseline-srm \
        baselines train-infonce train-protonce train-atm train scaling-curve \
        eval-all status clean-runs clean-cache test lint

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

env: ## create the conda environment
	conda env create -f environment.yml

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

download-meta: ## metadata only (~50 MB): events.tsv, phenotype, VTD.csv
	mkdir -p $(BIDS)
	$(SYNC) $(S3) $(BIDS) \
	  --exclude "*" \
	  --include "*.tsv" --include "*.json" --include "CHANGES" --include "README*" \
	  --include "code/analysis/*" --include "phenotype/*" --include "sourcedata/*.csv"
	@echo "expect 80 x sub-XX/eeg/*_events.tsv and code/analysis/VTD.csv"

download-stim: ## the 384 stimulus mp4s (~1 GB)
	mkdir -p $(DATA_ROOT)/stimuli
	$(SYNC) $(S3)/code/experiment_files/stimuli $(DATA_ROOT)/stimuli

download-derivatives: ## authors' MNE epochs (~11 GB) -- Phase 0 triage only
	$(SYNC) $(S3)/derivatives $(BIDS)/derivatives

download-raw: ## raw BDF (~110 GB), backgrounded; a HARD dependency for ICA
	mkdir -p $(BIDS)
	nohup $(SYNC) $(S3) $(BIDS) > $(DATA_ROOT)/sync_raw.log 2>&1 &
	@echo "syncing in the background; tail -f $(DATA_ROOT)/sync_raw.log"

# --------------------------------------------------------------------------- #
# phase 0 audits
# --------------------------------------------------------------------------- #

audit: ## sequence x orientation crosstab, attribute correlations, phenotype QC
	$(AUDIT_CMD) --bids $(BIDS) --out results/phase0

census: ## video-encoder census: collapse, RDM-behaviour, attribute predictability
	$(CENSUS_CMD) --emb-dir $(DATA_ROOT)/derived/video_emb \
	  --vtd $(BIDS)/code/analysis/VTD.csv --out results/phase0

# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #

trials: ## build data/derived/trials.parquet from the 80 events.tsv
	$(EVENTS_CMD) --bids-root $(BIDS) $(EXTRA)

preprocess: trials ## raw BDF -> notch -> epochs (resumable); ICA/autoreject are separate stages
	$(PREPROCESS_CMD) --bids-root $(BIDS) --windows $(WINDOW) --stage epochs \
	  --subjects $(SUBJECTS) --n-jobs $(NJOBS) $(EXTRA)

preprocess-ica: ## 1 Hz high-passed copy -> picard ICA -> ocular components
	$(PREPROCESS_CMD) --bids-root $(BIDS) --stage ica \
	  --subjects $(SUBJECTS) --n-jobs $(NJOBS) $(EXTRA)

preprocess-autoreject: ## fit autoreject on the written memmaps; annotates, never drops
	$(PREPROCESS_CMD) --stage autoreject --subjects $(SUBJECTS) \
	  --n-jobs $(NJOBS) $(EXTRA)

embed: ## frozen video embeddings -> data/derived/video_emb/$(MODEL_TAG).npz
	$(EMBED_CMD) --stim-root $(BIDS) --model $(MODEL_TAG) $(EXTRA)

# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

baseline-mvpa: ## time-resolved LDA/ridge, leave-one-sequence-out (companion protocol)
	$(PY) -m tactus.baselines.linear_mvpa --window $(WINDOW) --cv sequence \
	  --subjects $(SUBJECTS) --n-jobs $(NJOBS) --report $(EXTRA)

baseline-mvpa-video: ## the same decoders under stimulus-disjoint CV
	$(PY) -m tactus.baselines.linear_mvpa --window $(WINDOW) --cv video --n-folds 9 \
	  --subjects $(SUBJECTS) --n-jobs $(NJOBS) --report $(EXTRA)

baseline-align: ## EA + ridge -> video embedding, cosine retrieval
	$(PY) -m tactus.baselines.linear_align --regime $(REGIME) --window $(WINDOW) \
	  --model-tag $(MODEL_TAG) $(EXTRA)

baseline-corrca: ## CorrCA / ISC spatial filters + per-subject ISC
	$(PY) -m tactus.baselines.corrca --window $(WINDOW) --subjects $(SUBJECTS) \
	  --split-half --per-condition $(EXTRA)

baseline-srm: ## SRM + cheap new-subject enrollment curve
	$(PY) -m tactus.baselines.srm --regime $(REGIME) --window $(WINDOW) \
	  --to-video --model-tag $(MODEL_TAG) --enroll-fracs 0.1,0.25,0.5,1.0 $(EXTRA)

baselines: baseline-mvpa baseline-align baseline-corrca baseline-srm ## all four

# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

train: ## train $(CONF) over $(REGIME); set GPUS>1 to shard folds across devices
	@if [ "$(GPUS)" -le 1 ]; then \
	  $(PY) -m tactus.train.run --config $(CONF) --regime $(REGIME) $(TAGFLAG) $(EXTRA); \
	else \
	  for i in $$(seq 0 $$(($(GPUS)-1))); do \
	    echo "launching shard $$i/$(GPUS) on cuda:$$i"; \
	    CUDA_VISIBLE_DEVICES=$$i $(PY) -m tactus.train.run --config $(CONF) \
	      --regime $(REGIME) --shard $$i --num-shards $(GPUS) $(TAGFLAG) $(EXTRA) & \
	  done; \
	  wait; \
	  $(PY) -m tactus.train.run --config $(CONF) --regime $(REGIME) $(TAGFLAG) --aggregate-only; \
	fi

train-infonce: ## reference model: NICE encoder + masked InfoNCE
	$(MAKE) train CONF=configs/nice_infonce.yaml

train-protonce: ## prototype-level contrast (loss-family sanity check)
	$(MAKE) train CONF=configs/nice_protonce.yaml

train-atm: ## ATM encoder + composite objective (lambda ablation grid)
	$(MAKE) train CONF=configs/atm_composite.yaml

scaling-curve: ## 10/20/40/79 training subjects, fixed evaluation set
	@for n in 10 20 40 79; do \
	  echo "=== $$n training subjects ==="; \
	  $(PY) -m tactus.train.run --config $(CONF) --regime $(REGIME) \
	    --override train.n_train_subjects=$$n --tag n$$n $(EXTRA); \
	done

# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

eval-all: ## re-aggregate every run directory under results/runs
	@for d in results/runs/*/; do \
	  echo "--- $$d"; \
	  $(PY) -c "from tactus.train.run import aggregate; aggregate('$$d')"; \
	done

status: ## how far along is the sweep
	@echo "epochs:      $$(ls -1 $(DATA_ROOT)/derived/epochs/*_$(WINDOW).npy 2>/dev/null | wc -l) / 80"
	@echo "video_emb:   $$(ls -1 $(DATA_ROOT)/derived/video_emb/*.npz 2>/dev/null | wc -l)"
	@echo "folds done:  $$(find results/runs -name done.json 2>/dev/null | wc -l)"
	@echo "folds failed:$$(find results/runs -name failed.json 2>/dev/null | wc -l)"

# --------------------------------------------------------------------------- #
# housekeeping
# --------------------------------------------------------------------------- #

test: ## unit tests
	pytest -q tests

lint:
	ruff check tactus && black --check tactus

clean-runs: ## delete every training run (NOT the derived data)
	rm -rf results/runs

clean-cache: ## delete condition-average and transform caches
	rm -rf results/cache
