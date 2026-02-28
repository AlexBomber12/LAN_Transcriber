PR-FIX-OMEGACONF-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-omegaconf-01
PR title: PR-FIX-OMEGACONF-01 Add missing omegaconf dependency (runtime + CI) and a dependency smoke test
Base branch: main

Problem
Processing fails with:
  No module named 'omegaconf'
This indicates the runtime environment does not include OmegaConf, which is required by whisperx and related components.

Goals
1) Ensure omegaconf is installed in runtime Docker image(s).
2) Ensure CI catches this class of dependency regressions early.
3) Keep changes minimal and safe.

Non-goals
- Do not change pipeline logic.
- Do not add any network/model downloads.
- Do not modify docker compose topology.

Implementation steps

A) Add omegaconf to dependency lists
- Locate dependency files used by CI and Docker builds. In this repo they are typically:
  - requirements.txt
  - ci-requirements.txt
  - requirements.in (if present)
- Add:
  - omegaconf>=2.3.0
to requirements.txt and ci-requirements.txt.
- If requirements.in exists and you maintain pins, add a compatible pin, for example:
  - omegaconf==2.3.0
- If the repo uses a lock/compile process, regenerate locked requirements per project convention.

B) Add a fast dependency smoke test
- Create tests/test_dependency_smoke.py with:
  - test_omegaconf_importable:
      from omegaconf import OmegaConf
      assert hasattr(OmegaConf, "create")
- Keep it fully offline and fast.

C) Optional: strengthen docker smoke
- If tests/test_docker_smoke.py runs inside the container during CI, add a quick import probe:
  - python -c "from omegaconf import OmegaConf; print(OmegaConf.create({'ok': 1}))"
- Do not download any ML models.

D) Verify
- Run:
  - scripts/ci.sh
- Build the docker image used in production and confirm import works inside the container:
  - python -c "from omegaconf import OmegaConf; print(OmegaConf.create({'ok': 1}))"

Success criteria
- Runtime no longer fails with missing omegaconf.
- CI fails if omegaconf is removed in the future.
- scripts/ci.sh is green.
```
