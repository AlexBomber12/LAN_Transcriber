PR-FIX-CTRANSLATE2-EXECSTACK-02

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-ctranslate2-execstack-02
PR title: PR-FIX-CTRANSLATE2-EXECSTACK-02 Ensure GHCR runtime images are patched (no execstack) and run Docker smoke on the exact built image (no tag races)
Base branch: main

Problem
Even after PR-FIX-CTRANSLATE2-EXECSTACK-01, importing ctranslate2 still fails in the published GHCR image:

  docker pull ghcr.io/alexbomber12/lan-transcriber:main
  docker run --rm ghcr.io/alexbomber12/lan-transcriber:main python -c "import ctranslate2; print('ok')"

Error:
  ImportError: libctranslate2-<hash>.so.<ver>: cannot enable executable stack as shared object requires: Invalid argument

This also breaks GH Actions Docker smoke (runner kernel refuses execstack).

Root causes to address
1) The execstack patch is not being applied to the final runtime image layers pushed to GHCR
   (common reasons: patch runs before the .so exists, runs in the wrong stage, or only on one target).
2) CI can test or pull an old :main image if image build/push and smoke are not chained (tag race).
   We must run smoke against the exact image digest that was just built.

Goals
Fix 1 (runtime image correctness)
- Guarantee execstack is cleared for any libctranslate2*.so* inside the final runtime image (all published targets).
- Fail docker build if the patch was not applied (deterministic).
- Keep the fix lightweight and fully automated at build time.

Fix 2 (CI correctness)
- Ensure Docker smoke runs against the exact image produced by the build (digest), never an older :main tag.
- Remove or avoid duplicate docker-smoke execution that pulls :main in parallel.

Non-goals
- Do not change application logic.
- Do not disable smoke checks that import ctranslate2.

Implementation details

A) Dockerfile: patch in the final runtime stages after deps are installed
1) Locate the Dockerfile stages that produce the published images.
   Typical names: runtime-full and runtime-lite.
   If names differ, apply equivalent edits.

2) Ensure patch tools exist in the stage where the patch runs:
- Install patchelf and binutils (for readelf verification) in each final runtime stage where the patch is executed:
  apt-get install -y --no-install-recommends patchelf binutils

3) Add a robust patch and verification step at the end of each final runtime stage, after all pip installs and after any copy of site-packages into the runtime stage.

Use this exact RUN snippet (in each runtime stage):

  RUN python - <<'PY'
  import glob, site, subprocess, sys
  root = site.getsitepackages()[0]
  libs = glob.glob(root + "/**/libctranslate2*.so*", recursive=True)
  print("Found libctranslate2 libs:", len(libs))
  if not libs:
      raise SystemExit("No libctranslate2*.so* found; expected ctranslate2 to be installed")
  for p in libs:
      subprocess.check_call(["patchelf", "--clear-execstack", p])
      out = subprocess.check_output(["readelf", "-W", "-l", p], text=True)
      gnu = next((l for l in out.splitlines() if "GNU_STACK" in l), "")
      print(p, gnu)
      if "RWE" in gnu:
          raise SystemExit(f"GNU_STACK still RWE for {p}")
  PY

4) Add a final runtime sanity import in each runtime stage (after the patch):
  RUN python -c "import ctranslate2; print('ctranslate2', ctranslate2.__version__)"

If this import fails, the docker build must fail.

5) If the Dockerfile installs python deps in a builder stage and then copies site-packages into runtime:
- Run the patch snippet in the runtime stage after the copy as well.
- The verification import must be in the runtime stage.

B) CI: remove tag races by running smoke on the exact image digest
1) Locate the workflow that builds and pushes GHCR images (name may differ), for example:
- .github/workflows/docker-build-and-push.yml

2) In that workflow:
- Ensure the build step uses docker/build-push-action and has an id, for example id: build
- After the push step, run smoke using the produced digest:

  IMAGE="ghcr.io/alexbomber12/lan-transcriber@${{ steps.build.outputs.digest }}"
  docker pull "$IMAGE"
  docker run --rm "$IMAGE" python -c "import ctranslate2; print('ok')"

If you already use tests/test_docker_smoke.py, run it against this exact image:
  python -m pip install -U pytest requests
  SMOKE_IMAGE="$IMAGE" python -m pytest -q tests/test_docker_smoke.py

3) In the unit test workflow (ci.yml):
- Remove docker-smoke steps that pull :main, or gate them off so they do not run by default.
- There must be exactly 1 docker-smoke path by default, and it must run on the build digest.

C) Verification
1) Local reproduction against GHCR after merge:
- docker pull ghcr.io/alexbomber12/lan-transcriber:main
- docker run --rm ghcr.io/alexbomber12/lan-transcriber:main python -c "import ctranslate2; print('ok')"

2) GH Actions:
- The build workflow runs smoke using the digest and is green.

Success criteria
- GHCR :main image imports ctranslate2 successfully on the user machine and GH runners.
- Docker build fails if GNU_STACK remains RWE for any libctranslate2*.so*.
- Docker smoke runs against the exact built image digest; tag races are eliminated.
- CI is green.