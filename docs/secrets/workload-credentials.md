# Giving a workload a credential

> **Audience:** anyone whose step needs to authenticate to something at run time — clone a private
> repo, pull a gated model, call an internal API — and who wants the credential to reach the job
> without being committed, baked into an image, or pasted into a shell.
>
> Covers SkyPilot environments (LSF/BlueVela, Slurm, AWS, Kubernetes-via-SkyPilot). The plain
> Kubernetes environment class names secrets explicitly instead; see [k8s.md](../environments/k8s.md).

## The mechanism, in one sentence

**Every secret in the space's secret manager is injected into the job as an environment variable under
its own name.** A step that needs `$GITHUB_IBM_PAT` needs a space secret called `GITHUB_IBM_PAT` — no
declaration, no `secret_refs`, no plumbing in the `build.yaml`.

The chain, if you need to follow it:

| Step | Code |
|------|------|
| the build asks the space for its secrets | `build/build.py:465` — `secrets=self.space.get_secrets()` |
| they reach the target, then the environment | `build/target.py:121`, `environment/environment.py:263` |
| the launcher copies them verbatim into the job env | `environment/skypilot.py:908-911` — `env_vars.update(self.secrets)` |

Because the last step is `dict.update` with no prefixing or filtering, the env var name **is** the
secret name. That also means all of a space's secrets are visible to every job it runs — scope a
space's secrets accordingly.

## Procedure

### 1. Mint the credential

For `github.ibm.com`, a **classic** personal access token:

1. <https://github.ibm.com/settings/tokens> → **Generate new token (classic)**
   (via the UI: avatar → Settings → Developer settings → Personal access tokens → Tokens (classic))
2. Scope: **`repo`** is sufficient to clone a private repo. Add `read:org` only if the workload also
   queries org membership.
3. Set an **expiry** you will actually notice, and note it somewhere — an expired token surfaces as a
   mid-build clone failure, not as a warning.

Do not reuse a token of unknown origin. A `ghp_` token is 40 characters on both `github.com` and
`github.ibm.com`, so the format tells you nothing about which host issued it; the only way to find out
is an authenticated API call, and a token offered to the wrong host leaves a failed-auth record there.
To test one you already have, query the *expected* host first:

```bash
# prints only the status and login; token goes via stdin, so it stays out of `ps`
printf 'silent\nurl = "https://github.ibm.com/api/v3/user"\nheader = "Authorization: Bearer %s"\n' "$TOKEN" \
  | curl --config - -w '\nHTTP %{http_code}\n' | grep -E '"login"|HTTP'
```

### 2. Store it as a space secret

Write the value to a file first and pass `--from-file`, so it never appears in argv, in `ps`, or in
your shell history:

```bash
umask 077
printf '%s' 'PASTE_TOKEN_HERE' > /tmp/pat.txt     # or use an editor
export GB_ENVIRONMENT=STANDALONE
gb secret create GITHUB_IBM_PAT --space public --from-file /tmp/pat.txt
shred -u /tmp/pat.txt
```

`--space public` is the `name:` from your `space.yaml`, not the directory — for
`configurations/spaces/local/space.yaml` that name is `public`. Use `--personal` instead for a
per-user secret, which the space merges over its own at fetch time.

Verify, without revealing the value:

```bash
gb secret list --space public          # names only
```

### 3. Consume it in the step

Read it as an ordinary environment variable. The pattern used by
`steps/autotunex-tune/step.yaml`, which clones a private repo:

```bash
set +x                                  # belt and braces: keep it out of any trace output
if [ -n "${GITHUB_IBM_PAT:-}" ]; then
  CLONE_AUTH="https://oauth2:${GITHUB_IBM_PAT}@${REPO_URL}"
else
  echo "WARNING: GITHUB_IBM_PAT unset - attempting unauthenticated clone"
  CLONE_AUTH="https://${REPO_URL}"
fi
git clone --quiet "$CLONE_AUTH" "$CLONE_DIR" 2>&1 | sed 's#//[^@]*@#//***@#g'
unset CLONE_AUTH
```

Three things that matter here and are easy to get wrong:

- **Pipe git's output through a redacting `sed`.** Git echoes the remote URL on some failures, which
  would put the token in the job log — and job logs are retrieved and stored as build events.
- **Do not use `git config --global url.…insteadOf`.** It persists the credential into the container's
  home directory, where it outlives the step.
- **Use `${VAR:-}` under `set -u`**, so an unset secret produces your own diagnostic rather than an
  unbound-variable abort.

## When a restart is needed

| Backend | New/changed secret picked up by |
|---------|-------------------------------|
| `local` | the next build — `LocalSpaceSecretManager._load_all_secrets` reads the file on each `get_secrets()` call, and the build calls it at start |
| `env` | **a gbserver restart** — the value comes from the server process's own environment (`GBSERVER_SECRET_<NAME>`), which is fixed at exec time |
| `ibmcloud` | the next build (read-only lookup) |

## Gotchas

- **The on-disk file is group- and world-readable.** `gb secret create` writes
  `<gb_home>/space_secrets/<space>.yaml` (default `~/.granite.build/space_secrets/`) with mode `644`,
  created by the *server* process, so your shell's `umask` does not apply. Tighten it:
  ```bash
  chmod 700 ~/.granite.build/space_secrets && chmod 600 ~/.granite.build/space_secrets/*.yaml
  ```
- **Values on disk are base64, not encrypted.** Base64 is obfuscation, not protection — the file
  deserves the same handling as a private key. See
  [local-secrets-manager.md](local-secrets-manager.md#local-secrets-file-structure) for both supported
  on-disk shapes (`gb secret create` writes the flat `<name>: <base64>` form).
- **Deleting the last secret leaves the file behind** containing `{}`. Harmless; `gb secret list`
  correctly reports nothing.
- **Never bake a credential into an image.** A token in a `Dockerfile` `ARG` + `git config --global`
  cannot be rotated without a rebuild, expires silently, and is readable by anyone who can pull the
  image. It is also unreliable under enroot: the LSF provisioner sets `ENROOT_MOUNT_HOME=false` and
  starts the container without `--root`, so `$HOME` inside the container is not the image's `/root`,
  and a `--global` git config written at build time may never be read.

## See also

- [Secrets overview](README.md) — choosing a backend
- [local-secrets-manager.md](local-secrets-manager.md) — file layout, remote sync
- [env-secrets-manager.md](env-secrets-manager.md) — `GBSERVER_SECRET_<NAME>`, for CI and containers
- [skypilot-lsf.md](../environments/skypilot-lsf.md) — the BlueVela environment
