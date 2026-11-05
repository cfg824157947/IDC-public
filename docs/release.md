# Publish v0.1.0 (commands only; not executed)

The local repository has branch main and annotated tag v0.1.0. Its configured
origin is https://github.com/cfg824157947/IDC. When approved, ensure that this
repository exists on GitHub and is empty. Authenticate using GitHub's supported
login tools, not credentials stored in files.

From the prepared IDC repository:

```bash
git status --short
git show --no-patch v0.1.0
git remote -v
git push --atomic origin main refs/tags/v0.1.0
```

If downloading the accompanying bundle instead, reconstruct the same history:

```bash
git clone IDC_v0.1.0.bundle IDC
cd IDC
git remote set-url origin https://github.com/cfg824157947/IDC
git show --no-patch v0.1.0
git push --atomic origin main refs/tags/v0.1.0
```

If the remote is not empty, inspect and reconcile its history first; do not force
push. A source ZIP excludes Git history: use the bundle to preserve the exact
commit and tag. Check signed-out public access before citing the repository as
publicly available. No publication commands have been executed by this audit.
