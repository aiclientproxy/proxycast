# npm releases

Use `build_npm_package.py` to stage the lightweight root package or one native
platform package. For example, to stage the root package for version `1.140.0`:

```bash
python3 packages/cli/scripts/build_npm_package.py \
  --release-version 1.140.0 \
  --package lime \
  --staging-dir /tmp/lime-npm-root
```

Platform packages require `--vendor-src` to point to a prehydrated vendor tree:

```text
vendor/<target-triple>/bin/
  lime[.exe]
  app-server[.exe]
  code-mode-host[.exe]
  windows-sandbox-setup.exe      # Windows only
  windows-sandbox-runner.exe     # Windows only
  <required runtime libraries>
```

For package-specific debugging, add `--pack-output <path>` to produce an npm
tarball. Release packaging is owned by `.github/workflows/release.yml`: it
builds and uploads every supported platform package, publishes those versions
serially under platform tags, and publishes the root `@limecloud/lime` package
last.

Keep the package catalog in this script, `bin/lime.js`, the release workflow,
and `packages/cli/tests/npm-package.test.mjs` synchronized. A platform without
real build and runtime evidence must not be added as an empty optional package.
