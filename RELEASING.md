# Releasing ThinTensor

ThinTensor has two release artifacts:

* the Python runtime, published to PyPI;
* the native `thintensor-core` binary, published as the `thintensor` crate.

The Cargo package is deliberately limited to `src/`, the manifests, the lock
file, and the README. Benchmark outputs, model weights, and local artifacts
must not be included in the crate.

## Publish the native core

1. Update the version in `Cargo.toml` and commit the matching `Cargo.lock`.
   Cargo versions are immutable, so every published version must be new.
2. Run the checks from a clean checkout:

   ```bash
   cargo fmt --check
   cargo test --locked
   cargo package --locked
   cargo publish --dry-run --locked
   ```

3. Authenticate locally with a crates.io API token. Do not put the token in
   the repository or send it in chat:

   ```bash
   cargo login
   ```

4. Publish:

   ```bash
   cargo publish --locked
   ```

5. Test the shipped artifact from a directory outside the checkout:

   ```bash
   cargo install --force thintensor --version <version> --locked
   command -v thintensor-core
   thintensor-core --help
   ```

Push the repository changes and update the project submission with the crates.io
link and the registry-only installation commands from `README.md`.
