#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/release.sh [build|testpypi|pypi] [--yes]

Build and validate PyConnviz distributions, then optionally upload the exact
artifacts created by this run.

Actions:
  build       Build and validate only (default).
  testpypi    Build, validate, and upload to TestPyPI.
  pypi        Build, validate, and upload to production PyPI.

Options:
  --yes       Skip the production PyPI confirmation. Valid only with pypi.
  -h, --help  Show this help message.

Environment:
  PYCONNVIZ_PYTHON  Python executable to use. Defaults to .venv/bin/python
                    when available, otherwise python.

Twine reads credentials from its normal environment variables, keyring, or
configuration files. This script never stores credentials.
EOF
}

action="build"
action_seen=false
assume_yes=false

for argument in "$@"; do
    case "$argument" in
        build|testpypi|pypi)
            if [[ "$action_seen" == true ]]; then
                printf 'Only one action may be specified.\n' >&2
                usage >&2
                exit 2
            fi
            action="$argument"
            action_seen=true
            ;;
        --yes)
            assume_yes=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$assume_yes" == true && "$action" != "pypi" ]]; then
    printf '%s\n' '--yes is valid only with the pypi action.' >&2
    exit 2
fi

if [[ "$action" == "pypi" && "$assume_yes" != true ]]; then
    if [[ ! -t 0 ]]; then
        printf '%s\n' \
            'Production upload requires an interactive confirmation or explicit --yes.' >&2
        exit 2
    fi
    printf '%s' 'Type "release pyconnviz" to upload to production PyPI: ' >&2
    read -r confirmation
    if [[ "$confirmation" != "release pyconnviz" ]]; then
        printf '%s\n' 'Production upload cancelled.' >&2
        exit 1
    fi
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"

if [[ -n "${PYCONNVIZ_PYTHON:-}" ]]; then
    python_bin="$PYCONNVIZ_PYTHON"
elif [[ -x "${project_root}/.venv/bin/python" ]]; then
    python_bin="${project_root}/.venv/bin/python"
else
    python_bin="python"
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 1
fi

if ! "$python_bin" -c 'import build, twine' >/dev/null 2>&1; then
    printf '%s\n' \
        'Missing release tools. Run: python -m pip install -e ".[dev]"' >&2
    exit 1
fi

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/pyconnviz-release.XXXXXX")"
cleanup() {
    rm -rf -- "$staging_dir"
}
trap cleanup EXIT

printf 'Building PyConnviz with %s\n' "$python_bin"
"$python_bin" -m build --outdir "$staging_dir" "$project_root"

shopt -s nullglob
wheels=("$staging_dir"/*.whl)
sdists=("$staging_dir"/*.tar.gz)
shopt -u nullglob

if (( ${#wheels[@]} != 1 || ${#sdists[@]} != 1 )); then
    printf 'Expected one wheel and one sdist; found %d wheel(s) and %d sdist(s).\n' \
        "${#wheels[@]}" "${#sdists[@]}" >&2
    exit 1
fi

artifacts=("${sdists[@]}" "${wheels[@]}")
"$python_bin" -m twine check "${artifacts[@]}"

dist_dir="${project_root}/dist"
mkdir -p -- "$dist_dir"
cp -- "${artifacts[@]}" "$dist_dir/"

printf '%s\n' 'Validated artifacts copied to dist:'
for artifact in "${artifacts[@]}"; do
    printf '  %s\n' "${artifact##*/}"
done

case "$action" in
    build)
        printf '%s\n' 'Build complete; no upload performed.'
        ;;
    testpypi)
        printf '%s\n' 'Uploading current artifacts to TestPyPI...'
        "$python_bin" -m twine upload --repository testpypi "${artifacts[@]}"
        ;;
    pypi)
        printf '%s\n' 'Uploading current artifacts to production PyPI...'
        "$python_bin" -m twine upload --repository pypi "${artifacts[@]}"
        ;;
esac
